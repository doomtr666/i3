extern crate nalgebra_glm;

use examples_common::basic_scene::BasicScene;
use examples_common::{ExampleApp, init_tracing, main_loop};
use i3_egui::egui;
use i3_gfx::graph::types::*;
use i3_gfx::prelude::*;
use i3_io::asset::AssetLoader;
use i3_io::material::MaterialAsset;
use i3_io::mesh::MeshAsset;
use i3_io::scene_asset::SceneAsset;
use i3_io::texture::{TextureAsset, TextureFormat};
use i3_io::vfs::{BundleBackend, Vfs};
use i3_renderer::passes::debug_viz::DebugChannel;
use i3_renderer::render_graph::{DefaultRenderGraph, RenderConfig};
use i3_renderer::scene::SceneProvider;
use i3_vulkan_backend::VulkanBackend;
use nalgebra_glm as glm;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use uuid::Uuid;

struct DeferredGltfApp {
    backend: VulkanBackend,
    window: WindowHandle,
    render_graph: DefaultRenderGraph,
    ui: Arc<i3_egui::UiSystem>,
    scene: BasicScene,
    loader: Arc<i3_io::asset::AssetLoader>,
    time: f32,
    dt: f32,
    smoothed_dt: f32,
    camera: examples_common::camera_controller::CameraController,
    is_fullscreen: bool,
    current_scene: String,
    frame_time_history: VecDeque<f32>,
    show_perf_graph: bool,
    sample_accum_time: f32,
    sample_max_dt: f32,
}

impl DeferredGltfApp {
    fn load_scene(&mut self, scene_name: &str) {
        let start = Instant::now();
        info!("Loading SceneAsset '{}'...", scene_name);

        let scene_handle = self.loader.load::<SceneAsset>(scene_name);
        let scene_asset = match scene_handle.wait_loaded() {
            Ok(asset) => asset,
            Err(e) => {
                warn!("Failed to load scene '{}': {}", scene_name, e);
                return;
            }
        };

        // Reset scene
        self.scene = BasicScene::new();
        self.current_scene = scene_name.to_string();

        // Collect distinct materials required by meshes
        let mut required_materials = std::collections::HashSet::new();

        // Load all meshes referenced by the scene
        for mesh_uuid in &scene_asset.mesh_refs {
            let mesh_handle = match self.loader.load_by_uuid::<MeshAsset>(mesh_uuid) {
                Ok(h) => h,
                Err(_) => continue,
            };
            let mesh_asset = match mesh_handle.wait_loaded() {
                Ok(a) => a,
                Err(_) => continue,
            };

            let mat_uuid = uuid::Uuid::from_bytes(mesh_asset.header.material_id);
            if !mat_uuid.is_nil() {
                required_materials.insert(mat_uuid);
            }

            self.scene
                .add_baked_mesh(&mut self.backend, &mesh_asset, *mesh_uuid);
        }

        // Load unique materials and their textures
        let loader_arc = self.loader.clone();
        for mat_uuid in required_materials {
            if let Ok(mat_handle) = self.loader.load_by_uuid::<MaterialAsset>(&mat_uuid) {
                if let Ok(mat_asset) = mat_handle.wait_loaded() {
                    let mut texture_loader = |tex_uuid: &Uuid,
                                              be: &mut VulkanBackend|
                     -> Option<ImageHandle> {
                        if let Ok(tex_handle) = loader_arc.load_by_uuid::<TextureAsset>(tex_uuid) {
                            if let Ok(tex_asset) = tex_handle.wait_loaded() {
                                let width = tex_asset.header.width;
                                let height = tex_asset.header.height;
                                let mips = tex_asset.header.mip_levels;

                                let format = match tex_asset.header.format {
                                    f if f == TextureFormat::BC7_SRGB as u32 => Format::BC7_SRGB,
                                    f if f == TextureFormat::BC7_UNORM as u32 => Format::BC7_UNORM,
                                    f if f == TextureFormat::BC5_UNORM as u32 => Format::BC5_UNORM,
                                    f if f == TextureFormat::BC1_RGB_SRGB as u32 => {
                                        Format::BC1_RGB_SRGB
                                    }
                                    f if f == TextureFormat::BC1_RGB_UNORM as u32 => {
                                        Format::BC1_RGB_UNORM
                                    }
                                    f if f == TextureFormat::BC3_SRGB as u32 => Format::BC3_SRGB,
                                    f if f == TextureFormat::BC3_UNORM as u32 => Format::BC3_UNORM,
                                    _ => Format::R8G8B8A8_SRGB,
                                };

                                let image = be.create_image(&ImageDesc {
                                    width,
                                    height,
                                    depth: 1,
                                    format,
                                    usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
                                    mip_levels: mips as u32,
                                    array_layers: 1,
                                    view_type: ImageViewType::Type2D,
                                    swizzle: Default::default(),
                                });

                                let handle = ImageHandle(SymbolId(image.0));

                                let mut current_offset = 0;
                                for mip in 0..mips {
                                    let mip_width = (width >> mip).max(1);
                                    let mip_height = (height >> mip).max(1);

                                    let blocks_x = (mip_width + 3) / 4;
                                    let blocks_y = (mip_height + 3) / 4;

                                    let bpb = match format {
                                        Format::BC1_RGB_SRGB | Format::BC1_RGB_UNORM => 8,
                                        Format::R8G8B8A8_SRGB | Format::R8G8B8A8_UNORM => 0, // Not block based
                                        _ => 16,
                                    };

                                    let mip_size = if bpb == 0 {
                                        (mip_width * mip_height * 4) as usize
                                    } else {
                                        (blocks_x * blocks_y) as usize * bpb
                                    };

                                    if current_offset + mip_size <= tex_asset.data.len() {
                                        let _ = be.upload_image(
                                            image,
                                            &tex_asset.data[current_offset..current_offset + mip_size],
                                            0,
                                            0,
                                            mip_width as u32,
                                            mip_height as u32,
                                            mip as u32,
                                            0,
                                        );
                                        current_offset += mip_size;
                                    }
                                }
                                return Some(handle);
                            }
                        }
                        None
                    };

                    self.scene.add_baked_material(
                        &mut self.backend,
                        &mut self.render_graph.bindless_manager,
                        &mat_asset,
                        mat_uuid,
                        &mut texture_loader,
                    );
                }
            }
        }

        // Populate scene objects/lights
        self.scene.load_baked_scene(&scene_asset);
        if SceneProvider::light_count(&self.scene) == 0 {
            self.scene.add_default_lights();
        }

        // Adjust camera for scene scale
        let scene_diag = self.scene.bounds().diagonal_length();
        self.camera.move_speed = (scene_diag * 0.2).max(1.0);

        if scene_name.contains("Sponza") {
            self.camera.position = glm::vec3(0.0, 2.0, 0.0);
        } else if scene_name.contains("BistroExterior") {
            self.camera.position = glm::vec3(-15.0, 2.0, 0.0);
        } else {
            self.camera.position = glm::vec3(0.0, 0.0, (scene_diag * 0.8).max(3.0));
        }

        let duration = start.elapsed();
        info!(
            "Scene '{}' loaded in {:.2}s",
            scene_name,
            duration.as_secs_f32()
        );
    }

    fn draw_performance_graph(&self, ui: &mut egui::Ui) {
        use egui::{Color32, Pos2, Stroke, RichText};

        let graph_height = 80.0;
        let graph_width = 320.0;
        let max_samples = 256;

        if self.frame_time_history.is_empty() {
            return;
        }

        // --- 1. Frame Time Graph (ms) ---
        ui.label(RichText::new("Frame Time (ms)").strong().color(Color32::LIGHT_BLUE));
        let (rect_ms, _) = ui.allocate_at_least(egui::vec2(graph_width, graph_height), egui::Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(rect_ms, 2.0, Color32::from_black_alpha(180));

        let mut actual_max_ms: f32 = 0.1;
        let mut min_ms: f32 = 1000.0;
        for &dt in &self.frame_time_history {
            let ms = dt * 1000.0;
            actual_max_ms = actual_max_ms.max(ms);
            min_ms = min_ms.min(ms);
        }
        
        // Round max_ms to a "nice" number for stable Y-axis
        let max_ms = if actual_max_ms < 1.0 { 1.0 }
                    else if actual_max_ms < 2.0 { 2.0 }
                    else if actual_max_ms < 5.0 { 5.0 }
                    else if actual_max_ms < 10.0 { 10.0 }
                    else if actual_max_ms < 20.0 { 20.0 }
                    else if actual_max_ms < 50.0 { 50.0 }
                    else { (actual_max_ms / 10.0).ceil() * 10.0 };

        let draw_line_ms = |ms: f32, color: Color32, label: &str| {
            let y = rect_ms.bottom() - (ms / max_ms) * graph_height;
            if y > rect_ms.top() && y <= rect_ms.bottom() {
                painter.line_segment([Pos2::new(rect_ms.left(), y), Pos2::new(rect_ms.right(), y)], Stroke::new(1.0, color.linear_multiply(0.5)));
                painter.text(Pos2::new(rect_ms.right() - 5.0, y - 2.0), egui::Align2::RIGHT_BOTTOM, label, egui::FontId::monospace(9.0), color);
            }
        };

        draw_line_ms(16.6, Color32::GREEN, "16.6ms");
        draw_line_ms(33.3, Color32::GOLD, "33.3ms");
        if max_ms < 5.0 {
            draw_line_ms(1.0, Color32::from_rgb(100, 100, 255), "1.0ms");
            draw_line_ms(0.5, Color32::from_rgb(150, 150, 255), "0.5ms");
        }

        let mut points_ms = Vec::with_capacity(self.frame_time_history.len());
        for (i, &dt) in self.frame_time_history.iter().enumerate() {
            let x = rect_ms.left() + (i as f32 / (max_samples - 1) as f32) * graph_width;
            let y = rect_ms.bottom() - (dt * 1000.0 / max_ms) * graph_height;
            points_ms.push(Pos2::new(x, y.clamp(rect_ms.top(), rect_ms.bottom())));
        }
        if points_ms.len() > 1 {
            painter.add(egui::Shape::line(points_ms, Stroke::new(1.0, Color32::LIGHT_BLUE)));
        }
        painter.text(Pos2::new(rect_ms.left() + 5.0, rect_ms.top() + 5.0), egui::Align2::LEFT_TOP, format!("max: {:.2}ms", actual_max_ms), egui::FontId::monospace(10.0), Color32::WHITE);

        ui.add_space(8.0);

        // --- 2. FPS Graph ---
        ui.label(RichText::new("FPS").strong().color(Color32::LIGHT_YELLOW));
        let (rect_fps, _) = ui.allocate_at_least(egui::vec2(graph_width, graph_height), egui::Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(rect_fps, 2.0, Color32::from_black_alpha(180));

        let mut actual_max_fps: f32 = 60.0;
        for &dt in &self.frame_time_history {
            if dt > 0.0 { actual_max_fps = actual_max_fps.max(1.0 / dt); }
        }
        
        // Stable FPS scale
        let max_fps = if actual_max_fps < 150.0 { 150.0 }
                     else if actual_max_fps < 300.0 { 300.0 }
                     else if actual_max_fps < 1000.0 { 1000.0 }
                     else if actual_max_fps < 2000.0 { 2000.0 }
                     else if actual_max_fps < 5000.0 { 5000.0 }
                     else { (actual_max_fps / 1000.0).ceil() * 1000.0 };

        let draw_line_fps = |fps: f32, color: Color32, label: &str| {
            let y = rect_fps.bottom() - (fps / max_fps) * graph_height;
            if y > rect_fps.top() && y <= rect_fps.bottom() {
                painter.line_segment([Pos2::new(rect_fps.left(), y), Pos2::new(rect_fps.right(), y)], Stroke::new(1.0, color.linear_multiply(0.5)));
                painter.text(Pos2::new(rect_fps.right() - 5.0, y - 2.0), egui::Align2::RIGHT_BOTTOM, label, egui::FontId::monospace(9.0), color);
            }
        };

        draw_line_fps(60.0, Color32::GREEN, "60");
        draw_line_fps(144.0, Color32::from_rgb(0, 200, 200), "144");
        if max_fps > 1000.0 { draw_line_fps(1000.0, Color32::from_rgb(200, 200, 0), "1k"); }
        if max_fps > 2500.0 { draw_line_fps(2500.0, Color32::from_rgb(200, 100, 0), "2.5k"); }

        let mut points_fps = Vec::with_capacity(self.frame_time_history.len());
        for (i, &dt) in self.frame_time_history.iter().enumerate() {
            let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
            let x = rect_fps.left() + (i as f32 / (max_samples - 1) as f32) * graph_width;
            let y = rect_fps.bottom() - (fps / max_fps) * graph_height;
            points_fps.push(Pos2::new(x, y.clamp(rect_fps.top(), rect_fps.bottom())));
        }
        if points_fps.len() > 1 {
            painter.add(egui::Shape::line(points_fps, Stroke::new(1.0, Color32::LIGHT_YELLOW)));
        }
        painter.text(Pos2::new(rect_fps.left() + 5.0, rect_fps.top() + 5.0), egui::Align2::LEFT_TOP, format!("max: {:.0} FPS", actual_max_fps), egui::FontId::monospace(10.0), Color32::WHITE);

        ui.add_space(4.0);
        if let Some(&last) = self.frame_time_history.back() {
            ui.label(format!("current: {:.2}ms ({:.0} FPS)", last * 1000.0, 1.0 / last));
        }
    }
}

impl ExampleApp for DeferredGltfApp {
    fn update(&mut self, delta: Duration, smoothed_delta: Duration) {
        self.dt = delta.as_secs_f32();
        self.smoothed_dt = smoothed_delta.as_secs_f32();
        self.time += self.dt;

        // Accumulate for temporal graph stability (record max dt over 20ms interval)
        self.sample_accum_time += self.dt;
        self.sample_max_dt = self.sample_max_dt.max(self.dt);

        if self.sample_accum_time >= 0.020 { // 50Hz sampling = ~5s history for 256 samples
            self.frame_time_history.push_back(self.sample_max_dt);
            if self.frame_time_history.len() > 256 {
                self.frame_time_history.pop_front();
            }
            self.sample_accum_time = 0.0;
            self.sample_max_dt = 0.0;
        }

        self.camera.update(delta);
    }

    fn render(&mut self) {
        // --- Egui UI Definition ---
        self.ui.begin_frame();
        let egui_ctx = self.ui.context().clone();

        let mut scene_to_load = None;

        egui::Window::new("Engine Debug").show(&egui_ctx, |ui| {
            ui.heading("Renderer");
            ui.label(format!(
                "Frame time: {:.2}ms ({:.1} FPS)",
                self.smoothed_dt * 1000.0,
                1.0 / self.smoothed_dt
            ));
            ui.checkbox(&mut self.show_perf_graph, "Show Performance Graph");
            ui.separator();

            ui.label("Scene:");
            egui::ComboBox::from_label("Select Scene")
                .selected_text(&self.current_scene)
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_label(self.current_scene == "Sponza_scene", "Sponza")
                        .clicked()
                    {
                        scene_to_load = Some("Sponza_scene");
                    }
                    if ui
                        .selectable_label(
                            self.current_scene == "BistroExterior_scene",
                            "Bistro Exterior",
                        )
                        .clicked()
                    {
                        scene_to_load = Some("BistroExterior_scene");
                    }
                    if ui
                        .selectable_label(
                            self.current_scene == "BistroInterior_scene",
                            "Bistro Interior",
                        )
                        .clicked()
                    {
                        scene_to_load = Some("BistroInterior_scene");
                    }
                    if ui
                        .selectable_label(
                            self.current_scene == "DamagedHelmet_scene",
                            "Damaged Helmet",
                        )
                        .clicked()
                    {
                        scene_to_load = Some("DamagedHelmet_scene");
                    }
                    if ui
                        .selectable_label(
                            self.current_scene == "NormalTangentTest_scene",
                            "Normal Tangent Test",
                        )
                        .clicked()
                    {
                        scene_to_load = Some("NormalTangentTest_scene");
                    }
                    if ui
                        .selectable_label(
                            self.current_scene == "NormalTangentMirrorTest_scene",
                            "Normal Tangent Mirror",
                        )
                        .clicked()
                    {
                        scene_to_load = Some("NormalTangentMirrorTest_scene");
                    }
                });

            ui.separator();
            ui.label("Debug Channel:");
            ui.radio_value(
                &mut self.render_graph.debug_channel,
                DebugChannel::Lit,
                "Lit (Final)",
            );
            ui.radio_value(
                &mut self.render_graph.debug_channel,
                DebugChannel::Albedo,
                "Albedo",
            );
            ui.radio_value(
                &mut self.render_graph.debug_channel,
                DebugChannel::Normal,
                "Normals",
            );
            ui.radio_value(
                &mut self.render_graph.debug_channel,
                DebugChannel::Roughness,
                "Roughness",
            );
            ui.radio_value(
                &mut self.render_graph.debug_channel,
                DebugChannel::Metallic,
                "Metallic",
            );
            ui.radio_value(
                &mut self.render_graph.debug_channel,
                DebugChannel::Emissive,
                "Emissive",
            );
            ui.radio_value(
                &mut self.render_graph.debug_channel,
                DebugChannel::Depth,
                "Depth",
            );
        });

        if self.show_perf_graph {
            egui::Window::new("Performance Graph").show(&egui_ctx, |ui| {
                self.draw_performance_graph(ui);
            });
        }

        if let Some(name) = scene_to_load {
            self.load_scene(name);
        }

        // Finalize UI and update textures before recording the graph
        self.ui.update_textures(&mut self.backend);

        let view = self.camera.view_matrix();
        let (width, height) = self.backend.window_size(self.window).unwrap_or((1280, 720));
        let scene_diag = self.scene.bounds().diagonal_length();
        let far = (scene_diag * 3.0).max(1000.0);
        let near = 0.1;

        let projection = glm::perspective_rh_zo(
            width as f32 / height as f32,
            std::f32::consts::FRAC_PI_4,
            near,
            far,
        );

        self.render_graph.sync(&mut self.backend, &self.scene);

        let mut graph = FrameGraph::new();
        graph.publish("UiSystem", self.ui.clone());
        graph.publish("AssetLoader", self.loader.clone());

        self.render_graph.record(
            &mut graph,
            self.window,
            &self.scene,
            view,
            projection,
            near,
            far,
            width,
            height,
            self.dt,
        );

        let compiler = graph.compile();
        if let Err(e) = compiler.execute(
            &mut self.backend,
            Some(&mut self.render_graph.temporal_registry),
        ) {
            warn!("Graph execution failed: {}", e);
        }
    }

    fn poll_events(&mut self) -> Vec<Event> {
        self.backend.poll_events()
    }

    fn handle_event(&mut self, event: &Event) {
        self.ui.handle_event(event);

        if let Event::KeyDown { key } = event {
            if *key == KeyCode::F11 {
                self.is_fullscreen = !self.is_fullscreen;
                self.backend.set_fullscreen(self.window, self.is_fullscreen);
            }
        }

        // Only let camera handle event if egui doesn't want it
        let wants_input =
            self.ui.context().wants_pointer_input() || self.ui.context().wants_keyboard_input();

        if !wants_input {
            self.camera.handle_event(event);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = init_tracing("viewer.log");
    info!("Starting I3 viewer demo application");

    // 1. Initialize Backend
    let mut backend = VulkanBackend::new()?;
    examples_common::maybe_list_gpus(&backend);
    backend.initialize(examples_common::get_gpu_index())?;

    // 2. Create Window
    let window = backend.create_window(WindowDesc {
        title: "Deferred glTF (Baked)".to_string(),
        width: 1280,
        height: 720,
    })?;

    // 3. Setup IO and VFS
    let assets_dir = if let Ok(exe_path) = std::env::current_exe() {
        let exe_dir = exe_path.parent().unwrap();
        if exe_dir.join("viewer_scenes.i3b").exists() {
            exe_dir.to_path_buf()
        } else {
            PathBuf::from("assets")
        }
    } else {
        PathBuf::from("assets")
    };

    let blob_path = assets_dir.join("viewer_scenes.i3b");
    let catalog_path = assets_dir.join("viewer_scenes.i3c");

    info!("Mounting bundle from {:?}", assets_dir);
    let bundle = BundleBackend::mount(&catalog_path, &blob_path)?;
    let vfs = Vfs::new();
    vfs.mount(Box::new(bundle));
    let vfs = Arc::new(vfs);
    let loader = AssetLoader::new(vfs);
    let loader_arc = Arc::new(loader);

    let config = RenderConfig {
        width: 1280,
        height: 720,
    };

    let ui = Arc::new(i3_egui::UiSystem::new(config.width, config.height));
    let mut render_graph = DefaultRenderGraph::new(&mut backend, &config);
    render_graph.publish("UiSystem", ui.clone());
    render_graph.publish("AssetLoader", loader_arc.clone());
    render_graph.init(&mut backend);

    let scene_name = std::env::var("I3_SCENE").unwrap_or_else(|_| "Sponza_scene".to_string());

    let mut app = DeferredGltfApp {
        backend,
        window,
        render_graph,
        ui,
        scene: BasicScene::new(),
        loader: loader_arc.clone(),
        time: 0.0,
        dt: 0.016,
        smoothed_dt: 0.016,
        camera: examples_common::camera_controller::CameraController::new(),
        is_fullscreen: false,
        current_scene: String::new(),
        frame_time_history: VecDeque::with_capacity(2000),
        show_perf_graph: false,
        sample_accum_time: 0.0,
        sample_max_dt: 0.0,
    };

    // Initial load
    app.load_scene(&scene_name);

    main_loop(app);

    Ok(())
}
