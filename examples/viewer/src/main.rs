extern crate nalgebra_glm;

use examples_common::basic_scene::BasicScene;
use examples_common::{ExampleApp, init_tracing, main_loop};
use i3_egui::prelude::*;
use i3_gfx::prelude::*;
use i3_io::prelude::*;
use i3_renderer::passes::debug_viz::DebugChannel;
use i3_renderer::prelude::*;
use i3_vulkan_backend::prelude::*;
use nalgebra_glm as glm;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use uuid::Uuid;

// ─── CPU frustum cull (mirrors the GPU shader logic) ─────────────────────────
// Returns true = AABB is (at least partially) inside the frustum.

fn frustum_cull_cpu(min: [f32; 3], max: [f32; 3], vp: &glm::Mat4) -> bool {
    let corners = [
        [min[0], min[1], min[2]],
        [max[0], min[1], min[2]],
        [min[0], max[1], min[2]],
        [max[0], max[1], min[2]],
        [min[0], min[1], max[2]],
        [max[0], min[1], max[2]],
        [min[0], max[1], max[2]],
        [max[0], max[1], max[2]],
    ];
    // For each of the 6 clip planes, if ALL corners are outside → cull.
    for plane in 0..6_usize {
        let all_outside = corners.iter().all(|&[x, y, z]| {
            let c = vp * glm::vec4(x, y, z, 1.0);
            match plane {
                0 => c.x < -c.w,
                1 => c.x > c.w,
                2 => c.y < -c.w,
                3 => c.y > c.w,
                4 => c.z < 0.0, // near (reverse-Z)
                _ => c.z > c.w, // far
            }
        });
        if all_outside {
            return false;
        }
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────

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
    available_scenes: Vec<String>,
    frame_time_history: VecDeque<f32>,
    show_perf_graph: bool,
    sample_accum_time: f32,
    sample_max_dt: f32,
    show_culling_debug: bool,
    culling_show_ids: bool,
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

        // Reset scene and render graph scene-specific state (frees GPU buffers + AS)
        self.render_graph
            .clear_scene(&mut self.backend, &mut self.scene);
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
                                    clear_value: None,
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
                                            &tex_asset.data
                                                [current_offset..current_offset + mip_size],
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
        let obj_count = self.scene.load_baked_scene(&scene_asset);
        tracing::info!(
            "Loaded scene '{}': {} objects, {} mesh refs",
            scene_name,
            obj_count,
            scene_asset.mesh_refs.len()
        );
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
        use egui::{Color32, Pos2, RichText, Stroke};

        let graph_height = 80.0;
        let graph_width = 320.0;
        let max_samples = 256;

        if self.frame_time_history.is_empty() {
            return;
        }

        // --- 1. Frame Time Graph (ms) ---
        ui.label(
            RichText::new("Frame Time (ms)")
                .strong()
                .color(Color32::LIGHT_BLUE),
        );
        let (rect_ms, _) =
            ui.allocate_at_least(egui::vec2(graph_width, graph_height), egui::Sense::hover());
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
        let max_ms = if actual_max_ms < 1.0 {
            1.0
        } else if actual_max_ms < 2.0 {
            2.0
        } else if actual_max_ms < 5.0 {
            5.0
        } else if actual_max_ms < 10.0 {
            10.0
        } else if actual_max_ms < 20.0 {
            20.0
        } else if actual_max_ms < 50.0 {
            50.0
        } else {
            (actual_max_ms / 10.0).ceil() * 10.0
        };

        let draw_line_ms = |ms: f32, color: Color32, label: &str| {
            let y = rect_ms.bottom() - (ms / max_ms) * graph_height;
            if y > rect_ms.top() && y <= rect_ms.bottom() {
                painter.line_segment(
                    [Pos2::new(rect_ms.left(), y), Pos2::new(rect_ms.right(), y)],
                    Stroke::new(1.0, color.linear_multiply(0.5)),
                );
                painter.text(
                    Pos2::new(rect_ms.right() - 5.0, y - 2.0),
                    egui::Align2::RIGHT_BOTTOM,
                    label,
                    egui::FontId::monospace(9.0),
                    color,
                );
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
            painter.add(egui::Shape::line(
                points_ms,
                Stroke::new(1.0, Color32::LIGHT_BLUE),
            ));
        }
        painter.text(
            Pos2::new(rect_ms.left() + 5.0, rect_ms.top() + 5.0),
            egui::Align2::LEFT_TOP,
            format!("max: {:.2}ms", actual_max_ms),
            egui::FontId::monospace(10.0),
            Color32::WHITE,
        );

        ui.add_space(8.0);

        // --- 2. FPS Graph ---
        ui.label(RichText::new("FPS").strong().color(Color32::LIGHT_YELLOW));
        let (rect_fps, _) =
            ui.allocate_at_least(egui::vec2(graph_width, graph_height), egui::Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(rect_fps, 2.0, Color32::from_black_alpha(180));

        let mut actual_max_fps: f32 = 60.0;
        for &dt in &self.frame_time_history {
            if dt > 0.0 {
                actual_max_fps = actual_max_fps.max(1.0 / dt);
            }
        }

        // Stable FPS scale
        let max_fps = if actual_max_fps < 150.0 {
            150.0
        } else if actual_max_fps < 300.0 {
            300.0
        } else if actual_max_fps < 1000.0 {
            1000.0
        } else if actual_max_fps < 2000.0 {
            2000.0
        } else if actual_max_fps < 5000.0 {
            5000.0
        } else {
            (actual_max_fps / 1000.0).ceil() * 1000.0
        };

        let draw_line_fps = |fps: f32, color: Color32, label: &str| {
            let y = rect_fps.bottom() - (fps / max_fps) * graph_height;
            if y > rect_fps.top() && y <= rect_fps.bottom() {
                painter.line_segment(
                    [
                        Pos2::new(rect_fps.left(), y),
                        Pos2::new(rect_fps.right(), y),
                    ],
                    Stroke::new(1.0, color.linear_multiply(0.5)),
                );
                painter.text(
                    Pos2::new(rect_fps.right() - 5.0, y - 2.0),
                    egui::Align2::RIGHT_BOTTOM,
                    label,
                    egui::FontId::monospace(9.0),
                    color,
                );
            }
        };

        draw_line_fps(60.0, Color32::GREEN, "60");
        draw_line_fps(144.0, Color32::from_rgb(0, 200, 200), "144");
        if max_fps > 1000.0 {
            draw_line_fps(1000.0, Color32::from_rgb(200, 200, 0), "1k");
        }
        if max_fps > 2500.0 {
            draw_line_fps(2500.0, Color32::from_rgb(200, 100, 0), "2.5k");
        }

        let mut points_fps = Vec::with_capacity(self.frame_time_history.len());
        for (i, &dt) in self.frame_time_history.iter().enumerate() {
            let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
            let x = rect_fps.left() + (i as f32 / (max_samples - 1) as f32) * graph_width;
            let y = rect_fps.bottom() - (fps / max_fps) * graph_height;
            points_fps.push(Pos2::new(x, y.clamp(rect_fps.top(), rect_fps.bottom())));
        }
        if points_fps.len() > 1 {
            painter.add(egui::Shape::line(
                points_fps,
                Stroke::new(1.0, Color32::LIGHT_YELLOW),
            ));
        }
        painter.text(
            Pos2::new(rect_fps.left() + 5.0, rect_fps.top() + 5.0),
            egui::Align2::LEFT_TOP,
            format!("max: {:.0} FPS", actual_max_fps),
            egui::FontId::monospace(10.0),
            Color32::WHITE,
        );

        ui.add_space(4.0);
        if let Some(&last) = self.frame_time_history.back() {
            ui.label(format!(
                "current: {:.2}ms ({:.0} FPS)",
                last * 1000.0,
                1.0 / last
            ));
        }
    }
}

impl ExampleApp for DeferredGltfApp {
    fn update(&mut self, delta: Duration, smoothed_delta: Duration) {
        self.dt = delta.as_secs_f32();
        self.smoothed_dt = smoothed_delta.as_secs_f32();
        self.time += self.dt;

        // Accumulate for temporal graph stability (declare max dt over 20ms interval)
        self.sample_accum_time += self.dt;
        self.sample_max_dt = self.sample_max_dt.max(self.dt);

        if self.sample_accum_time >= 0.020 {
            // 50Hz sampling = ~5s history for 256 samples
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

        let mut scene_to_load: Option<String> = None;

        egui::Window::new("Engine Debug").show(&egui_ctx, |ui| {
            ui.heading("Renderer");
            ui.label(format!(
                "Frame time: {:.2}ms ({:.1} FPS)",
                self.smoothed_dt * 1000.0,
                1.0 / self.smoothed_dt
            ));
            ui.checkbox(&mut self.show_perf_graph, "Show Performance Graph");
            ui.separator();

            if self.camera.camera_locked {
                ui.label("📷 Camera: LOCKED (Tab to unlock)");
            } else {
                ui.label("📷 Camera: FREE (Tab to lock)");
            }
            ui.separator();

            ui.label("Scene:");
            egui::ComboBox::from_label("Select Scene")
                .selected_text(&self.current_scene)
                .show_ui(ui, |ui| {
                    for scene in &self.available_scenes {
                        if ui
                            .selectable_label(self.current_scene == *scene, scene)
                            .clicked()
                        {
                            scene_to_load = Some(scene.clone());
                        }
                    }
                });

            ui.separator();
            ui.checkbox(&mut self.render_graph.fxaa_enabled, "FXAA");
            ui.horizontal(|ui| {
                use i3_renderer::render_graph::AoMode;
                ui.label("AO:");
                ui.radio_value(&mut self.render_graph.ao_mode, AoMode::None, "None");
                ui.radio_value(&mut self.render_graph.ao_mode, AoMode::Gtao, "GTAO");
                ui.radio_value(&mut self.render_graph.ao_mode, AoMode::Rtao, "RTAO");
            });
            match self.render_graph.ao_mode {
                i3_renderer::render_graph::AoMode::Gtao => {
                    let gtao = &mut self.render_graph.gtao_group.gtao_pass;
                    ui.add(egui::Slider::new(&mut gtao.radius, 0.1_f32..=2.0).text("AO Radius"));
                    ui.add(egui::Slider::new(&mut gtao.final_power, 0.5_f32..=4.0).text("AO Final Power"));
                    ui.add(egui::Slider::new(&mut gtao.slice_count, 1_u32..=4).text("AO Slices"));
                    ui.add(egui::Slider::new(&mut gtao.step_count, 2_u32..=8).text("AO Steps"));
                    let alpha = &mut self.render_graph.gtao_group.gtao_temporal_pass.alpha;
                    ui.add(
                        egui::Slider::new(alpha, 0.01_f32..=1.0)
                            .text("AO Temporal Alpha")
                            .logarithmic(true),
                    );
                }
                i3_renderer::render_graph::AoMode::Rtao => {
                    let rtao = &mut self.render_graph.rtao_group.rtao_pass;
                    ui.add(egui::Slider::new(&mut rtao.radius, 0.1_f32..=5.0).text("RTAO Radius"));
                    let temporal = &mut self.render_graph.rtao_group.rtao_temporal_pass;
                    ui.add(egui::Slider::new(&mut temporal.alpha, 0.01_f32..=0.5).text("RTAO Temporal Alpha"));
                }
                i3_renderer::render_graph::AoMode::None => {}
            }

            ui.separator();
            {
                let ssr = &mut self.render_graph.sssr_sample_pass;
                ui.checkbox(&mut ssr.enabled, "SSR");
                if ssr.enabled {
                    ui.add(egui::Slider::new(&mut ssr.max_mip_level, 1_u32..=5).text("SSR Max Mip"));
                    ui.add(egui::Slider::new(&mut ssr.thickness, 0.01_f32..=1.0).text("SSR Thickness"));
                    let intensity = &mut self.render_graph.sssr_composite_pass.intensity;
                    ui.add(egui::Slider::new(intensity, 0.0_f32..=2.0).text("SSR Intensity"));

                    let dbg = &mut self.render_graph.sssr_sample_pass.debug_mode;
                    ui.horizontal(|ui| {
                        ui.label("SSR debug:");
                        ui.selectable_value(dbg, 0, "Off");
                        ui.selectable_value(dbg, 1, "R=hit G=iter B=thickness");
                        ui.selectable_value(dbg, 2, "RG=hit_uv B=hit");
                    });

                    // Downsample factor: applies to sample and bilateral passes together.
                    let mut factor = self.render_graph.sssr_sample_pass.downsample_factor;
                    if ui.add(egui::Slider::new(&mut factor, 1_u32..=4).text("SSR Downsample")).changed() {
                        self.render_graph.sssr_sample_pass.downsample_factor = factor;
                        self.render_graph.sssr_bilateral_pass.downsample_factor = factor;
                    }
                }
            }

            ui.separator();
            {
                let bloom = &mut self.render_graph.bloom_pass;
                ui.checkbox(&mut bloom.enabled, "Bloom");
                if bloom.enabled {
                    ui.add(
                        egui::Slider::new(&mut bloom.threshold, 0.5_f32..=4.0)
                            .text("Bloom Threshold"),
                    );
                    ui.add(egui::Slider::new(&mut bloom.knee, 0.0_f32..=1.0).text("Bloom Knee"));
                    ui.add(
                        egui::Slider::new(&mut bloom.intensity, 0.0_f32..=1.0)
                            .logarithmic(true)
                            .text("Bloom Intensity"),
                    );
                }
            }

            ui.separator();
            ui.label("Culling Debug:");
            ui.checkbox(&mut self.show_culling_debug, "Show bounding boxes");
            if self.show_culling_debug {
                ui.checkbox(&mut self.culling_show_ids, "  Instance IDs");
            }

            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Debug Channel:");
                let selected_label = match self.render_graph.debug_channel {
                    DebugChannel::Lit => "Lit (Final)",
                    DebugChannel::Albedo => "Albedo",
                    DebugChannel::Normal => "Normals",
                    DebugChannel::Roughness => "Roughness",
                    DebugChannel::Metallic => "Metallic",
                    DebugChannel::Emissive => "Emissive",
                    DebugChannel::Depth => "Depth",
                    DebugChannel::AO => "AO (accumulated)",
                    DebugChannel::SsrRaw => "SSR (raw stochastic)",
                    DebugChannel::SsrUpsampled => "SSR (upsampled)",
                    DebugChannel::BloomBuffer => "Bloom buffer",

                    DebugChannel::LightDensity => "Light density",
                    DebugChannel::ClusterGrid => "Cluster grid",
                };
                egui::ComboBox::from_id_salt("debug_channel")
                    .selected_text(selected_label)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::Lit,
                            "Lit (Final)",
                        );
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::Albedo,
                            "Albedo",
                        );
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::Normal,
                            "Normals",
                        );
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::Roughness,
                            "Roughness",
                        );
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::Metallic,
                            "Metallic",
                        );
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::Emissive,
                            "Emissive",
                        );
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::Depth,
                            "Depth",
                        );
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::AO,
                            "AO (accumulated)",
                        );
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::SsrRaw,
                            "SSR (raw stochastic)",
                        );
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::SsrUpsampled,
                            "SSR (upsampled)",
                        );
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::BloomBuffer,
                            "Bloom buffer",
                        );

                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::LightDensity,
                            "Light density",
                        );
                        ui.selectable_value(
                            &mut self.render_graph.debug_channel,
                            DebugChannel::ClusterGrid,
                            "Cluster grid",
                        );
                    });
            });
        });

        if self.show_perf_graph {
            egui::Window::new("Performance Graph").show(&egui_ctx, |ui| {
                self.draw_performance_graph(ui);
            });
        }

        if let Some(name) = scene_to_load {
            self.load_scene(&name);
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
            far, // reverse-Z: swap near/far so near→1, far→0
            near,
        );

        // ── Debug draw: fill AABB wireframes before submitting the frame ──
        {
            let vp = projection * view;
            self.render_graph.debug_draw_pass.clear();

            // Camera billboard vectors extracted from the view matrix (row 0 = right, row 1 = up).
            let cam_right = [view[(0, 0)], view[(0, 1)], view[(0, 2)]];
            let cam_up = [view[(1, 0)], view[(1, 1)], view[(1, 2)]];

            if self.show_culling_debug {
                let col = [0.0_f32, 1.0, 0.2, 0.85]; // green = frustum-visible
                for (idx, inst) in self.render_graph.cached_instances.iter().enumerate() {
                    // Skip if outside the camera frustum.
                    if !frustum_cull_cpu(inst.world_aabb_min, inst.world_aabb_max, &vp) {
                        continue;
                    }

                    self.render_graph.debug_draw_pass.push_aabb(
                        inst.world_aabb_min,
                        inst.world_aabb_max,
                        col,
                    );

                    // Optional: draw instance index as 7-segment label (= thread ID in draw_call_gen).
                    if self.culling_show_ids {
                        let cx = (inst.world_aabb_min[0] + inst.world_aabb_max[0]) * 0.5;
                        let cy = (inst.world_aabb_min[1] + inst.world_aabb_max[1]) * 0.5;
                        let cz = (inst.world_aabb_min[2] + inst.world_aabb_max[2]) * 0.5;
                        let ext_x = (inst.world_aabb_max[0] - inst.world_aabb_min[0]).abs();
                        let ext_y = (inst.world_aabb_max[1] - inst.world_aabb_min[1]).abs();
                        let ext_z = (inst.world_aabb_max[2] - inst.world_aabb_min[2]).abs();
                        let scale = (ext_x.min(ext_y).min(ext_z) * 0.4).clamp(0.05, 2.0);
                        self.render_graph.debug_draw_pass.push_label_3d(
                            [cx, cy, cz],
                            idx as u32,
                            col,
                            scale,
                            cam_right,
                            cam_up,
                        );
                    }
                }
            }
        }

        if let Err(e) = self.render_graph.render(
            &mut self.backend,
            self.window,
            &self.scene,
            view,
            projection,
            near,
            far,
            width,
            height,
            self.dt,
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
            if *key == KeyCode::F11 || *key == KeyCode::Return {
                self.is_fullscreen = !self.is_fullscreen;
                self.backend.set_fullscreen(self.window, self.is_fullscreen);
            }
        }

        // Camera always receives events — Tab lock prevents mouse look when GUI is needed.
        self.camera.handle_event(event);
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
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

    let mut available_scenes = loader_arc.list_assets::<SceneAsset>();
    available_scenes.sort();

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
        available_scenes,
        frame_time_history: VecDeque::with_capacity(2000),
        show_perf_graph: false,
        sample_accum_time: 0.0,
        sample_max_dt: 0.0,
        show_culling_debug: false,
        culling_show_ids: false,
    };

    // Initial load
    app.load_scene(&scene_name);

    main_loop(app);

    Ok(())
}
