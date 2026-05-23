extern crate nalgebra_glm;

use examples_common::basic_scene::BasicScene;
use examples_common::{AppRenderer, ExampleApp, RendererDebugGui, init_renderer, init_tracing, main_loop};
use i3_egui::prelude::*;
use i3_gfx::prelude::*;
use i3_io::prelude::*;
use i3_renderer::prelude::*;
use i3_vulkan_backend::prelude::*;
use nalgebra_glm as glm;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use uuid::Uuid;

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
    debug_gui: RendererDebugGui,
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

}

impl ExampleApp for DeferredGltfApp {
    fn update(&mut self, delta: Duration, smoothed_delta: Duration) {
        self.dt = delta.as_secs_f32();
        self.smoothed_dt = smoothed_delta.as_secs_f32();
        self.time += self.dt;
        self.debug_gui.update(self.dt);
        self.camera.update(delta);
    }

    fn render(&mut self) {
        self.ui.begin_frame();
        let egui_ctx = self.ui.context().clone();

        let mut scene_to_load: Option<String> = None;
        let current_scene   = &self.current_scene;
        let available_scenes = &self.available_scenes;
        let scene_to_load_ref = &mut scene_to_load;

        self.debug_gui.show(
            &egui_ctx,
            &mut self.render_graph,
            &self.camera,
            self.smoothed_dt,
            |ui| {
                ui.separator();
                ui.label("Scene:");
                egui::ComboBox::from_label("Select Scene")
                    .selected_text(current_scene.as_str())
                    .show_ui(ui, |ui| {
                        for scene in available_scenes {
                            if ui.selectable_label(*current_scene == *scene, scene).clicked() {
                                *scene_to_load_ref = Some(scene.clone());
                            }
                        }
                    });
            },
        );

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

            if self.debug_gui.show_culling_debug {
                let col = [0.0_f32, 1.0, 0.2, 0.85]; // green = frustum-visible
                for (idx, inst) in self.render_graph.cached_instances.iter().enumerate() {
                    let aabb = i3_math::AABB::new(
                        nalgebra::Point3::from(inst.world_aabb_min),
                        nalgebra::Point3::from(inst.world_aabb_max),
                    );
                    if !aabb.is_in_frustum(&vp) {
                        continue;
                    }

                    self.render_graph.debug_draw_pass.push_aabb(
                        inst.world_aabb_min,
                        inst.world_aabb_max,
                        col,
                    );

                    if self.debug_gui.culling_show_ids {
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

    // Setup IO and VFS (bundle-specific — stays here)
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
    let loader_arc = Arc::new(AssetLoader::new(Arc::new(vfs)));

    let AppRenderer { backend, window, render_graph, ui } =
        init_renderer("Deferred glTF (Baked)", 1280, 720, Some(loader_arc.clone()))?;

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
        debug_gui: RendererDebugGui::new(),
    };

    // Initial load
    app.load_scene(&scene_name);

    main_loop(app);

    Ok(())
}
