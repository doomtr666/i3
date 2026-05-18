mod brickmap_validate;
mod clipmap_pass;
mod compute_bake_pass;

use std::f32::consts::FRAC_PI_4;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use examples_common::basic_scene::BasicScene;
use examples_common::camera_controller::CameraController;
use examples_common::{
    ExampleApp, RendererDebugGui, get_gpu_index, init_tracing, main_loop, maybe_list_gpus,
};
use i3_egui::prelude::*;
use i3_gfx::prelude::*;
use i3_io::prelude::*;
use i3_math::nalgebra::UnitQuaternion;
use i3_renderer::prelude::*;
use i3_renderer::render_graph::RenderConfig;
use i3_vulkan_backend::backend::VulkanBackend;
use nalgebra_glm as glm;
use tracing::warn;

use clipmap_pass::ClipmapGBufferPass;
use compute_bake_pass::{ClipmapGpuBuffers, ComputeBakePass};

// ─── Clipmap scene builder ────────────────────────────────────────────────────

fn build_clipmap_scene(gem_pos: [f32; 3]) -> i3_voxel::SdfScene {
    use i3_math::Transform;
    use i3_math::nalgebra::Vector3;
    use i3_voxel::{SdfPrimitive, SdfScene};

    let id = UnitQuaternion::identity();
    let rot_x90 = UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    let mut scene = SdfScene::new();

    scene.add(
        &Transform::new(Vector3::new(0.0, 1.2, 0.0), id, 1.0),
        &SdfPrimitive::Sphere { radius: 1.0 },
    );
    scene.add(
        &Transform::new(Vector3::new(0.0, 1.2, 0.0), id, 1.0),
        &SdfPrimitive::Torus { major_radius: 1.35, minor_radius: 0.18 },
    );
    scene.add(
        &Transform::new(Vector3::new(-2.8, 1.1, -0.5), id, 1.0),
        &SdfPrimitive::Capsule { half_height: 1.1, radius: 0.28 },
    );
    scene.add(
        &Transform::new(Vector3::new(2.8, 1.0, -0.5), id, 1.0),
        &SdfPrimitive::Cylinder { half_height: 1.0, radius: 0.42 },
    );
    scene.add(
        &Transform::new(Vector3::new(gem_pos[0], gem_pos[1], gem_pos[2]), id, 1.0),
        &SdfPrimitive::Sphere { radius: 0.6 },
    );
    // Wall with spherical niche (subtraction)
    scene.add(
        &Transform::new(Vector3::new(0.0, 1.6, -3.8), id, 1.0),
        &SdfPrimitive::Box { half_extents: Vector3::new(1.4, 1.6, 0.35) },
    );
    scene.sub(
        &Transform::new(Vector3::new(0.0, 1.6, -4.0), id, 1.0),
        &SdfPrimitive::Sphere { radius: 0.85 },
    );
    scene.add(
        &Transform::new(Vector3::new(2.8, 1.5, -3.0), rot_x90, 1.0),
        &SdfPrimitive::Torus { major_radius: 0.7, minor_radius: 0.14 },
    );
    // Floor
    scene.add(
        &Transform::new(Vector3::new(0.0, -1.0, 0.0), id, 1.0),
        &SdfPrimitive::Box { half_extents: Vector3::new(200.0, 1.0, 200.0) },
    );

    scene.build_bvh();
    scene
}

// ─── SdfApp ───────────────────────────────────────────────────────────────────

struct SdfApp {
    backend:        VulkanBackend,
    window:         WindowHandle,
    render_graph:   DefaultRenderGraph,
    ui:             Arc<i3_egui::UiSystem>,
    camera:         CameraController,
    scene:          BasicScene,
    dt:             f32,
    smoothed_dt:    f32,
    time:           f32,
    debug_gui:      RendererDebugGui,
    bm_enabled:      Arc<AtomicBool>,
    bm_debug_flags:  Arc<AtomicU32>,
    clipmap_scene:  Arc<RwLock<i3_voxel::SdfScene>>,
    clipmap_state:  Arc<RwLock<i3_brickmap::BrickmapClipmapState>>,
    dig_radius:     f32,
    gem_pos_last:   [f32; 3],
    pending_action: Option<bool>,
}

impl ExampleApp for SdfApp {
    fn update(&mut self, delta: Duration, smoothed: Duration) {
        self.dt         = delta.as_secs_f32();
        self.smoothed_dt = smoothed.as_secs_f32();
        self.time       += self.dt;
        self.debug_gui.update(self.dt);
        self.camera.update(delta);
    }

    fn render(&mut self) {
        // Update animated gem position
        {
            let angle   = self.time * 1.5;
            let new_gem = [
                2.8 + angle.cos() * 0.55,
                2.38 + (self.time * 3.0).sin() * 0.15,
                -0.5 + angle.sin() * 0.55,
            ];
            let prev = self.gem_pos_last;
            let d = ((new_gem[0]-prev[0]).powi(2)
                    + (new_gem[1]-prev[1]).powi(2)
                    + (new_gem[2]-prev[2]).powi(2)).sqrt();
            if d > 0.05 {
                *self.clipmap_scene.write().unwrap() = build_clipmap_scene(new_gem);
                let mut cm = self.clipmap_state.write().unwrap();
                cm.invalidate_sphere(prev, 0.7);
                cm.invalidate_sphere(new_gem, 0.7);
                self.gem_pos_last = new_gem;
            }
        }

        // Camera snap (no more CPU bake_frame — compute pass handles it)
        {
            let cam     = self.camera.position;
            let cam_pos = [cam.x, cam.y, cam.z];
            self.clipmap_state.write().unwrap().update_camera(cam_pos);
        }

        self.ui.begin_frame();
        let egui_ctx = self.ui.context().clone();

        let mut dig_radius    = self.dig_radius;
        let bm_enabled        = self.bm_enabled.clone();
        let bm_debug_flags    = self.bm_debug_flags.clone();
        let clipmap_state_g   = self.clipmap_state.clone();

        self.debug_gui.show(
            &egui_ctx,
            &mut self.render_graph,
            &self.camera,
            self.smoothed_dt,
            |ui| {
                ui.separator();
                ui.label("Clipmap");
                let mut en = bm_enabled.load(Ordering::Relaxed);
                ui.checkbox(&mut en, "Enabled");
                bm_enabled.store(en, Ordering::Relaxed);

                let mut flags = bm_debug_flags.load(Ordering::Relaxed);
                let mut b0 = (flags & 1) != 0;
                let mut b1 = (flags & 2) != 0;
                let mut b2 = (flags & 4) != 0;
                let mut b3 = (flags & 8) != 0;
                ui.checkbox(&mut b0, "Debug: level colors");
                ui.checkbox(&mut b1, "Debug: brick grid");
                ui.checkbox(&mut b2, "Debug: world-Y gradient");
                ui.checkbox(&mut b3, "Debug: step count heat");
                flags = (b0 as u32) | ((b1 as u32) << 1) | ((b2 as u32) << 2) | ((b3 as u32) << 3);
                bm_debug_flags.store(flags, Ordering::Relaxed);
                if let Ok(cm) = clipmap_state_g.try_read() {
                    for lev in 0..i3_brickmap::NUM_LEVELS {
                        let bricks  = cm.levels[lev].data.brick_count;
                        let pending = cm.levels[lev].bake_state.pending_count;
                        ui.label(
                            egui::RichText::new(format!("L{lev}: {bricks} bricks  ({pending} pending)"))
                                .monospace(),
                        );
                    }
                }
                ui.separator();
                ui.label("Edit (LMB=dig  RMB=fill)");
                ui.add(egui::Slider::new(&mut dig_radius, 0.25_f32..=3.0).text("Radius (m)"));
            },
        );

        self.dig_radius = dig_radius;

        if let Some(is_dig) = self.pending_action.take() {
            if !egui_ctx.wants_pointer_input() {
                self.apply_edit(is_dig);
            }
        }

        self.ui.update_textures(&mut self.backend);

        let view = self.camera.view_matrix();
        let (w, h) = self.backend.window_size(self.window).unwrap_or((1280, 720));
        let near = 0.1_f32;
        let far  = 500.0_f32;
        let proj = glm::perspective_rh_zo(w as f32 / h as f32, FRAC_PI_4, far, near);

        if let Err(e) = self.render_graph.render(
            &mut self.backend, self.window, &self.scene,
            view, proj, near, far, w, h, self.dt,
        ) {
            warn!("Render error: {}", e);
        }
    }

    fn poll_events(&mut self) -> Vec<Event> {
        self.backend.poll_events()
    }

    fn handle_event(&mut self, event: &Event) {
        self.ui.handle_event(event);
        self.camera.handle_event(event);
        match event {
            Event::MouseDown { button: 1, .. } => self.pending_action = Some(true),
            Event::MouseDown { button: 3, .. } => self.pending_action = Some(false),
            _ => {}
        }
    }
}

impl SdfApp {
    fn apply_edit(&mut self, is_dig: bool) {
        use i3_math::Transform;
        use i3_math::nalgebra::{Point3, UnitQuaternion, Vector3};
        use i3_voxel::{AABB, SdfPrimitive, SdfScene};

        let yaw   = self.camera.yaw;
        let pitch = self.camera.pitch;
        let forward = glm::vec3(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        );
        let ro = self.camera.position;

        let hit = {
            let scene = self.clipmap_scene.read().unwrap();
            let big   = AABB::new(Point3::new(-200.0, -200.0, -200.0), Point3::new(200.0, 200.0, 200.0));
            let nodes = scene.get_nodes(&big);
            let mut t = 0.2f32;
            loop {
                let p = ro + forward * t;
                let d = SdfScene::value(&nodes, &Point3::new(p.x, p.y, p.z));
                if d < 0.01 { break Some(p); }
                t += d.max(0.01).min(1.0);
                if t > 60.0 { break None; }
            }
        };

        if let Some(hit) = hit {
            let xf = Transform::new(
                Vector3::new(hit.x, hit.y, hit.z),
                UnitQuaternion::identity(),
                1.0,
            );
            let sphere = SdfPrimitive::Sphere { radius: self.dig_radius };
            {
                let mut scene = self.clipmap_scene.write().unwrap();
                if is_dig { scene.sub(&xf, &sphere); } else { scene.add(&xf, &sphere); }
                scene.build_bvh();
            }
            let center = [hit.x, hit.y, hit.z];
            self.clipmap_state.write().unwrap().invalidate_sphere(center, self.dig_radius);
            tracing::info!(
                "{} at ({:.2}, {:.2}, {:.2}) r={:.2}",
                if is_dig { "DIG" } else { "FILL" },
                hit.x, hit.y, hit.z, self.dig_radius
            );
        }
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let _guard = init_tracing("sdf.log");

    let mut backend = VulkanBackend::new()?;
    maybe_list_gpus(&backend);
    backend.initialize(get_gpu_index())?;

    let window = backend.create_window(WindowDesc {
        title: "SDF".to_string(),
        width: 1280,
        height: 720,
    })?;

    let config = RenderConfig { width: 1280, height: 720 };
    let ui     = Arc::new(i3_egui::UiSystem::new(1280, 720));

    // ── Asset setup ──────────────────────────────────────────────────────────
    let vfs     = Arc::new(Vfs::new());
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_default();

    for (cat, blob) in [("system.i3c", "system.i3b"), ("sdf.i3c", "sdf.i3b")] {
        if exe_dir.join(cat).exists() {
            if let Ok(bundle) = BundleBackend::mount(
                exe_dir.join(cat).to_str().unwrap(),
                exe_dir.join(blob).to_str().unwrap(),
            ) {
                let _ = vfs.mount(Box::new(bundle));
            }
        }
    }
    let loader = Arc::new(AssetLoader::new(vfs));

    // ── Shared GPU atlas buffers (created before render graph init) ───────────
    let gpu_buffers = Arc::new(ClipmapGpuBuffers::new(&mut backend));

    // ── Render graph ──────────────────────────────────────────────────────────
    let mut render_graph = DefaultRenderGraph::new(&mut backend, &config);
    render_graph.ao_mode = AoMode::None;
    render_graph.publish("UiSystem", ui.clone());
    render_graph.publish("AssetLoader", loader);

    // ── Clipmap state ─────────────────────────────────────────────────────────
    let initial_cam   = [0.0f32, 2.5, 6.0];
    let clipmap_scene = Arc::new(RwLock::new(build_clipmap_scene([3.35, 2.38, -0.5])));
    let clipmap_state = Arc::new(RwLock::new(i3_brickmap::BrickmapClipmapState::new(initial_cam)));
    let bm_enabled     = Arc::new(AtomicBool::new(true));
    let bm_debug_flags = Arc::new(AtomicU32::new(0));

    // ── Compute bake pass (pre-GBuffer) ───────────────────────────────────────
    render_graph.extra_pre_gbuffer_passes.push(Box::new(ComputeBakePass::new(
        clipmap_state.clone(),
        clipmap_scene.clone(),
        gpu_buffers.clone(),
    )));

    // ── Clipmap GBuffer pass ───────────────────────────────────────────────────
    render_graph.extra_gbuffer_passes.push(Box::new(ClipmapGBufferPass::new(
        clipmap_state.clone(),
        bm_enabled.clone(),
        bm_debug_flags.clone(),
        gpu_buffers.clone(),
    )));

    render_graph.init(&mut backend);

    brickmap_validate::run();

    // ── Camera ────────────────────────────────────────────────────────────────
    let mut camera = CameraController::new();
    camera.position   = glm::vec3(0.0, 2.5, 6.0);
    camera.pitch      = -0.2;
    camera.move_speed = 5.0;

    main_loop(SdfApp {
        backend,
        window,
        render_graph,
        ui,
        camera,
        scene:          BasicScene::new(),
        dt:             0.016,
        smoothed_dt:    0.016,
        time:           0.0,
        debug_gui:      RendererDebugGui::new(),
        bm_enabled,
        bm_debug_flags,
        clipmap_scene,
        clipmap_state,
        dig_radius:     1.0,
        gem_pos_last:   [3.35, 2.38, -0.5],
        pending_action: None,
    });

    Ok(())
}
