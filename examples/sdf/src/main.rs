mod sdf_gbuffer_pass;
mod sdf_tree;

use std::f32::consts::FRAC_PI_4;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use examples_common::basic_scene::BasicScene;
use examples_common::camera_controller::CameraController;
use examples_common::{ExampleApp, RendererDebugGui, get_gpu_index, init_tracing, main_loop, maybe_list_gpus};
use i3_egui::prelude::*;
use i3_gfx::prelude::*;
use i3_io::prelude::*;
use i3_math::Transform;
use i3_math::nalgebra::{UnitQuaternion, Vector3};
use i3_renderer::prelude::*;
use i3_renderer::render_graph::RenderConfig;
use i3_vulkan_backend::backend::VulkanBackend;
use nalgebra_glm as glm;
use tracing::warn;

use sdf_gbuffer_pass::SdfGBufferPass;
use sdf_tree::{GpuSdfNode, SdfMaterial, SdfNode, SdfShape};

// ─── Scene builder ────────────────────────────────────────────────────────────

fn leaf(shape: SdfShape, pos: [f32; 3], rot: UnitQuaternion<f32>, mat: SdfMaterial) -> SdfNode {
    SdfNode::Leaf {
        shape,
        transform: Transform::new(Vector3::new(pos[0], pos[1], pos[2]), rot, 1.0),
        material: mat,
    }
}

fn union(a: SdfNode, b: SdfNode) -> SdfNode {
    SdfNode::Union {
        left: Box::new(a),
        right: Box::new(b),
    }
}

fn sub(solid: SdfNode, hole: SdfNode) -> SdfNode {
    SdfNode::Subtraction {
        solid: Box::new(solid),
        hole: Box::new(hole),
    }
}

fn smooth(a: SdfNode, b: SdfNode, k: f32) -> SdfNode {
    SdfNode::SmoothUnion {
        left: Box::new(a),
        right: Box::new(b),
        k,
    }
}

fn build_scene(time: f32, terrain_scale: f32, terrain_amplitude: f32) -> SdfNode {
    let id = UnitQuaternion::identity();
    // 90° around X → torus ring becomes vertical (XY plane)
    let rot_x90 = UnitQuaternion::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0);

    // Materials
    let m_marble = SdfMaterial {
        albedo: [0.95, 0.93, 0.90],
        roughness: 0.25,
        metallic: 0.0,
    };
    let m_gold = SdfMaterial {
        albedo: [0.83, 0.68, 0.21],
        roughness: 0.1,
        metallic: 1.0,
    };
    let m_orange = SdfMaterial {
        albedo: [0.80, 0.38, 0.14],
        roughness: 0.65,
        metallic: 0.0,
    };
    let m_steel = SdfMaterial {
        albedo: [0.60, 0.64, 0.70],
        roughness: 0.2,
        metallic: 1.0,
    };
    let m_ruby = SdfMaterial {
        albedo: [0.80, 0.08, 0.10],
        roughness: 0.1,
        metallic: 0.0,
    };
    let m_stone = SdfMaterial {
        albedo: [0.55, 0.50, 0.45],
        roughness: 0.85,
        metallic: 0.0,
    };

    // fBm terrain — base plane sits at y = -(amplitude*2), hills peak near y=0
    let terrain_y = -(terrain_amplitude * 2.0);
    let terrain = leaf(
        SdfShape::Terrain { scale: terrain_scale, amplitude: terrain_amplitude, octaves: 7 },
        [0.0, terrain_y, 0.0],
        id,
        m_stone.clone(),
    );

    // Central marble sphere
    let sphere = leaf(
        SdfShape::Sphere { radius: 1.0 },
        [0.0, 1.2, 0.0],
        id,
        m_marble,
    );

    // Gold equatorial ring around the sphere (smooth-unioned for a subtle merge)
    let eq_ring = leaf(
        SdfShape::Torus {
            major_radius: 1.35,
            minor_radius: 0.09,
        },
        [0.0, 1.2, 0.0],
        id,
        m_gold.clone(),
    );

    let center = smooth(sphere, eq_ring, 0.06);

    // Left: orange capsule pillar
    let left_capsule = leaf(
        SdfShape::Capsule {
            half_height: 1.1,
            radius: 0.28,
        },
        [-2.8, 1.1, -0.5],
        id,
        m_orange,
    );

    // Right: steel cylinder pedestal + ruby sphere on top (smooth union)
    let pedestal = leaf(
        SdfShape::Cylinder {
            half_height: 1.0,
            radius: 0.42,
        },
        [2.8, 1.0, -0.5],
        id,
        m_steel,
    );
    let angle = time * 1.5;
    let gem = leaf(
        SdfShape::Sphere { radius: 0.6 },
        [
            2.8 + angle.cos() * 0.55,
            2.38 + (time * 3.0).sin() * 0.15,
            -0.5 + angle.sin() * 0.55,
        ],
        id,
        m_ruby,
    );
    let right_piece = smooth(pedestal, gem, 0.12);

    // Back: stone block with a spherical niche (subtraction)
    let wall = leaf(
        SdfShape::Box {
            half_extents: [1.4, 1.6, 0.35],
        },
        [0.0, 1.6, -3.8],
        id,
        m_stone.clone(),
    );
    let niche = leaf(
        SdfShape::Sphere { radius: 0.85 },
        [0.0, 1.6, -4.0],
        id,
        m_stone.clone(),
    );
    let arch = sub(wall, niche);

    // Gold vertical torus (portal ring, 90° rotated) on the right-back
    let portal = leaf(
        SdfShape::Torus {
            major_radius: 0.7,
            minor_radius: 0.07,
        },
        [2.8, 1.5, -3.0],
        rot_x90,
        m_gold,
    );

    // Compose everything with nested unions
    union(
        terrain,
        union(
            center,
            union(left_capsule, union(right_piece, union(arch, portal))),
        ),
    )
}

// ─── SdfApp ───────────────────────────────────────────────────────────────────

struct SdfApp {
    backend:           VulkanBackend,
    window:            WindowHandle,
    render_graph:      DefaultRenderGraph,
    ui:                Arc<i3_egui::UiSystem>,
    camera:            CameraController,
    scene:             BasicScene,
    sdf_nodes:         Arc<RwLock<Vec<GpuSdfNode>>>,
    dt:                f32,
    smoothed_dt:       f32,
    time:              f32,
    debug_gui:         RendererDebugGui,
    terrain_scale:     f32,
    terrain_amplitude: f32,
}

impl ExampleApp for SdfApp {
    fn update(&mut self, delta: Duration, smoothed: Duration) {
        self.dt = delta.as_secs_f32();
        self.smoothed_dt = smoothed.as_secs_f32();
        self.time += self.dt;
        self.debug_gui.update(self.dt);
        self.camera.update(delta);
    }

    fn render(&mut self) {
        self.ui.begin_frame();
        let egui_ctx = self.ui.context().clone();

        let mut t_scale = self.terrain_scale;
        let mut t_amp   = self.terrain_amplitude;

        self.debug_gui.show(
            &egui_ctx,
            &mut self.render_graph,
            &self.camera,
            self.smoothed_dt,
            |ui| {
                ui.separator();
                ui.label("Terrain");
                ui.add(egui::Slider::new(&mut t_scale, 0.5_f32..=10.0).text("Scale"));
                ui.add(egui::Slider::new(&mut t_amp,   0.1_f32..=2.0 ).text("Amplitude"));
            },
        );

        self.terrain_scale     = t_scale;
        self.terrain_amplitude = t_amp;

        self.ui.update_textures(&mut self.backend);

        // Rebuild the flat node list from the CPU tree and push it to the pass
        let mut flat = Vec::new();
        build_scene(self.time, self.terrain_scale, self.terrain_amplitude).to_gpu_flat(&mut flat);
        *self.sdf_nodes.write().unwrap() = flat;

        let view = self.camera.view_matrix();
        let (w, h) = self.backend.window_size(self.window).unwrap_or((1280, 720));
        let near = 0.1_f32;
        let far = 500.0_f32;
        let proj = glm::perspective_rh_zo(w as f32 / h as f32, FRAC_PI_4, far, near);

        if let Err(e) = self.render_graph.render(
            &mut self.backend,
            self.window,
            &self.scene,
            view,
            proj,
            near,
            far,
            w,
            h,
            self.dt,
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

    let config = RenderConfig {
        width: 1280,
        height: 720,
    };
    let ui = Arc::new(i3_egui::UiSystem::new(1280, 720));

    // ── Asset setup ──────────────────────────────────────────────────────────
    let vfs = Arc::new(Vfs::new());
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

    // ── Shared SDF node list ─────────────────────────────────────────────────
    let sdf_nodes: Arc<RwLock<Vec<GpuSdfNode>>> = Arc::new(RwLock::new(Vec::new()));

    // ── Render graph + SDF pass ──────────────────────────────────────────────
    let mut render_graph = DefaultRenderGraph::new(&mut backend, &config);
    render_graph.ao_mode = AoMode::None;
    render_graph.publish("UiSystem", ui.clone());
    render_graph.publish("AssetLoader", loader);

    render_graph
        .extra_gbuffer_passes
        .push(Box::new(SdfGBufferPass::new(sdf_nodes.clone())));

    render_graph.init(&mut backend);

    // ── Camera ───────────────────────────────────────────────────────────────
    let mut camera = CameraController::new();
    camera.position = glm::vec3(0.0, 2.5, 6.0);
    camera.pitch = -0.2;
    camera.move_speed = 5.0;

    main_loop(SdfApp {
        backend,
        window,
        render_graph,
        ui,
        camera,
        scene: BasicScene::new(),
        sdf_nodes,
        dt:                0.016,
        smoothed_dt:       0.016,
        time:              0.0,
        debug_gui:         RendererDebugGui::new(),
        terrain_scale:     3.0,
        terrain_amplitude: 0.75,
    });

    Ok(())
}
