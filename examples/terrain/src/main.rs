mod brickmap_validate;

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
use i3_math::{AABB, nalgebra::UnitQuaternion};
use i3_renderer::prelude::*;
use i3_renderer::render_graph::RenderConfig;
use i3_voxel::{
    ClipmapState, ClipmapBuffers,
    debug_ui::{ClipmapDebugUi, ClipmapParams},
    passes::create_clipmap_passes,
};
use i3_vulkan_backend::backend::VulkanBackend;
use nalgebra_glm as glm;
use tracing::warn;

// ─── Edit operation ───────────────────────────────────────────────────────────

struct EditOp {
    transform: i3_math::Transform,
    primitive: i3_voxel::SdfPrimitive,
    subtract: bool,
}

// ─── Scene builder ────────────────────────────────────────────────────────────

/// Terrain amplitude — MUST match `T_AMP` in noise.slangh — and the SVO region.
const TERRAIN_AMP: f32 = 14.0;
const TERRAIN_REGION: [f32; 3] = [200.0, 36.0, 200.0];

/// The terrain noise graph — single source of truth for the CPU generator
/// (`build_rnoise`) and the GPU bytecode (`compile`). DomainScale carries the base
/// frequency so both backends agree without a duplicated constant.
fn terrain_graph() -> i3_voxel::NoiseGraph {
    use i3_voxel::NoiseGraph as G;
    // Base terrain: rich fBm(Perlin), base wavelength 48 m, 5 octaves (down to ~3 m).
    let base = G::DomainScale {
        factor: 1.0 / 48.0,
        src: Box::new(G::Fbm {
            seed: 1337,
            octaves: 6,
            lacunarity: 2.0,
            gain: 0.5,
        }),
    };
    // Warp fields: low-frequency fBm (wavelength 96 m) → large-scale organic distortion.
    let field = |seed: u32| {
        Box::new(G::DomainScale {
            factor: 1.0 / 96.0,
            src: Box::new(G::Fbm {
                seed,
                octaves: 3,
                lacunarity: 2.0,
                gain: 0.5,
            }),
        })
    };
    G::DomainWarp {
        inner: Box::new(base),
        x: field(11),
        y: field(22),
        z: field(33),
        intensity: 42.0, // domain displacement ≈ ±42 m → strong distortion
    }
}

fn build_sdf_scene() -> i3_voxel::SdfScene {
    use i3_math::Transform;
    use i3_math::nalgebra::Vector3;
    use i3_voxel::{SdfPrimitive, SdfScene};

    let id = UnitQuaternion::identity();
    let mut scene = SdfScene::new();

    // Terrain-only scene. Analytic primitives stay supported (the runtime CSG edits —
    // dig/add — depend on them), but no static/animated instances are seeded here.
    // 3D volumetric terrain as the ground. IDENTITY transform is required: the GPU
    // bake evaluates the graph in world space, so local must equal world for CPU/GPU
    // parity. The CPU generator and the GPU bytecode both come from `terrain_graph()`
    // (single source of truth); the frequency lives in the graph (DomainScale), so
    // the primitive's frequency is 1.0. Amplitude must match T_AMP in noise.slangh.
    scene.add_mat(
        &Transform::new(Vector3::zeros(), id, 1.0),
        &SdfPrimitive::volume_terrain(
            Vector3::new(TERRAIN_REGION[0], TERRAIN_REGION[1], TERRAIN_REGION[2]),
            TERRAIN_AMP, // amplitude  → T_AMP
            terrain_graph().build_rnoise(),
        ),
        5, // terrain material → TERRAIN_MAT
    );

    scene.build_bvh();
    scene
}

fn build_full_scene(edits: &[EditOp]) -> i3_voxel::SdfScene {
    let mut scene = build_sdf_scene();
    for edit in edits {
        if edit.subtract {
            scene.sub(&edit.transform, &edit.primitive);
        } else {
            scene.add(&edit.transform, &edit.primitive);
        }
    }
    scene.build_bvh();
    scene
}

// ─── SdfApp ───────────────────────────────────────────────────────────────────

struct SdfApp {
    backend: VulkanBackend,
    window: WindowHandle,
    render_graph: DefaultRenderGraph,
    ui: Arc<i3_egui::UiSystem>,
    camera: CameraController,
    scene: BasicScene,
    dt: f32,
    smoothed_dt: f32,
    time: f32,
    debug_gui: RendererDebugGui,

    clip_enabled: Arc<AtomicBool>,
    clip_debug_flags: Arc<AtomicU32>,
    clip_params: ClipmapParams,
    clipmap: Arc<RwLock<ClipmapState>>,
    sdf_scene: Arc<RwLock<i3_voxel::SdfScene>>,

    clip_freeze: bool,
    dig_radius: f32,
    pending_action: Option<bool>,
    edits: Vec<EditOp>,
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
        // ── Clipmap re-centre on the camera ───────────────────────────────────
        if !self.clip_freeze {
            let cam = self.camera.position;
            let scene = self.sdf_scene.read().unwrap();
            let mut clip = self.clipmap.write().unwrap();
            clip.update_camera(cam.into(), &scene);
        }

        // ── UI ────────────────────────────────────────────────────────────────
        self.ui.begin_frame();
        let egui_ctx = self.ui.context().clone();

        let clip_enabled = self.clip_enabled.clone();
        let clip_debug_flags = self.clip_debug_flags.clone();
        let mut clip_params = std::mem::take(&mut self.clip_params);
        let mut dig_radius = self.dig_radius;
        let mut clip_freeze = self.clip_freeze;
        let clipmap_g = self.clipmap.clone();

        self.debug_gui.show(
            &egui_ctx,
            &mut self.render_graph,
            &self.camera,
            self.smoothed_dt,
            |ui| {
                ui.separator();
                let mut en = clip_enabled.load(Ordering::Relaxed);
                ui.checkbox(&mut en, "Terrain render");
                clip_enabled.store(en, Ordering::Relaxed);

                ui.checkbox(&mut clip_freeze, "FREEZE clipmap (no re-centre/invalidate)");

                if let Ok(clip) = clipmap_g.try_read() {
                    ClipmapDebugUi::show(ui, &clip, &mut clip_params);
                }
                clip_debug_flags.store(clip_params.debug_flags, Ordering::Relaxed);

                ui.separator();
                ui.label("Edit (LMB=dig  RMB=fill)");
                ui.add(egui::Slider::new(&mut dig_radius, 0.25_f32..=3.0).text("Radius (m)"));
            },
        );

        self.clip_params = clip_params;
        self.dig_radius = dig_radius;
        self.clip_freeze = clip_freeze;

        if let Some(is_dig) = self.pending_action.take() {
            if !egui_ctx.wants_pointer_input() {
                self.apply_edit(is_dig);
            }
        }

        self.ui.update_textures(&mut self.backend);

        let view = self.camera.view_matrix();
        let (w, h) = self.backend.window_size(self.window).unwrap_or((1280, 720));
        let near = 0.1f32;
        let far = 500.0f32;
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
        match event {
            Event::MouseDown { button: 1, .. } => self.pending_action = Some(true),
            Event::MouseDown { button: 2, .. } => self.pending_action = Some(false),
            _ => {}
        }
    }
}

impl SdfApp {
    fn apply_edit(&mut self, is_dig: bool) {
        use i3_math::Transform;
        use i3_math::nalgebra::{Point3, UnitQuaternion, Vector3};
        use i3_voxel::{SdfPrimitive, SdfScene};

        let yaw = self.camera.yaw;
        let pitch = self.camera.pitch;
        let forward = glm::vec3(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        );
        let ro = self.camera.position;

        let hit = {
            let scene = self.sdf_scene.read().unwrap();
            let big = AABB::new(
                Point3::new(-200.0, -200.0, -200.0),
                Point3::new(200.0, 200.0, 200.0),
            );
            let nodes = scene.get_nodes(&big);
            let mut t = 0.2f32;
            loop {
                let p = ro + forward * t;
                let d = SdfScene::sample(&nodes, &Point3::new(p.x, p.y, p.z)).value;
                if d < 0.01 {
                    break Some(p);
                }
                t += d.max(0.01).min(1.0);
                if t > 60.0 {
                    break None;
                }
            }
        };

        if let Some(hit) = hit {
            let xf = Transform::new(
                Vector3::new(hit.x, hit.y, hit.z),
                UnitQuaternion::identity(),
                1.0,
            );
            let sphere = SdfPrimitive::Sphere {
                radius: self.dig_radius,
            };
            let center = [hit.x, hit.y, hit.z];

            if !is_dig {
                let thr_sq = (self.dig_radius * 1.5f32).powi(2);
                self.edits.retain(|e| {
                    if !e.subtract {
                        return true;
                    }
                    let tx = e.transform.translation();
                    let d2 =
                        (tx.x - hit.x).powi(2) + (tx.y - hit.y).powi(2) + (tx.z - hit.z).powi(2);
                    d2 > thr_sq
                });
                *self.sdf_scene.write().unwrap() = build_full_scene(&self.edits);
            } else {
                let mut scene = self.sdf_scene.write().unwrap();
                scene.sub(&xf, &sphere);
                scene.build_bvh();
                drop(scene);
                self.edits.push(EditOp {
                    transform: xf,
                    primitive: sphere,
                    subtract: true,
                });
            }

            let inv_radius = if is_dig {
                self.dig_radius
            } else {
                self.dig_radius * 2.5
            };
            let scene = self.sdf_scene.read().unwrap();
            self.clipmap.write().unwrap().invalidate_sphere(center, inv_radius, &scene);
            tracing::info!(
                "{} at ({:.2}, {:.2}, {:.2}) r={:.2}",
                if is_dig { "DIG" } else { "FILL" },
                hit.x,
                hit.y,
                hit.z,
                self.dig_radius
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
        title: "Terrain".to_string(),
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

    for (cat, blob) in [("system.i3c", "system.i3b"), ("terrain.i3c", "terrain.i3b")] {
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

    // ── Clipmap setup ─────────────────────────────────────────────────────────
    let sdf_scene = Arc::new(RwLock::new(build_sdf_scene()));
    let clipmap = Arc::new(RwLock::new(ClipmapState::new()));
    let gpu_buffers = Arc::new(ClipmapBuffers::new(&mut backend));
    let clip_enabled = Arc::new(AtomicBool::new(true));
    let clip_debug_flags = Arc::new(AtomicU32::new(0));

    // ── Render graph ──────────────────────────────────────────────────────────
    let mut render_graph = DefaultRenderGraph::new(&mut backend, &config);
    render_graph.ao_mode = AoMode::None;
    render_graph.publish("UiSystem", ui.clone());

    // ── Terrain material textures → bindless indices (layer order: grass/rock/snow/dirt)
    let terrain_mat = {
        use examples_common::basic_scene::BasicScene;
        use i3_io::texture::TextureAsset;
        let mut tex = |name: &str| -> i32 {
            match loader.load::<TextureAsset>(name).wait_loaded() {
                Ok(t) => BasicScene::upload_and_register_texture(
                    &mut backend,
                    &mut render_graph.bindless_manager,
                    &t,
                ),
                Err(e) => {
                    tracing::warn!("terrain texture '{name}' not loaded: {e}");
                    -1
                }
            }
        };
        let mut tm = i3_voxel::TerrainMatParams::default();
        for (i, layer) in ["grass", "rock", "snow", "dirt"].iter().enumerate() {
            tm.albedo[i] = tex(&format!("terrain_{layer}_albedo.png"));
            tm.normal[i] = tex(&format!("terrain_{layer}_normal.png"));
            tm.orm[i] = tex(&format!("terrain_{layer}_orm.png"));
        }
        tm
    };

    render_graph.publish("AssetLoader", loader);

    let (clip_compute, clip_render) = create_clipmap_passes(
        clipmap.clone(),
        sdf_scene.clone(),
        gpu_buffers.clone(),
        clip_debug_flags.clone(),
        clip_enabled.clone(),
        terrain_graph().compile(),
        terrain_mat,
    );
    for pass in clip_compute {
        render_graph.extra_pre_gbuffer_passes.push(pass);
    }
    render_graph.extra_gbuffer_passes.push(clip_render);

    render_graph.init(&mut backend);

    brickmap_validate::run();

    // ── Camera ────────────────────────────────────────────────────────────────
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
        dt: 0.016,
        smoothed_dt: 0.016,
        time: 0.0,
        debug_gui: RendererDebugGui::new(),
        clip_enabled,
        clip_debug_flags,
        clip_params: ClipmapParams::default(),
        clipmap,
        sdf_scene,
        clip_freeze: false,
        dig_radius: 1.0,
        pending_action: None,
        edits: Vec::new(),
    });

    Ok(())
}
