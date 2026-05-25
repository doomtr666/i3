#![allow(dead_code)]
#![allow(unused_variables)]
use i3_noise::FbmBuilder;
use nalgebra::{UnitQuaternion, vector};
use std::sync::Arc;

use examples_common::basic_scene::BasicScene;
use examples_common::camera_controller::CameraController;
use examples_common::{ExampleApp, RendererDebugGui, main_loop};
use i3_gfx::prelude::*;
use i3_io::mesh::BoundingBox;
use i3_io::prelude::*;
use i3_renderer::prelude::*;
use i3_renderer::scene::{MaterialData, ObjectData, ObjectId};
use i3_voxel::{SdfPrimitive, SdfScene, Transform, VoxelOctree, VoxelSceneSink, VoxelVertex};
use i3_vulkan_backend::prelude::*;
use nalgebra::Point3;
use nalgebra_glm as glm;
use std::f32::consts::FRAC_PI_4;
mod voxel_gbuffer_pass;
use voxel_gbuffer_pass::VoxelGBufferPass;

use std::time::Duration;
use tracing::warn;

// ─── Terrain parameters ───────────────────────────────────────────────────────

// 10 LOD levels.
// root_half_size = 31 * 0.05 * 2^10 / 2 = 793.6 m  →  block_size = 1587.2 m
// Grid [7, 1, 7]: 7 * 1587 ≈ 11.1 km in XZ,  1 * 1587 ≈ 1.6 km in Y   (49 root blocks)
//
// Voxel dist per depth:
//   depth 0 → 51.2 m   (one voxel = size of a building block, visible at ~3 km)
//   depth 3 → 6.4 m    (visible at ~350 m)
//   depth 6 → 0.8 m    (visible at ~43 m)
//   depth 10 → 0.05 m  (finest, visible at ~2.7 m)
const MAX_DEPTH: u32 = 10;
const OCTREE_GRID: [u32; 3] = [7, 1, 7]; // XYZ root block counts
const SPLIT_FACTOR: f32 = 3.5;
const MERGE_HYSTERESIS: f32 = 1.5;
const FRAME_BUDGET: usize = 4;

// Derived constants
const ROOT_HALF_SIZE: f32 = 31.0 * 0.05 * (1u32 << MAX_DEPTH) as f32 * 0.5; // 793.6 m
const BLOCK_SIZE: f32 = ROOT_HALF_SIZE * 2.0; // 1587.2 m
const SCENE_XZ: f32 = OCTREE_GRID[0] as f32 * BLOCK_SIZE; // 11 110 m
const SCENE_Y: f32 = OCTREE_GRID[1] as f32 * BLOCK_SIZE; // 1 587 m

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn voxel_to_gbuffer(v: &VoxelVertex) -> [f32; 12] {
    let n = v.normal;
    let up = if n.z.abs() < 0.9 {
        nalgebra::Vector3::z()
    } else {
        nalgebra::Vector3::x()
    };
    let t = up.cross(&n).normalize();
    [
        v.position.x,
        v.position.y,
        v.position.z,
        n.x,
        n.y,
        n.z,
        0.0,
        0.0,
        t.x,
        t.y,
        t.z,
        1.0,
    ]
}

// ─── VoxelSink ────────────────────────────────────────────────────────────────

struct VoxelSink<'a> {
    backend: &'a mut VulkanBackend,
    scene: &'a mut BasicScene,
    grass_mat: u32,
    rock_mat: u32,
    dirt_mat: u32,
}

impl<'a> VoxelSceneSink for VoxelSink<'a> {
    fn add_mesh(
        &mut self,
        vertices: &[VoxelVertex],
        indices: &[u32],
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
    ) -> (u32, u64) {
        let gb_verts: Vec<[f32; 12]> = vertices.iter().map(voxel_to_gbuffer).collect();
        let vb_bytes = bytemuck::cast_slice(&gb_verts);
        let aabb = BoundingBox {
            min: aabb_min,
            max: aabb_max,
        };
        let mesh_id =
            self.scene
                .add_mesh_u32(self.backend, vb_bytes, vertices.len() as u32, indices, aabb);
        let object_id = self.scene.add_object(ObjectData {
            world_transform: glm::identity(),
            prev_transform: glm::identity(),
            mesh_id,
            material_id: 0xFFFFFFFE, // Special voxel material ID
            flags: self.grass_mat,
            _pad: self.rock_mat,
        });

        // Use an ugly bitcast to float for the 3rd material to stuff it in `_pad2` (or set a transform to 1.0 scale etc).
        // Actually, basic_scene's iter_instances does:
        // `_pad2: 0.0,`
        // So we can't easily modify `_pad2` from `ObjectData`.
        // Wait, ObjectData doesn't have `_pad2`. It has `flags` and `_pad`.
        // Where can we stuff the 3rd material?
        // We can pack grass and rock in `flags` (16 bits each) and dirt in `_pad` (32 bits).
        // Let's pack grass and rock into `flags`.

        let packed_flags = (self.grass_mat & 0xFFFF) | ((self.rock_mat & 0xFFFF) << 16);
        self.scene
            .set_voxel_materials(object_id, packed_flags, self.dirt_mat);

        (mesh_id, object_id.0)
    }

    fn remove_mesh(&mut self, mesh_id: u32, object_id: u64) {
        self.scene.remove_object(ObjectId(object_id));
        self.scene.remove_mesh(self.backend, mesh_id);
    }
}

// ─── VoxelApp ─────────────────────────────────────────────────────────────────

struct VoxelApp {
    backend: VulkanBackend,
    window: WindowHandle,
    render_graph: DefaultRenderGraph,
    ui: Arc<i3_egui::UiSystem>,
    camera: CameraController,
    scene: BasicScene,
    voxel_octree: VoxelOctree,
    grass_mat: u32,
    rock_mat: u32,
    dirt_mat: u32,
    dt: f32,
    smoothed_dt: f32,
    show_debug_draw: bool,
    debug_gui: RendererDebugGui,
}

impl ExampleApp for VoxelApp {
    fn update(&mut self, delta: Duration, smoothed: Duration) {
        self.dt = delta.as_secs_f32();
        self.smoothed_dt = smoothed.as_secs_f32();
        self.debug_gui.update(self.dt);
        self.camera.update(delta);

        let p = self.camera.position;
        let cam_pos = Point3::new(p.x, p.y, p.z);
        let view = self.camera.view_matrix();
        let (width, height) = self.backend.window_size(self.window).unwrap_or((1280, 720));
        let projection = nalgebra_glm::perspective_rh_zo(
            width as f32 / height as f32,
            std::f32::consts::FRAC_PI_4,
            1000.0,
            0.1,
        );
        let vp = projection * view;

        let mut sink = VoxelSink {
            backend: &mut self.backend,
            scene: &mut self.scene,
            grass_mat: self.grass_mat,
            rock_mat: self.rock_mat,
            dirt_mat: self.dirt_mat,
        };

        self.voxel_octree
            .update(cam_pos, &vp, &mut sink, FRAME_BUDGET);
    }

    fn render(&mut self) {
        self.ui.begin_frame();
        let egui_ctx = self.ui.context().clone();

        let show_debug_draw = &mut self.show_debug_draw;
        let camera_locked = self.camera.camera_locked;
        self.debug_gui.show(
            &egui_ctx,
            &mut self.render_graph,
            &self.camera,
            self.smoothed_dt,
            |ui| {
                ui.separator();
                ui.label(format!(
                    "Terrain  {:.1} × {:.1} × {:.1} km",
                    SCENE_XZ / 1000.0,
                    SCENE_Y / 1000.0,
                    SCENE_XZ / 1000.0,
                ));
                ui.label(format!(
                    "MAX_DEPTH {}  ({:.2} m – {:.1} m voxels)",
                    MAX_DEPTH,
                    0.05_f32,
                    0.05 * (1u32 << MAX_DEPTH) as f32,
                ));
                ui.label(format!("Budget {FRAME_BUDGET} blocks/frame"));
                ui.separator();
                ui.checkbox(show_debug_draw, "Debug AABB");
                ui.separator();
                if camera_locked {
                    ui.label("Camera: LOCKED  (Tab to unlock)");
                } else {
                    ui.label("Camera: FREE   (Tab to lock)");
                }
            },
        );

        self.ui.update_textures(&mut self.backend);

        let view = self.camera.view_matrix();
        let (w, h) = self.backend.window_size(self.window).unwrap_or((1280, 720));
        let near = 1.0_f32;
        let far = 20_000.0_f32;
        let proj = glm::perspective_rh_zo(w as f32 / h as f32, FRAC_PI_4, far, near);

        self.render_graph.debug_draw_pass.clear();

        if self.show_debug_draw {
            for aabb in self.voxel_octree.iter_node_aabbs() {
                let min = [aabb.min.x, aabb.min.y, aabb.min.z];
                let max = [aabb.max.x, aabb.max.y, aabb.max.z];
                self.render_graph
                    .debug_draw_pass
                    .push_aabb(min, max, [0.2, 0.85, 1.0, 1.0]);
            }
        }

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
    let _guard = examples_common::init_tracing("voxel.log");

    let mut backend = VulkanBackend::new()?;
    examples_common::maybe_list_gpus(&backend);
    backend.initialize(examples_common::get_gpu_index())?;

    let window = backend.create_window(WindowDesc {
        title: "Voxel".to_string(),
        width: 1280,
        height: 720,
    })?;

    let config = i3_renderer::render_graph::RenderConfig {
        width: 1280,
        height: 720,
    };
    let ui = Arc::new(i3_egui::UiSystem::new(1280, 720));

    let mut render_graph = DefaultRenderGraph::new(&mut backend, &config);
    render_graph.publish("UiSystem", ui.clone());

    let loader = Arc::new(AssetLoader::new(Arc::new(Vfs::new())));

    // Mount bundles
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_default();

    for (cat, blob) in [("system.i3c", "system.i3b"), ("voxel.i3c", "voxel.i3b")] {
        if exe_dir.join(cat).exists() {
            if let Ok(bundle) = i3_io::prelude::BundleBackend::mount(
                exe_dir.join(cat).to_str().unwrap(),
                exe_dir.join(blob).to_str().unwrap(),
            ) {
                let _ = loader.vfs().mount(Box::new(bundle));
            }
        }
    }
    render_graph.publish("AssetLoader", loader.clone());

    for key in loader.list_assets::<TextureAsset>() {
        tracing::info!("Loaded VFS Texture: {}", key);
    }

    // Inject our custom voxel GBuffer pass
    render_graph
        .extra_gbuffer_passes
        .push(Box::new(VoxelGBufferPass::new()));

    render_graph.init(&mut backend);

    let mut scene = BasicScene::new();

    // Load textures and create materials
    let mut load_mat = |name: &str| -> MaterialData {
        let albedo = BasicScene::load_texture_by_name(
            &mut backend,
            &mut render_graph.bindless_manager,
            &loader,
            &format!("{}_albedo.png", name),
        );
        let normal = BasicScene::load_texture_by_name(
            &mut backend,
            &mut render_graph.bindless_manager,
            &loader,
            &format!("{}_normal.png", name),
        );
        let orm = BasicScene::load_texture_by_name(
            &mut backend,
            &mut render_graph.bindless_manager,
            &loader,
            &format!("{}_orm.png", name),
        );
        MaterialData {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            emissive_factor_and_alpha_cutoff: [0.0, 0.0, 0.0, 0.0],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            _pad_pbr: [0.0; 2],
            albedo_tex_index: albedo,
            normal_tex_index: normal,
            rmao_tex_index: orm,
            emissive_tex_index: -1,
        }
    };

    let grass_mat = scene.add_material(load_mat("grass")).0;
    let rock_mat = scene.add_material(load_mat("rock")).0;
    let dirt_mat = scene.add_material(load_mat("dirt")).0;

    let mut camera = CameraController::new();
    let half_xz = SCENE_XZ * 0.5; // 5555 m — centre XZ
    let terrain_world_y = SCENE_Y * 0.5; // 794 m — centre de la colonne Y

    // Vue depuis la bordure, en hauteur, pour voir l'horizon
    camera.position = glm::vec3(half_xz, terrain_world_y + 400.0, SCENE_XZ + 500.0);
    camera.yaw = -std::f32::consts::FRAC_PI_2;
    camera.pitch = -0.18; // légèrement vers le bas
    camera.move_speed = 200.0; // 200 m/s — adapté à l'échelle
    camera.camera_locked = true;

    // ── SDF ───────────────────────────────────────────────────────────────────
    //
    // TerrainBox centré au milieu XZ de la scène, à mi-hauteur Y.
    // half_extents.x/z > half_xz pour éviter les artefacts DC aux bords.
    // amplitude = 300 m → reliefs de type collines/montagnes basses.
    let amplitude = 1000.0_f32;
    let terrain_half_y = 4000.0_f32; // doit être > amplitude

    let generator = Arc::new(
        FbmBuilder::new()
            .seed(0)
            .octaves(9)
            .lacunarity(2.0)
            .gain(0.5)
            .build(),
    );

    let mut sdf_scene = SdfScene::new();
    sdf_scene.add(
        &Transform::new(
            vector![half_xz, terrain_world_y + 1000.0, half_xz],
            UnitQuaternion::identity(),
            1.0,
        ),
        &SdfPrimitive::terrain_box(
            vector![half_xz + 200.0, terrain_half_y, half_xz + 200.0],
            amplitude,
            generator,
        ),
    );

    // Quelques cratères/cavernes pour rendre le terrain moins uniforme
    sdf_scene.sub(
        &Transform::new(
            vector![half_xz - 1200.0, terrain_world_y - 100.0, half_xz - 800.0],
            UnitQuaternion::identity(),
            1.0,
        ),
        &SdfPrimitive::Sphere { radius: 350.0 },
    );
    sdf_scene.sub(
        &Transform::new(
            vector![half_xz + 2000.0, terrain_world_y - 150.0, half_xz + 1500.0],
            UnitQuaternion::identity(),
            1.0,
        ),
        &SdfPrimitive::Sphere { radius: 500.0 },
    );

    sdf_scene.build_bvh();

    // ── Octree ────────────────────────────────────────────────────────────────
    let voxel_octree = VoxelOctree::new(
        Arc::new(sdf_scene),
        Point3::origin(),
        OCTREE_GRID,
        MAX_DEPTH,
        SPLIT_FACTOR,
        MERGE_HYSTERESIS,
    );

    main_loop(VoxelApp {
        backend,
        window,
        render_graph,
        ui,
        camera,
        scene,
        voxel_octree,
        grass_mat,
        rock_mat,
        dirt_mat,
        dt: 0.016,
        smoothed_dt: 0.016,
        show_debug_draw: false,
        debug_gui: RendererDebugGui::new(),
    });

    Ok(())
}
