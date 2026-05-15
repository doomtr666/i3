#![allow(dead_code)]
#![allow(unused_variables)]
use libnoise::prelude::*;
use nalgebra::{UnitQuaternion, vector};

use examples_common::basic_scene::BasicScene;
use examples_common::camera_controller::CameraController;
use examples_common::{AppRenderer, ExampleApp, init_renderer, init_tracing, main_loop};
use i3_egui::prelude::*;
use i3_gfx::prelude::*;
use i3_io::mesh::BoundingBox;
use i3_io::prelude::*;
use i3_renderer::prelude::*;
use i3_renderer::scene::{ObjectData, ObjectId};
use i3_voxel::{SdfPrimitive, SdfScene, Transform, VoxelOctree, VoxelSceneSink, VoxelVertex};
use i3_vulkan_backend::prelude::*;
use nalgebra::Point3;
use nalgebra_glm as glm;
use std::f32::consts::FRAC_PI_4;
use std::sync::Arc;
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
            material_id: 0,
            flags: 0,
            _pad: 0,
        });
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
    dt: f32,
    show_debug_draw: bool,
}

impl ExampleApp for VoxelApp {
    fn update(&mut self, delta: Duration, _smoothed: Duration) {
        self.dt = delta.as_secs_f32();
        self.camera.update(delta);

        let p = self.camera.position;
        let cam_pos = Point3::new(p.x, p.y, p.z);
        let mut sink = VoxelSink {
            backend: &mut self.backend,
            scene: &mut self.scene,
        };
        self.voxel_octree.update(cam_pos, &mut sink, FRAME_BUDGET);
    }

    fn render(&mut self) {
        self.ui.begin_frame();
        let egui_ctx = self.ui.context().clone();

        egui::Window::new("Voxel").show(&egui_ctx, |ui| {
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
            ui.checkbox(&mut self.show_debug_draw, "Debug AABB");
            ui.separator();
            if self.camera.camera_locked {
                ui.label("Camera: LOCKED  (Tab to unlock)");
            } else {
                ui.label("Camera: FREE   (Tab to lock)");
            }
        });

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
    let _guard = init_tracing("voxel.log");

    let loader = Arc::new(AssetLoader::new(Arc::new(Vfs::new())));
    let AppRenderer {
        backend,
        window,
        render_graph,
        ui,
    } = init_renderer("Voxel", 1280, 720, Some(loader))?;

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
    let amplitude = 3000.0_f32;
    let terrain_half_y = 4000.0_f32; // doit être > amplitude

    // FBM : 9 octaves pour avoir du détail sur plusieurs km
    let generator = Source::perlin(42).fbm(9, 1.0, 2.0, 0.5);

    let mut sdf_scene = SdfScene::new();
    sdf_scene.add(
        &Transform::new(
            vector![half_xz, terrain_world_y, half_xz],
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
        scene: BasicScene::new(),
        voxel_octree,
        dt: 0.016,
        show_debug_draw: false,
    });

    Ok(())
}
