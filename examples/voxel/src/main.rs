#![allow(dead_code)]
#![allow(unused_variables)]
use nalgebra::{UnitQuaternion, vector};

use examples_common::basic_scene::BasicScene;
use examples_common::camera_controller::CameraController;
use examples_common::{AppRenderer, ExampleApp, init_renderer, init_tracing, main_loop};
use i3_egui::prelude::*;
use i3_gfx::prelude::*;
use i3_io::mesh::BoundingBox;
use i3_io::prelude::*;
use i3_renderer::prelude::*;
use i3_renderer::scene::ObjectData;
use i3_voxel::{SdfPrimitive, SdfScene, Transform, VoxelScene, VoxelVertex};
use i3_vulkan_backend::prelude::*;
use nalgebra_glm as glm;
use std::f32::consts::FRAC_PI_4;
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

fn voxel_to_gbuffer(v: &VoxelVertex) -> [f32; 12] {
    let n = v.normal;
    // Build a tangent perpendicular to n using the "avoid parallel" trick.
    let up = if n.z.abs() < 0.9 { nalgebra::Vector3::z() } else { nalgebra::Vector3::x() };
    let t = up.cross(&n).normalize();
    [
        v.position.x,
        v.position.y,
        v.position.z,
        n.x,
        n.y,
        n.z,
        0.0,
        0.0, // uv — pas de texturing pour l'instant
        t.x,
        t.y,
        t.z,
        1.0, // tangent w = handedness
    ]
}

fn quads_to_tris(indices: &[u32]) -> Vec<u32> {
    indices
        .chunks(4)
        .flat_map(|q| [q[0], q[1], q[2], q[0], q[2], q[3]])
        .collect()
}

struct VoxelApp {
    backend: VulkanBackend,
    window: WindowHandle,
    render_graph: DefaultRenderGraph,
    ui: Arc<i3_egui::UiSystem>,
    camera: CameraController,
    scene: BasicScene,
    voxel_scene: VoxelScene,
    dt: f32,
    show_debug_draw: bool,
    show_debug_normals: bool,
}

impl ExampleApp for VoxelApp {
    fn update(&mut self, delta: Duration, _smoothed: Duration) {
        self.dt = delta.as_secs_f32();
        self.camera.update(delta);
    }

    fn render(&mut self) {
        self.ui.begin_frame();
        let egui_ctx = self.ui.context().clone();

        egui::Window::new("Voxel").show(&egui_ctx, |ui| {
            let [xmin, ymin, zmin] = self.voxel_scene.world_min();
            let [xmax, ymax, zmax] = self.voxel_scene.world_max();
            ui.label(format!(
                "Block  [{xmin:.1}, {ymin:.1}, {zmin:.1}] → [{xmax:.1}, {ymax:.1}, {zmax:.1}]"
            ));
            ui.label(format!(
                "Cell size: {} m   Grid: {}³",
                self.voxel_scene.voxel_dist(),
                self.voxel_scene.voxel_width(),
            ));
            ui.separator();
            ui.checkbox(&mut self.show_debug_draw, "Debug draw (AABB + wireframe)");
            ui.checkbox(&mut self.show_debug_normals, "Debug normals + vertices");
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
        let near = 0.05_f32;
        let far = 200.0_f32;
        let proj = glm::perspective_rh_zo(w as f32 / h as f32, FRAC_PI_4, far, near);

        self.render_graph.debug_draw_pass.clear();

        if self.show_debug_draw {
            for block in &self.voxel_scene {
                if block.get_packed_vertices().is_empty() {
                    continue;
                }
                self.render_graph.debug_draw_pass.push_aabb(
                    block.world_min(),
                    block.world_max(),
                    [0.2, 0.85, 1.0, 1.0],
                );

                let verts = block.get_packed_vertices();
                let col_wire = [0.2_f32, 1.0, 0.2, 1.0];
                for quad in block.get_packed_indices().chunks(4) {
                    let [i0, i1, i2, i3] = [quad[0], quad[1], quad[2], quad[3]];
                    let p = |i: u32| {
                        let v = &verts[i as usize].position;
                        [v.x, v.y, v.z]
                    };
                    self.render_graph
                        .debug_draw_pass
                        .push_line(p(i0), p(i1), col_wire);
                    self.render_graph
                        .debug_draw_pass
                        .push_line(p(i1), p(i2), col_wire);
                    self.render_graph
                        .debug_draw_pass
                        .push_line(p(i2), p(i3), col_wire);
                    self.render_graph
                        .debug_draw_pass
                        .push_line(p(i3), p(i0), col_wire);
                }

                if self.show_debug_normals {
                    let col_pt = [1.0_f32, 1.0, 0.0, 1.0];
                    let col_n = [1.0_f32, 0.2, 0.2, 1.0];
                    let scale = 0.04_f32;
                    for v in verts {
                        let p = [v.position.x, v.position.y, v.position.z];
                        let tip = [
                            v.position.x + v.normal.x * scale,
                            v.position.y + v.normal.y * scale,
                            v.position.z + v.normal.z * scale,
                        ];
                        self.render_graph
                            .debug_draw_pass
                            .push_cross(p, 0.01, col_pt);
                        self.render_graph.debug_draw_pass.push_line(p, tip, col_n);
                    }
                }
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
        mut backend,
        window,
        render_graph,
        ui,
    } = init_renderer("Voxel", 1280, 720, Some(loader))?;

    let mut camera = CameraController::new();
    camera.position = glm::vec3(1.6, 5.0, 7.0);
    camera.yaw = -std::f32::consts::FRAC_PI_2;
    camera.pitch = -0.56;
    camera.move_speed = 3.0;
    camera.camera_locked = true; // start with GUI visible

    let mut sdf_scene = SdfScene::new();
    sdf_scene.add(
        &Transform::new(vector![1.6, 5.0, 7.0], UnitQuaternion::identity(), 1.0),
        &SdfPrimitive::Box {
            half_extents: vector![1.33, 1.33, 1.33],
        },
    );
    sdf_scene.add(
        &Transform::new(vector![2.0, 3.0, 7.0], UnitQuaternion::identity(), 1.0),
        &SdfPrimitive::Sphere { radius: 2.0 },
    );

    let mut voxel_scene = VoxelScene::new(Arc::new(sdf_scene));
    voxel_scene.compute_meshes();

    let mut scene = BasicScene::new();
    for block in &voxel_scene {
        let verts = block.get_packed_vertices();
        let inds = block.get_packed_indices();
        if verts.is_empty() || inds.is_empty() {
            continue;
        }
        let gb_verts: Vec<[f32; 12]> = verts.iter().map(voxel_to_gbuffer).collect();
        let vb_bytes = bytemuck::cast_slice(&gb_verts);
        let tri_indices = quads_to_tris(inds);
        let aabb = BoundingBox {
            min: block.world_min(),
            max: block.world_max(),
        };
        let mesh_id = scene.add_mesh_u32(
            &mut backend,
            vb_bytes,
            verts.len() as u32,
            &tri_indices,
            aabb,
        );
        scene.add_object(ObjectData {
            world_transform: glm::identity(),
            prev_transform: glm::identity(),
            mesh_id,
            material_id: 0,
            flags: 0,
            _pad: 0,
        });
    }

    main_loop(VoxelApp {
        backend,
        window,
        render_graph,
        ui,
        camera,
        scene,
        voxel_scene,
        dt: 0.016,
        show_debug_draw: false,
        show_debug_normals: false,
    });

    Ok(())
}
