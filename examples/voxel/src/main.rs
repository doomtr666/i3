#![allow(dead_code)]
#![allow(unused_variables)]
extern crate nalgebra as na;
use na::{point, vector};

mod sdf;
mod voxel;

use sdf::{BoxSdf, Sdf};
use voxel::VoxelScene;

use examples_common::basic_scene::BasicScene;
use examples_common::camera_controller::CameraController;
use examples_common::{AppRenderer, ExampleApp, init_renderer, init_tracing, main_loop};
use i3_egui::prelude::*;
use i3_gfx::prelude::*;
use i3_io::prelude::*;
use i3_renderer::prelude::*;
use i3_vulkan_backend::prelude::*;
use nalgebra_glm as glm;
use std::f32::consts::FRAC_PI_4;
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

struct VoxelApp {
    backend: VulkanBackend,
    window: WindowHandle,
    render_graph: DefaultRenderGraph,
    ui: Arc<i3_egui::UiSystem>,
    camera: CameraController,
    scene: BasicScene,
    voxel_scene: VoxelScene,
    dt: f32,
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

        for block in &self.voxel_scene {
            self.render_graph.debug_draw_pass.push_aabb(
                block.world_min(),
                block.world_max(),
                [0.2, 0.85, 1.0, 1.0],
            );

            // debug draw voxel vertices and normals
            let block_vertices = block.get_vertices();

            for vertex in block_vertices {
                match vertex {
                    Some(v) => {
                        self.render_graph.debug_draw_pass.push_cross(
                            [v.position.x, v.position.y, v.position.z],
                            0.02,
                            [1.0, 0.8, 0.1, 1.0],
                        );

                        let n_end = v.position + 0.05 * v.normal;

                        self.render_graph.debug_draw_pass.push_line(
                            [v.position.x, v.position.y, v.position.z],
                            [n_end.x, n_end.y, n_end.z],
                            [1.0, 0.0, 1.0, 1.0],
                        );
                    }
                    _ => continue,
                }
            }

            let verts = block.get_packed_vertices();
            let col = [0.2_f32, 1.0, 0.2, 1.0];
            for quad in block.get_packed_indices().chunks(4) {
                let [i0, i1, i2, i3] = [quad[0], quad[1], quad[2], quad[3]];
                let p = |i: u32| {
                    let v = &verts[i as usize].position;
                    [v.x, v.y, v.z]
                };
                self.render_graph
                    .debug_draw_pass
                    .push_line(p(i0), p(i1), col);
                self.render_graph
                    .debug_draw_pass
                    .push_line(p(i1), p(i2), col);
                self.render_graph
                    .debug_draw_pass
                    .push_line(p(i2), p(i3), col);
                self.render_graph
                    .debug_draw_pass
                    .push_line(p(i3), p(i0), col);
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
    let AppRenderer { backend, window, render_graph, ui } =
        init_renderer("Voxel", 1280, 720, Some(loader))?;

    let mut camera = CameraController::new();
    camera.position = glm::vec3(1.6, 5.0, 7.0);
    camera.yaw = -std::f32::consts::FRAC_PI_2;
    camera.pitch = -0.56;
    camera.move_speed = 3.0;
    camera.camera_locked = true; // start with GUI visible

    //let sdf = SphereSdf::new(point![1.6, 1.6, 1.6], 1.0);

    let sdf = BoxSdf::new(point![5.0, 2.17, 1.57], vector![3.0, 1.33, 1.33]);

    let mut voxel_scene = VoxelScene::new(Arc::new(sdf));
    voxel_scene.compute_meshes();

    //let mut block = VoxelBlock::new(Arc::new(sdf), 0, 0, 0);
    //block.compute_mesh();

    main_loop(VoxelApp {
        backend,
        window,
        render_graph,
        ui,
        camera,
        scene: BasicScene::new(),
        voxel_scene,
        dt: 0.016,
    });

    Ok(())
}
