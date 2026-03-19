extern crate nalgebra_glm;

use examples_common::basic_scene::BasicScene;
use examples_common::{ExampleApp, init_tracing, main_loop};
use i3_gfx::prelude::*;
use i3_renderer::render_graph::{DefaultRenderGraph, RenderConfig};
use i3_renderer::scene::{LightData, LightType, ObjectData};
use i3_vulkan_backend::VulkanBackend;
use nalgebra_glm as glm;
use std::time::Duration;
use tracing::{info, warn};

struct DeferredStressApp {
    backend: VulkanBackend,
    window: WindowHandle,
    render_graph: DefaultRenderGraph,
    scene: BasicScene,
    time: f32,
    dt: f32,
    light_indices: Vec<i3_renderer::scene::LightId>,
    camera: examples_common::camera_controller::CameraController,
    is_fullscreen: bool,
}

impl ExampleApp for DeferredStressApp {
    fn update(&mut self, delta: Duration) {
        self.dt = delta.as_secs_f32();
        self.time += self.dt;

        // Update light positions and intensities to create flickering/movement
        for (i, &light_id) in self.light_indices.iter().enumerate() {
            let angle = self.time * 0.5 + (i as f32 * 0.2);
            let radius = 5.0 + (i as f32 % 5.0);
            let x = angle.cos() * radius;
            let z = angle.sin() * radius;
            let y = 1.0 + (angle * 0.7).sin() * 0.5;

            // Fluctuating intensity
            let intensity = 2.0 + (self.time * 3.0 + i as f32).sin() * 1.5;

            if let Some((_, light)) = self.scene.iter_lights_mut().find(|(id, _)| *id == light_id) {
                light.position = glm::vec3(x, y, z);
                light.intensity = intensity;
            }
        }

        self.camera.update(delta);
    }

    fn render(&mut self) {
        let view = self.camera.view_matrix();
        let (width, height) = self.backend.window_size(self.window).unwrap_or((1280, 720));
        let projection = glm::perspective_rh_zo(
            width as f32 / height as f32,
            std::f32::consts::FRAC_PI_4,
            0.1,
            200.0,
        );

        self.render_graph.sync(&mut self.backend, &self.scene);

        let mut graph = FrameGraph::new();
        self.render_graph.record(
            &mut graph,
            self.window,
            &self.scene,
            view,
            projection,
            0.1,
            200.0,
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
        self.camera.handle_event(event);

        if let Event::KeyDown { key } = event {
            if *key == KeyCode::F11 {
                self.is_fullscreen = !self.is_fullscreen;
                self.backend.set_fullscreen(self.window, self.is_fullscreen);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = init_tracing("deferred_stress.log");
    info!("Starting Deferred Stress Test");

    // 1. Initialize Backend
    let mut backend = VulkanBackend::new()?;
    examples_common::maybe_list_gpus(&backend);
    backend.initialize(examples_common::get_gpu_index())?;

    // 2. Create Window
    let window = backend.create_window(WindowDesc {
        title: "Deferred Stress Test".to_string(),
        width: 1280,
        height: 720,
    })?;

    // 3. Build scene
    let mut scene = BasicScene::new();
    let cube_mesh = scene.add_white_cube_mesh(&mut backend);

    // Grid of cubes
    for x in -10..=10 {
        for z in -10..=10 {
            let height = 0.5 + ((x as f32 * 0.5).sin() + (z as f32 * 0.5).cos()).abs() * 2.0;
            let transform = glm::translate(
                &glm::identity(),
                &glm::vec3(x as f32 * 1.5, height * 0.5, z as f32 * 1.5),
            ) * glm::scale(&glm::identity(), &glm::vec3(1.0, height, 1.0));

            scene.add_object(ObjectData {
                world_transform: transform,
                prev_transform: transform,
                material_id: 0,
                mesh_id: cube_mesh,
                flags: 0,
                _pad: 0,
            });
        }
    }

    // Floor
    let floor_transform = glm::translate(&glm::identity(), &glm::vec3(0.0, -0.5, 0.0))
        * glm::scale(&glm::identity(), &glm::vec3(100.0, 1.0, 100.0));
    scene.add_object(ObjectData {
        world_transform: floor_transform,
        prev_transform: floor_transform,
        material_id: 0,
        mesh_id: cube_mesh,
        flags: 0,
        _pad: 0,
    });

    // 256 Dynamic Point Lights
    let mut light_indices = Vec::new();
    for i in 0..256 {
        let h = i as f32 / 256.0;
        let color = hsv_to_rgb(h, 1.0, 1.0); // Full saturation

        let id = scene.add_light(LightData {
            position: glm::vec3(0.0, 2.0, 0.0),
            direction: glm::vec3(0.0, 0.0, 0.0),
            color,
            intensity: 4.0, // Increased intensity
            radius: 8.0,    // Slightly larger radius
            light_type: LightType::Point,
        });
        light_indices.push(id);
    }

    // Directional "Sun" - Dimmer to let point lights shine
    scene.add_light(LightData {
        position: glm::vec3(0.0, 0.0, 0.0),
        direction: glm::normalize(&glm::vec3(-1.0, -1.0, -1.0)),
        color: glm::vec3(1.0, 0.9, 0.8),
        intensity: 5.0,
        radius: 0.0,
        light_type: LightType::Directional,
    });

    // 4. Create Render Graph
    let config = RenderConfig {
        width: 1280,
        height: 720,
    };
    let mut render_graph = DefaultRenderGraph::new(&mut backend, &config);
    render_graph.init(&mut backend);

    // 6. Run
    let app = DeferredStressApp {
        backend,
        window,
        render_graph,
        scene,
        time: 0.0,
        dt: 0.016,
        light_indices,
        camera: examples_common::camera_controller::CameraController::new(),
        is_fullscreen: false,
    };
    main_loop(app);

    Ok(())
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> glm::Vec3 {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 1.0 / 6.0 {
        (c, x, 0.0)
    } else if h < 2.0 / 6.0 {
        (x, c, 0.0)
    } else if h < 3.0 / 6.0 {
        (0.0, c, x)
    } else if h < 4.0 / 6.0 {
        (0.0, x, c)
    } else if h < 5.0 / 6.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    glm::vec3(r + m, g + m, b + m)
}
