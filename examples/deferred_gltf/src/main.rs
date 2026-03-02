extern crate nalgebra_glm;

use examples_common::basic_scene::BasicScene;
use examples_common::gltf_loader;
use examples_common::{ExampleApp, init_tracing, main_loop};
use i3_gfx::prelude::*;
use i3_renderer::render_graph::{DefaultRenderGraph, RenderConfig};
use i3_vulkan_backend::VulkanBackend;
use nalgebra_glm as glm;
use std::time::Duration;
use tracing::{error, info, warn};

struct DeferredGltfApp {
    backend: VulkanBackend,
    window: WindowHandle,
    render_graph: DefaultRenderGraph,
    scene: BasicScene,
    time: f32,
    dt: f32,
    camera: examples_common::camera_controller::CameraController,
}

impl ExampleApp for DeferredGltfApp {
    fn update(&mut self, delta: Duration) {
        self.dt = delta.as_secs_f32();
        self.time += self.dt;
        self.camera.update(delta);
    }

    fn render(&mut self) {
        let view = self.camera.view_matrix();
        let (width, height) = self.backend.window_size(self.window).unwrap_or((1280, 720));
        let projection = glm::perspective_rh_zo(
            width as f32 / height as f32,
            std::f32::consts::FRAC_PI_4,
            0.1,
            1000.0,
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
            1000.0,
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
        let events = self.backend.poll_events();
        for event in &events {
            self.camera.handle_event(event);
        }
        events
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = init_tracing("deferred_gltf.log");
    info!("Starting Deferred glTF Demo");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        error!("Usage: {} <path-to-gltf-or-glb>", args[0]);
        std::process::exit(1);
    }
    let gltf_path = std::path::Path::new(&args[1]);

    if !gltf_path.exists() {
        error!("File not found: {:?}", gltf_path);
        std::process::exit(1);
    }

    // 1. Initialize Backend
    let mut backend = VulkanBackend::new()?;
    backend.initialize(0)?;

    // 2. Create Window
    let window = backend.create_window(WindowDesc {
        title: "Deferred glTF".to_string(),
        width: 1280,
        height: 720,
    })?;

    // 3. Load scene from glTF
    info!("Loading glTF from {:?}", gltf_path);
    let scene = match gltf_loader::load_gltf(gltf_path, &mut backend) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to load glTF: {}", e);
            std::process::exit(1);
        }
    };

    // 4. Create Render Graph
    let config = RenderConfig {
        width: 1280,
        height: 720,
    };
    let render_graph = DefaultRenderGraph::new(&mut backend, &config);

    // 6. Run
    let app = DeferredGltfApp {
        backend,
        window,
        render_graph,
        scene,
        time: 0.0,
        dt: 0.016,
        camera: examples_common::camera_controller::CameraController::new(),
    };
    main_loop(app);

    Ok(())
}
