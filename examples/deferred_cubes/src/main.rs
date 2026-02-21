extern crate nalgebra_glm;

use examples_common::{init_tracing, main_loop, ExampleApp};
use i3_gfx::prelude::*;
use i3_renderer::render_graph::{DefaultRenderGraph, RenderConfig};
use i3_slang::prelude::*;
use i3_vulkan_backend::VulkanBackend;
use nalgebra_glm as glm;
use std::time::Duration;
use tracing::{info, warn};

struct DeferredCubesApp {
    backend: VulkanBackend,
    window: WindowHandle,
    render_graph: DefaultRenderGraph,
    time: f32,
}

impl ExampleApp for DeferredCubesApp {
    fn update(&mut self, delta: Duration) {
        self.time += delta.as_secs_f32();
    }

    fn render(&mut self) {
        // Orbiting camera
        let eye = glm::vec3(3.0 * self.time.cos(), 2.0, 3.0 * self.time.sin());
        let target = glm::vec3(0.0, 0.0, 0.0);
        let up = glm::vec3(0.0, 1.0, 0.0);

        let view = glm::look_at_rh(&eye, &target, &up);
        let projection =
            glm::perspective_rh_zo(1280.0 / 720.0, std::f32::consts::FRAC_PI_4, 0.1, 100.0);
        let vp = projection * view;

        let mut graph = FrameGraph::new();
        self.render_graph.record(&mut graph, self.window, vp);

        let compiler = graph.compile();
        if let Err(e) = compiler.execute(&mut self.backend) {
            warn!("Graph execution failed: {}", e);
        }
    }

    fn poll_events(&mut self) -> Vec<Event> {
        self.backend.poll_events()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = init_tracing("deferred_cubes.log");
    info!("Starting Deferred Cubes Demo");

    // 1. Initialize Backend
    let mut backend = VulkanBackend::new()?;
    backend.initialize(0)?;

    // 2. Create Window
    let window = backend.create_window(WindowDesc {
        title: "Deferred Cubes".to_string(),
        width: 1280,
        height: 720,
    })?;

    // 3. Compile Shaders
    let slang = SlangCompiler::new()?;
    let shader_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("crates/i3_renderer/shaders");
    let shader_path = shader_dir.to_str().unwrap();

    let gbuffer_shader = slang.compile_file("gbuffer", ShaderTarget::Spirv, &[shader_path])?;
    let debug_viz_shader = slang.compile_file("debug_viz", ShaderTarget::Spirv, &[shader_path])?;

    // 4. Create Render Graph
    let config = RenderConfig {
        width: 1280,
        height: 720,
    };
    let render_graph =
        DefaultRenderGraph::new(&mut backend, gbuffer_shader, debug_viz_shader, &config);

    // 5. Run
    let app = DeferredCubesApp {
        backend,
        window,
        render_graph,
        time: 0.0,
    };
    main_loop(app);

    Ok(())
}
