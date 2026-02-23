extern crate nalgebra_glm;

use examples_common::basic_scene::BasicScene;
use examples_common::{ExampleApp, init_tracing, main_loop};
use i3_gfx::prelude::*;
use i3_renderer::render_graph::{DefaultRenderGraph, RenderConfig};
use i3_renderer::scene::ObjectData;
use i3_slang::prelude::*;
use i3_vulkan_backend::VulkanBackend;
use nalgebra_glm as glm;
use std::time::Duration;
use tracing::{info, warn};

struct DeferredCubesApp {
    backend: VulkanBackend,
    window: WindowHandle,
    render_graph: DefaultRenderGraph,
    scene: BasicScene,
    time: f32,
    dt: f32,
    camera: examples_common::camera_controller::CameraController,
}

impl ExampleApp for DeferredCubesApp {
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
            100.0,
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
            100.0,
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

    // 3. Build scene
    let mut scene = BasicScene::new();
    let cube_mesh = scene.add_cube_mesh(&mut backend);
    scene.add_object(ObjectData {
        world_transform: glm::identity(),
        prev_transform: glm::identity(),
        material_id: 0,
        mesh_id: cube_mesh,
    });
    scene.add_default_lights();

    // 4. Compile Shaders
    let slang = SlangCompiler::new()?;
    let shader_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("crates/i3_renderer/shaders");
    let shader_path = shader_dir.to_str().unwrap();

    let gbuffer_shader = slang.compile_file("gbuffer", ShaderTarget::Spirv, &[shader_path])?;
    let debug_viz_shader = slang.compile_file("debug_viz", ShaderTarget::Spirv, &[shader_path])?;
    let cluster_build_shader =
        slang.compile_file("cluster_build", ShaderTarget::Spirv, &[shader_path])?;
    let light_cull_shader =
        slang.compile_file("light_cull", ShaderTarget::Spirv, &[shader_path])?;
    let histogram_build_shader =
        slang.compile_file("histogram_build", ShaderTarget::Spirv, &[shader_path])?;
    let average_luminance_shader =
        slang.compile_file("average_luminance", ShaderTarget::Spirv, &[shader_path])?;
    let deferred_resolve_shader =
        slang.compile_file("deferred_resolve", ShaderTarget::Spirv, &[shader_path])?;
    let tonemap_shader = slang.compile_file("tonemap", ShaderTarget::Spirv, &[shader_path])?;
    let sky_shader = slang.compile_file("sky", ShaderTarget::Spirv, &[shader_path])?;

    // 5. Create Render Graph
    let config = RenderConfig {
        width: 1280,
        height: 720,
    };
    let render_graph = DefaultRenderGraph::new(
        &mut backend,
        gbuffer_shader,
        debug_viz_shader,
        deferred_resolve_shader,
        cluster_build_shader,
        light_cull_shader,
        histogram_build_shader,
        average_luminance_shader,
        tonemap_shader,
        sky_shader,
        &config,
    );

    // 6. Run
    let app = DeferredCubesApp {
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
