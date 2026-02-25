mod common;

use i3_gfx::graph::types::Format;
use i3_gfx::prelude::*;
use i3_null_backend::NullBackend;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// A simple pass for testing that calls a closure on execute.
struct TestPass<R, E> {
    name: String,
    record: R,
    execute: E,
}

impl<R, E> RenderPass for TestPass<R, E>
where
    R: FnMut(&mut PassBuilder) + Send + Sync + 'static,
    E: Fn(&mut dyn PassContext) + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        (self.record)(builder);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        (self.execute)(ctx);
    }
}

/// A group node for testing hierarchy.
struct TestGroup<R> {
    name: String,
    record: R,
}

impl<R> RenderPass for TestGroup<R>
where
    R: FnMut(&mut PassBuilder) + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        (self.record)(builder);
    }
}

#[test]
fn test_triangle_frame_flow() {
    common::init_test_tracing();

    let mut backend = NullBackend::new();
    let mut graph = FrameGraph::new();

    let window = WindowHandle(0);
    let execution_count = Arc::new(AtomicU32::new(0));
    let exec_count_clone = execution_count.clone();

    graph.record(move |builder| {
        let backbuffer = builder.acquire_backbuffer(window);
        builder.publish("SharedHandle", backbuffer);

        builder.add_pass(TestPass {
            name: "ClearPass".to_string(),
            record: |sub: &mut PassBuilder| {
                let shared = *sub.consume::<ImageHandle>("SharedHandle");
                sub.write_image(shared, ResourceUsage::COLOR_ATTACHMENT);
            },
            execute: move |_ctx: &mut dyn PassContext| {
                tracing::info!("Mock clear execution");
                exec_count_clone.fetch_add(1, Ordering::SeqCst);
            },
        });
    });

    let compiled = graph.compile();
    compiled.execute(&mut backend, None).unwrap();

    assert_eq!(execution_count.load(Ordering::SeqCst), 1);
}

#[test]
fn test_complex_hierarchical_graph() {
    common::init_test_tracing();
    let mut backend = NullBackend::new();
    let mut graph = FrameGraph::new();
    let window = WindowHandle(1);

    let gbuffer_exec = Arc::new(AtomicU32::new(0));
    let gbuffer_exec_clone = gbuffer_exec.clone();

    graph.record(move |builder| {
        let main_target = builder.declare_image(
            "MainTarget",
            ImageDesc::new(1920, 1080, Format::R8G8B8A8_UNORM),
        );
        let backbuffer = builder.acquire_backbuffer(window);

        builder.publish("GlobalSettings", String::from("UltraQuality"));

        builder.add_pass(TestGroup {
            name: "SceneGroup".to_string(),
            record: move |scene: &mut PassBuilder| {
                let settings = scene.consume::<String>("GlobalSettings").clone();
                tracing::info!("Scene recording with settings: {}", settings);

                let gbuffer_exec_inner = gbuffer_exec_clone.clone();
                scene.add_pass(TestPass {
                    name: "GBufferPass".to_string(),
                    record: move |sub: &mut PassBuilder| {
                        sub.write_image(main_target, ResourceUsage::COLOR_ATTACHMENT);
                    },
                    execute: move |_ctx: &mut dyn PassContext| {
                        tracing::info!("Executing GBufferPass");
                        gbuffer_exec_inner.fetch_add(1, Ordering::SeqCst);
                    },
                });

                scene.add_pass(TestPass {
                    name: "LightingPass".to_string(),
                    record: move |sub: &mut PassBuilder| {
                        sub.read_image(main_target, ResourceUsage::SHADER_READ);
                    },
                    execute: move |ctx: &mut dyn PassContext| {
                        tracing::info!("Executing LightingPass");
                        ctx.draw(3, 0);
                    },
                });
            },
        });

        builder.add_pass(TestPass {
            name: "PostPass".to_string(),
            record: move |post: &mut PassBuilder| {
                post.read_image(main_target, ResourceUsage::SHADER_READ);
                post.write_image(backbuffer, ResourceUsage::COLOR_ATTACHMENT);
            },
            execute: |_ctx: &mut dyn PassContext| {
                tracing::info!("Executing PostPass");
            },
        });
    });

    let compiled = graph.compile();
    compiled.execute(&mut backend, None).unwrap();
    assert_eq!(gbuffer_exec.load(Ordering::SeqCst), 1);
}

#[test]
fn test_modular_resource_lifecycle() {
    common::init_test_tracing();
    let mut backend = NullBackend::new();
    let mut graph = FrameGraph::new();
    let window = WindowHandle(42);

    graph.record(move |builder| {
        let backbuffer = builder.acquire_backbuffer(window);
        let scene_texture = builder.declare_image(
            "SceneTex",
            ImageDesc::new(1920, 1080, Format::R8G8B8A8_UNORM),
        );

        builder.add_pass(TestGroup {
            name: "SceneGroup".to_string(),
            record: move |scene: &mut PassBuilder| {
                scene.add_pass(TestPass {
                    name: "GBufferPass".to_string(),
                    record: move |sub: &mut PassBuilder| {
                        sub.write_image(scene_texture, ResourceUsage::COLOR_ATTACHMENT);
                    },
                    execute: |_ctx: &mut dyn PassContext| {
                        _ctx.draw(1000, 0);
                    },
                });

                scene.add_pass(TestPass {
                    name: "LightingPass".to_string(),
                    record: move |sub: &mut PassBuilder| {
                        sub.read_image(scene_texture, ResourceUsage::SHADER_READ);
                    },
                    execute: |ctx: &mut dyn PassContext| {
                        ctx.draw(3, 0);
                    },
                });
            },
        });

        builder.add_pass(TestPass {
            name: "FinalBlit".to_string(),
            record: move |sub: &mut PassBuilder| {
                sub.read_image(scene_texture, ResourceUsage::SHADER_READ);
                sub.write_image(backbuffer, ResourceUsage::COLOR_ATTACHMENT);
            },
            execute: |_ctx: &mut dyn PassContext| {},
        });
    });

    let compiled = graph.compile();
    compiled.execute(&mut backend, None).unwrap();
}
