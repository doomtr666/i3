mod common;

use i3_gfx::graph::types::Format;
use i3_gfx::prelude::*;
use i3_null_backend::NullBackend;

#[test]
fn test_triangle_frame_flow() {
    common::init_test_tracing();

    let mut backend = NullBackend::new();
    let mut graph = FrameGraph::new();

    let window = WindowHandle(0);

    graph.record(move |builder| {
        let backbuffer = builder.acquire_backbuffer(window);
        builder.publish("SharedHandle", backbuffer);

        builder.add_node("ClearPass", move |sub| {
            let shared = *sub.consume::<ImageHandle>("SharedHandle");
            sub.write_image(shared, ResourceUsage::COLOR_ATTACHMENT);
            |_ctx| {
                tracing::info!("Mock clear execution");
            }
        });
    });

    // 4. Compile & Execute
    let compiled = graph.compile();
    compiled.execute(&mut backend).unwrap();

    // Success criteria: Test completed without panic.
    // NullBackend logs (viewable with --nocapture) would show the calls.
}

#[test]
fn test_complex_hierarchical_graph() {
    common::init_test_tracing();
    let mut backend = NullBackend::new();
    let mut graph = FrameGraph::new();
    let window = WindowHandle(1);

    graph.record(move |builder| {
        // 1. Root level setup
        let main_target = builder.declare_image(
            "MainTarget",
            ImageDesc::new(1920, 1080, Format::R8G8B8A8_UNORM),
        );
        let backbuffer = builder.acquire_backbuffer(window);

        // Publish some shared CPU data
        builder.publish("GlobalSettings", String::from("UltraQuality"));

        // 2. Nested Group: Scene Rendering
        builder.add_node("SceneGroup", move |scene| {
            // Consume from root
            let settings = scene.consume::<String>("GlobalSettings").clone();
            tracing::info!("Scene recording with settings: {}", settings);

            let gbuffer_settings = settings.clone();
            scene.add_node("GBufferPass", move |sub| {
                sub.write_image(main_target, ResourceUsage::COLOR_ATTACHMENT);
                move |_ctx| {
                    tracing::info!("Executing GBufferPass with settings: {}", gbuffer_settings);
                }
            });

            scene.add_node("LightingPass", move |sub| {
                sub.read_image(main_target, ResourceUsage::SHADER_READ);

                // Consume the settings captured from parent record closure
                let local_settings = settings.clone();

                move |ctx| {
                    ctx.bind_image(0, main_target);
                    tracing::info!("Executing LightingPass with settings: {}", local_settings);
                    ctx.draw(3, 0); // Fullscreen quad
                }
            });

            |_ctx| {} // Group execution (noop)
        });

        // 3. Post Processing
        builder.add_node("PostPass", move |post| {
            post.read_image(main_target, ResourceUsage::SHADER_READ);
            post.write_image(backbuffer, ResourceUsage::COLOR_ATTACHMENT);
            |_ctx| {
                tracing::info!("Executing PostPass");
            }
        });
    });

    let compiled = graph.compile();
    compiled.execute(&mut backend).unwrap();
}

// Modular helper for Scene rendering
fn record_scene(scene: &mut PassBuilder, main_target: ImageHandle) {
    scene.add_node("GBufferPass", move |sub| {
        sub.write_image(main_target, ResourceUsage::COLOR_ATTACHMENT);
        move |_ctx| {
            _ctx.draw(1000, 0);
        }
    });

    scene.add_node("LightingPass", move |sub| {
        sub.read_image(main_target, ResourceUsage::SHADER_READ);
        move |ctx| {
            ctx.bind_image(0, main_target);
            ctx.draw(3, 0); // Fullscreen quad
        }
    });
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

        // Delegate to helper
        builder.add_node("SceneGroup", move |scene| {
            record_scene(scene, scene_texture);
            |_| {}
        });

        // Post Process
        builder.add_node("FinalBlit", move |sub| {
            sub.read_image(scene_texture, ResourceUsage::SHADER_READ);
            sub.write_image(backbuffer, ResourceUsage::COLOR_ATTACHMENT);
            move |ctx| {
                ctx.bind_image(0, scene_texture);
            }
        });
    });

    let compiled = graph.compile();
    compiled.execute(&mut backend).unwrap();
}
