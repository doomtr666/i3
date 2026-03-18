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

        builder.add_owned_pass(TestPass {
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

        builder.add_owned_pass(TestGroup {
            name: "SceneGroup".to_string(),
            record: move |scene: &mut PassBuilder| {
                let settings = scene.consume::<String>("GlobalSettings").clone();
                tracing::info!("Scene recording with settings: {}", settings);

                let gbuffer_exec_inner = gbuffer_exec_clone.clone();
                scene.add_owned_pass(TestPass {
                    name: "ConsumerPass".to_string(),
                    record: move |sub: &mut PassBuilder| {
                        sub.write_image(main_target, ResourceUsage::COLOR_ATTACHMENT);
                    },
                    execute: move |_ctx: &mut dyn PassContext| {
                        tracing::info!("Executing GBufferPass");
                        gbuffer_exec_inner.fetch_add(1, Ordering::SeqCst);
                    },
                });

                scene.add_owned_pass(TestPass {
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

        builder.add_owned_pass(TestPass {
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

        builder.add_owned_pass(TestGroup {
            name: "SceneGroup".to_string(),
            record: move |scene: &mut PassBuilder| {
                scene.add_owned_pass(TestPass {
                    name: "ConsumerPass".to_string(),
                    record: move |sub: &mut PassBuilder| {
                        sub.write_image(scene_texture, ResourceUsage::COLOR_ATTACHMENT);
                    },
                    execute: |_ctx: &mut dyn PassContext| {
                        _ctx.draw(1000, 0);
                    },
                });

                scene.add_owned_pass(TestPass {
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

        builder.add_owned_pass(TestPass {
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

// ============================================================
// DAG dependency ordering tests
// ============================================================

/// Helper: records which order passes executed in using a shared atomic sequence counter.
fn make_order_pass(
    name: &str,
    seq: Arc<AtomicU32>,
    slot: Arc<AtomicU32>,
    record_fn: impl FnMut(&mut PassBuilder) + Send + Sync + 'static,
) -> TestPass<
    impl FnMut(&mut PassBuilder) + Send + Sync + 'static,
    impl Fn(&mut dyn PassContext) + Send + Sync + 'static,
> {
    TestPass {
        name: name.to_string(),
        record: record_fn,
        execute: move |_ctx: &mut dyn PassContext| {
            let order = seq.fetch_add(1, Ordering::SeqCst);
            slot.store(order, Ordering::SeqCst);
        },
    }
}

/// Diamond: A writes img → B reads img, A writes img → C reads img, B+C write img2 → D reads img2.
/// D must execute after both B and C.
#[test]
fn test_diamond_dependency_ordering() {
    common::init_test_tracing();
    let mut backend = NullBackend::new();
    let mut graph = FrameGraph::new();

    let seq = Arc::new(AtomicU32::new(0));
    let order_a = Arc::new(AtomicU32::new(u32::MAX));
    let order_b = Arc::new(AtomicU32::new(u32::MAX));
    let order_c = Arc::new(AtomicU32::new(u32::MAX));
    let order_d = Arc::new(AtomicU32::new(u32::MAX));

    let (s, oa, ob, oc, od) = (
        seq.clone(),
        order_a.clone(),
        order_b.clone(),
        order_c.clone(),
        order_d.clone(),
    );

    graph.record(move |builder| {
        let img = builder.declare_image("Shared", ImageDesc::new(64, 64, Format::R8G8B8A8_UNORM));
        let img2 = builder.declare_image("Shared2", ImageDesc::new(64, 64, Format::R8G8B8A8_UNORM));

        builder.add_owned_pass(make_order_pass(
            "A",
            s.clone(),
            oa.clone(),
            move |b: &mut PassBuilder| {
                b.write_image(img, ResourceUsage::COLOR_ATTACHMENT);
            },
        ));
        builder.add_owned_pass(make_order_pass(
            "B",
            s.clone(),
            ob.clone(),
            move |b: &mut PassBuilder| {
                b.read_image(img, ResourceUsage::SHADER_READ);
                b.write_image(img2, ResourceUsage::COLOR_ATTACHMENT);
            },
        ));
        builder.add_owned_pass(make_order_pass(
            "C",
            s.clone(),
            oc.clone(),
            move |b: &mut PassBuilder| {
                b.read_image(img, ResourceUsage::SHADER_READ);
                b.write_image(img2, ResourceUsage::COLOR_ATTACHMENT);
            },
        ));
        builder.add_owned_pass(make_order_pass(
            "D",
            s.clone(),
            od.clone(),
            move |b: &mut PassBuilder| {
                b.read_image(img2, ResourceUsage::SHADER_READ);
            },
        ));
    });

    let compiled = graph.compile();
    compiled.execute(&mut backend, None).unwrap();

    let a = order_a.load(Ordering::SeqCst);
    let b = order_b.load(Ordering::SeqCst);
    let c = order_c.load(Ordering::SeqCst);
    let d = order_d.load(Ordering::SeqCst);

    assert!(a < b, "A({a}) must execute before B({b})");
    assert!(a < c, "A({a}) must execute before C({c})");
    assert!(b < d, "B({b}) must execute before D({d})");
    assert!(c < d, "C({c}) must execute before D({d})");
}

/// Two independent passes with no shared resources — both must execute.
#[test]
fn test_independent_passes_both_execute() {
    common::init_test_tracing();
    let mut backend = NullBackend::new();
    let mut graph = FrameGraph::new();

    let exec_a = Arc::new(AtomicU32::new(0));
    let exec_b = Arc::new(AtomicU32::new(0));
    let ea = exec_a.clone();
    let eb = exec_b.clone();

    graph.record(move |builder| {
        let img_a = builder.declare_image("ImgA", ImageDesc::new(64, 64, Format::R8G8B8A8_UNORM));
        let img_b = builder.declare_image("ImgB", ImageDesc::new(64, 64, Format::R8G8B8A8_UNORM));

        builder.add_owned_pass(TestPass {
            name: "PassA".to_string(),
            record: move |b: &mut PassBuilder| {
                b.write_image(img_a, ResourceUsage::COLOR_ATTACHMENT);
            },
            execute: move |_: &mut dyn PassContext| {
                ea.fetch_add(1, Ordering::SeqCst);
            },
        });
        builder.add_owned_pass(TestPass {
            name: "PassB".to_string(),
            record: move |b: &mut PassBuilder| {
                b.write_image(img_b, ResourceUsage::COLOR_ATTACHMENT);
            },
            execute: move |_: &mut dyn PassContext| {
                eb.fetch_add(1, Ordering::SeqCst);
            },
        });
    });

    let compiled = graph.compile();
    compiled.execute(&mut backend, None).unwrap();

    assert_eq!(exec_a.load(Ordering::SeqCst), 1, "PassA must execute");
    assert_eq!(exec_b.load(Ordering::SeqCst), 1, "PassB must execute");
}

/// WAR: Pass B writes to a resource that Pass A reads.
/// B must execute after A (Write-After-Read hazard).
#[test]
fn test_war_dependency() {
    common::init_test_tracing();
    let mut backend = NullBackend::new();
    let mut graph = FrameGraph::new();

    let seq = Arc::new(AtomicU32::new(0));
    let order_reader = Arc::new(AtomicU32::new(u32::MAX));
    let order_writer = Arc::new(AtomicU32::new(u32::MAX));
    let (s, or, ow) = (seq.clone(), order_reader.clone(), order_writer.clone());

    graph.record(move |builder| {
        let buf = builder.declare_buffer(
            "SharedBuf",
            BufferDesc {
                size: 256,
                usage: BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::GpuOnly,
            },
        );

        // A reads the buffer first
        builder.add_owned_pass(make_order_pass(
            "Reader",
            s.clone(),
            or.clone(),
            move |b: &mut PassBuilder| {
                b.read_buffer(buf, ResourceUsage::SHADER_READ);
            },
        ));
        // B writes to the same buffer — WAR dependency
        builder.add_owned_pass(make_order_pass(
            "Writer",
            s.clone(),
            ow.clone(),
            move |b: &mut PassBuilder| {
                b.write_buffer(buf, ResourceUsage::SHADER_WRITE);
            },
        ));
    });

    let compiled = graph.compile();
    compiled.execute(&mut backend, None).unwrap();

    let r = order_reader.load(Ordering::SeqCst);
    let w = order_writer.load(Ordering::SeqCst);
    assert!(
        r < w,
        "Reader({r}) must execute before Writer({w}) — WAR hazard"
    );
}

/// WAW: Two passes write to the same resource.  
/// Second writer must execute after first writer.
#[test]
fn test_waw_dependency() {
    common::init_test_tracing();
    let mut backend = NullBackend::new();
    let mut graph = FrameGraph::new();

    let seq = Arc::new(AtomicU32::new(0));
    let order_w1 = Arc::new(AtomicU32::new(u32::MAX));
    let order_w2 = Arc::new(AtomicU32::new(u32::MAX));
    let (s, o1, o2) = (seq.clone(), order_w1.clone(), order_w2.clone());

    graph.record(move |builder| {
        let img = builder.declare_image("Target", ImageDesc::new(64, 64, Format::R8G8B8A8_UNORM));

        builder.add_owned_pass(make_order_pass(
            "Writer1",
            s.clone(),
            o1.clone(),
            move |b: &mut PassBuilder| {
                b.write_image(img, ResourceUsage::COLOR_ATTACHMENT);
            },
        ));
        builder.add_owned_pass(make_order_pass(
            "Writer2",
            s.clone(),
            o2.clone(),
            move |b: &mut PassBuilder| {
                b.write_image(img, ResourceUsage::COLOR_ATTACHMENT);
            },
        ));
    });

    let compiled = graph.compile();
    compiled.execute(&mut backend, None).unwrap();

    let w1 = order_w1.load(Ordering::SeqCst);
    let w2 = order_w2.load(Ordering::SeqCst);
    assert!(
        w1 < w2,
        "Writer1({w1}) must execute before Writer2({w2}) — WAW hazard"
    );
}

/// Linear chain: A→B→C with strict sequential ordering.
#[test]
fn test_linear_chain_ordering() {
    common::init_test_tracing();
    let mut backend = NullBackend::new();
    let mut graph = FrameGraph::new();

    let seq = Arc::new(AtomicU32::new(0));
    let order_a = Arc::new(AtomicU32::new(u32::MAX));
    let order_b = Arc::new(AtomicU32::new(u32::MAX));
    let order_c = Arc::new(AtomicU32::new(u32::MAX));
    let (s, oa, ob, oc) = (
        seq.clone(),
        order_a.clone(),
        order_b.clone(),
        order_c.clone(),
    );

    graph.record(move |builder| {
        let img1 = builder.declare_image("Stage1", ImageDesc::new(64, 64, Format::R8G8B8A8_UNORM));
        let img2 = builder.declare_image("Stage2", ImageDesc::new(64, 64, Format::R8G8B8A8_UNORM));

        builder.add_owned_pass(make_order_pass(
            "A",
            s.clone(),
            oa.clone(),
            move |b: &mut PassBuilder| {
                b.write_image(img1, ResourceUsage::COLOR_ATTACHMENT);
            },
        ));
        builder.add_owned_pass(make_order_pass(
            "B",
            s.clone(),
            ob.clone(),
            move |b: &mut PassBuilder| {
                b.read_image(img1, ResourceUsage::SHADER_READ);
                b.write_image(img2, ResourceUsage::COLOR_ATTACHMENT);
            },
        ));
        builder.add_owned_pass(make_order_pass(
            "C",
            s.clone(),
            oc.clone(),
            move |b: &mut PassBuilder| {
                b.read_image(img2, ResourceUsage::SHADER_READ);
            },
        ));
    });

    let compiled = graph.compile();
    compiled.execute(&mut backend, None).unwrap();

    let a = order_a.load(Ordering::SeqCst);
    let b = order_b.load(Ordering::SeqCst);
    let c = order_c.load(Ordering::SeqCst);
    assert!(a < b, "A({a}) must execute before B({b})");
    assert!(b < c, "B({b}) must execute before C({c})");
}

/// Test that CPU data dependencies (publish/consume) create DAG edges.
/// Two passes write the same data symbol (WAW). B must execute after A
/// even though they share no GPU resources.
#[test]
fn test_cpu_data_dependency_ordering() {
    let mut backend = NullBackend::new();
    backend.initialize(0).unwrap();

    let counter = Arc::new(AtomicU32::new(0));
    let order_a = Arc::new(AtomicU32::new(0));
    let order_b = Arc::new(AtomicU32::new(0));

    let s = counter.clone();
    let oa = order_a.clone();
    let s2 = counter.clone();
    let ob = order_b.clone();

    let mut graph = FrameGraph::new();
    graph.record(|builder| {
        // Pass A: CPU-only, publishes data
        builder.add_owned_pass(make_order_pass(
            "ProducerPass",
            s.clone(),
            oa.clone(),
            move |b: &mut PassBuilder| {
                b.publish("SharedData", 42u64);
            },
        ));

        // Pass B: also publishes SharedData (WAW dependency)
        builder.add_owned_pass(make_order_pass(
            "ConsumerPass",
            s2.clone(),
            ob.clone(),
            move |b: &mut PassBuilder| {
                b.publish("SharedData", 99u64);
            },
        ));
    });

    let compiled = graph.compile();
    compiled.execute(&mut backend, None).unwrap();

    let a = order_a.load(Ordering::SeqCst);
    let b = order_b.load(Ordering::SeqCst);
    assert!(
        a < b,
        "ProducerPass({a}) must execute before ConsumerPass({b}) — CPU data dependency"
    );
}
