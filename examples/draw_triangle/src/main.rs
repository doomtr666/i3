use examples_common::{ExampleApp, init_tracing, main_loop};
use i3_gfx::prelude::*;
use i3_slang::prelude::*;
use std::time::Duration;

use i3_vulkan_backend::VulkanBackend;
// use i3_null_backend::NullBackend;

struct TriangleApp {
    backend: VulkanBackend,
    pipeline_id: SymbolId,
    window: WindowHandle,
}

impl ExampleApp for TriangleApp {
    fn update(&mut self, _delta: Duration) {
        // Update logic (animations, etc.) would go here
    }

    fn render(&mut self) {
        // 1. Execute Graph (Swapchain is acquired internally)
        let mut graph = FrameGraph::new();
        let pipeline_id = self.pipeline_id;
        let window = self.window;

        graph.record(move |builder| {
            let backbuffer = builder.acquire_backbuffer(window);
            // Register the physical image for this virtual handle
            // This is a temporary way to map them until FrameGraph handles this automatically

            builder.add_node("MainPass", move |builder| {
                builder.bind_pipeline(PipelineHandle(pipeline_id));
                builder.write_image(backbuffer, ResourceUsage::COLOR_ATTACHMENT);

                move |ctx| {
                    ctx.draw(3, 0);
                    ctx.present(backbuffer); // Present via command
                }
            });
        });

        let compiler = graph.compile();
        match compiler.execute(&mut self.backend) {
            Ok(_) => {}
            Err(e) if e == "WindowMinimized" => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => panic!("Graph execution failed: {}", e),
        }
    }

    fn poll_events(&mut self) -> Vec<Event> {
        self.backend.poll_events()
    }
}

fn main() -> Result<(), String> {
    // 1. Initialize Tracing from common
    let _guard = init_tracing("draw_triangle.log");

    // 2. Initialize Backend & Window
    let mut backend = VulkanBackend::new()?;

    // Choose the first discrete GPU or first available
    // Note: enumerate_devices might act differently now, relying on new() default.
    // If we want specific device, we'd need to inspect instance.
    // backend.initialize(0)?; // initialize with default device for MVP

    backend.initialize(0)?;

    let window = backend.create_window(WindowDesc {
        title: "i3fx — Draw Triangle".to_string(),
        width: 1280,
        height: 720,
    })?;

    // Swapchain is configured implicitly with defaults (VSync: false, MinImage: 3)

    // 4. Create Pipeline Resources (requires initialized backend)
    let slang = SlangCompiler::new()?;
    let shader_code = r#"
        struct VSOutput {
            float4 pos : SV_Position;
            float3 color : COLOR;
        };

        [shader("vertex")]
        VSOutput vertexMain(uint vid : SV_VertexID) {
            float2 pos[3] = {
                float2(0.0, -0.5),
                float2(0.5, 0.5),
                float2(-0.5, 0.5)
            };
            float3 colors[3] = {
                float3(1.0, 0.0, 0.0),
                float3(0.0, 1.0, 0.0),
                float3(0.0, 0.0, 1.0)
            };

            VSOutput output;
            output.pos = float4(pos[vid], 0.0, 1.0);
            output.color = colors[vid];
            return output;
        }

        [shader("fragment")]
        float4 fragmentMain(VSOutput input) : SV_Target {
            return float4(input.color, 1.0);
        }
    "#;

    let shader = slang.compile_inline(
        "triangle",
        "triangle.slang",
        shader_code,
        ShaderTarget::Spirv,
    )?;

    // 3. Create Pipeline
    let pipeline_handle = backend.create_graphics_pipeline(&GraphicsPipelineCreateInfo {
        shader_module: shader,
        render_targets: RenderTargetsInfo {
            color_targets: vec![RenderTargetInfo {
                format: Format::B8G8R8A8_SRGB,
                ..Default::default()
            }],
            ..Default::default()
        },
        ..Default::default()
    });
    let pipeline_id = SymbolId(pipeline_handle.0);

    // 5. Run Main Loop
    let app = TriangleApp {
        backend,
        pipeline_id,
        window,
    };
    main_loop(app);

    Ok(())
}
