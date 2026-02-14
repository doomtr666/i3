use examples_common::{ExampleApp, init_tracing, main_loop};
use i3_gfx::prelude::*;
use i3_slang::prelude::*;
use i3_vulkan_backend::VulkanBackend;
use std::time::Duration;

struct TriangleApp {
    backend: VulkanBackend,
    pipeline_id: SymbolId,
}

impl ExampleApp for TriangleApp {
    fn update(&mut self, _delta: Duration) {
        // Update logic (animations, etc.) would go here
    }

    fn render(&mut self) {
        // For MVP we re-compile and execute every frame
        // In the future, CompiledGraph should be persistent
        let mut graph = FrameGraph::new();
        let pipeline_id = self.pipeline_id;

        graph.record(move |builder| {
            builder.add_node("MainPass", move |_builder| {
                move |ctx| {
                    ctx.bind_pipeline(PipelineHandle(pipeline_id));
                    ctx.draw(3, 0);
                }
            });
        });

        let compiler = graph.compile();
        compiler.execute(&mut self.backend);
    }
}

fn main() -> Result<(), String> {
    // 1. Initialize Tracing from common
    init_tracing();

    // 2. Initialize Backend & Window
    let mut backend = VulkanBackend::new()?;
    let window = i3_vulkan_backend::window::VulkanWindow::new(
        backend.instance.clone(),
        "i3fx — Draw Triangle",
        1280,
        720,
    )?;
    backend.window = Some(window);
    backend.create_swapchain(0, 0);

    // 3. Obtain Event Pump
    let event_pump = backend.take_event_pump().unwrap();

    // 4. Create Pipeline Resources
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

    let pipeline_desc = GraphicsPipelineDesc {
        shader,
        name: "triangle_pipeline".to_string(),
    };
    let pipeline_handle = backend.create_graphics_pipeline(&pipeline_desc);
    let pipeline_id = SymbolId(pipeline_handle.0);

    // 5. Run Main Loop
    let app = TriangleApp {
        backend,
        pipeline_id,
    };
    main_loop(app, event_pump);

    Ok(())
}
