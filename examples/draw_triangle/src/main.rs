use i3_gfx::prelude::*;
use i3_slang::prelude::*;
use i3_vulkan_backend::VulkanBackend;
use tracing::info;

fn main() -> Result<(), String> {
    // 1. Initialize Tracing
    tracing_subscriber::fmt::init();
    info!("Starting Draw Triangle example...");

    // 2. Initialize Backend
    let mut backend = VulkanBackend::new()?;

    // 3. Create Window & Swapchain
    // In our simplified VulkanBackend, we create the window first
    let window = i3_vulkan_backend::window::VulkanWindow::new(
        backend.instance.clone(),
        "i3fx — Draw Triangle",
        1280,
        720,
    )?;
    backend.window = Some(window);

    // Create swapchain via backend
    backend.create_swapchain(0, 0); // Handles window internally for now

    // 4. Compile Shader
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

    // 5. Create Pipeline
    let pipeline_desc = GraphicsPipelineDesc {
        shader,
        name: "triangle_pipeline".to_string(),
    };
    let pipeline_handle = backend.create_graphics_pipeline(&pipeline_desc);
    let pipeline_id = pipeline_handle.0;

    // 6. Record & Execute Frame Graph
    let mut graph = FrameGraph::new();

    graph.record(move |builder| {
        builder.add_node("MainPass", move |_builder| {
            // We'll just execute basic commands here for the MVP
            move |ctx| {
                ctx.bind_pipeline(PipelineHandle(SymbolId(pipeline_id)));
                ctx.draw(3, 0);
            }
        });
    });

    let compiler = graph.compile();
    compiler.execute(&mut backend);

    info!("Example finished successfully");
    Ok(())
}
