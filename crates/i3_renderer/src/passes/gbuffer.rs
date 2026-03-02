use i3_gfx::prelude::*;
use i3_slang::prelude::*;

/// A single draw command extracted from the scene for the GBuffer pass.
#[derive(Clone, Copy, Debug)]
pub struct DrawCommand {
    pub mesh: crate::scene::Mesh,
    pub push_constants: GBufferPushConstants,
}

/// GBuffer vertex layout: position + normal + color.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GBufferVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

/// Push constants for the GBuffer pass.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GBufferPushConstants {
    pub view_projection: nalgebra_glm::Mat4,
    pub model: nalgebra_glm::Mat4,
}

/// GBuffer pass struct implementing the RenderPass trait.
pub struct GBufferPass {
    pub depth_buffer: ImageHandle,
    pub gbuffer_albedo: ImageHandle,
    pub gbuffer_normal: ImageHandle,
    pub gbuffer_roughmetal: ImageHandle,
    pub gbuffer_emissive: ImageHandle,

    // Persistence
    shader: Option<ShaderModule>,
    pipeline: Option<BackendPipeline>,
    draw_commands: Vec<DrawCommand>,
}

impl GBufferPass {
    pub fn new(
        depth_buffer: ImageHandle,
        gbuffer_albedo: ImageHandle,
        gbuffer_normal: ImageHandle,
        gbuffer_roughmetal: ImageHandle,
        gbuffer_emissive: ImageHandle,
    ) -> Self {
        Self {
            depth_buffer,
            gbuffer_albedo,
            gbuffer_normal,
            gbuffer_roughmetal,
            gbuffer_emissive,
            shader: None,
            pipeline: None,
            draw_commands: Vec::new(),
        }
    }

    /// Helper to create the pipeline info for this pass.
    fn create_pipeline_info(&self) -> GraphicsPipelineCreateInfo {
        GraphicsPipelineCreateInfo {
            shader_module: self.shader.clone().expect("Shader not compiled"),
            vertex_input: VertexInputState {
                bindings: vec![VertexInputBinding {
                    binding: 0,
                    stride: std::mem::size_of::<GBufferVertex>() as u32,
                    input_rate: VertexInputRate::Vertex,
                }],
                attributes: vec![
                    VertexInputAttribute {
                        location: 0,
                        binding: 0,
                        format: VertexFormat::Float3,
                        offset: 0,
                    },
                    VertexInputAttribute {
                        location: 1,
                        binding: 0,
                        format: VertexFormat::Float3,
                        offset: 12,
                    },
                    VertexInputAttribute {
                        location: 2,
                        binding: 0,
                        format: VertexFormat::Float3,
                        offset: 24,
                    },
                ],
            },
            render_targets: RenderTargetsInfo {
                color_targets: vec![
                    RenderTargetInfo {
                        format: Format::R8G8B8A8_SRGB,
                        ..Default::default()
                    },
                    RenderTargetInfo {
                        format: Format::R16G16_SFLOAT,
                        ..Default::default()
                    },
                    RenderTargetInfo {
                        format: Format::R8G8_UNORM,
                        ..Default::default()
                    },
                    RenderTargetInfo {
                        format: Format::R11G11B10_UFLOAT,
                        ..Default::default()
                    },
                ],
                depth_stencil_format: Some(Format::D32_FLOAT),
                logic_op: None,
            },
            rasterization_state: RasterizationState {
                cull_mode: CullMode::Back,
                front_face: FrontFace::CounterClockwise,
                ..Default::default()
            },
            depth_stencil_state: DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: true,
                depth_compare_op: CompareOp::Less,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &str {
        "GBufferPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend) {
        if self.pipeline.is_some() {
            return;
        }

        // 1. Compile Shader
        let slang = SlangCompiler::new().expect("Failed to create Slang compiler");
        // For now, we assume a relative path or fixed location for shaders.
        // In a real scenario, this might be configurable.
        let shader_dir = "crates/i3_renderer/shaders";

        self.shader = Some(
            slang
                .compile_file("gbuffer", ShaderTarget::Spirv, &[shader_dir])
                .expect("Failed to compile GBuffer shader"),
        );

        // 2. Create Pipeline
        let info = self.create_pipeline_info();
        self.pipeline = Some(backend.create_graphics_pipeline(&info));
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        // Consume draw commands from blackboard
        self.draw_commands = builder
            .consume::<Vec<DrawCommand>>("GBufferCommands")
            .clone();

        // Declare write targets
        builder.write_image(self.gbuffer_albedo, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_normal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_roughmetal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_emissive, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer, ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let pipeline = self.pipeline.expect("GBufferPass pipeline not initialized");
        ctx.bind_pipeline_raw(pipeline);

        for cmd in &self.draw_commands {
            ctx.push_constant_data(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                &cmd.push_constants,
            );

            let vb = BufferHandle(SymbolId(cmd.mesh.vertex_buffer.0));
            let ib = BufferHandle(SymbolId(cmd.mesh.index_buffer.0));

            ctx.bind_vertex_buffer(0, vb);
            ctx.bind_index_buffer(ib, IndexType::Uint16);
            ctx.draw_indexed(cmd.mesh.index_count, 0, 0);
        }
    }
}
