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
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

/// Push constants for the GBuffer pass.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GBufferPushConstants {
    pub view_projection: nalgebra_glm::Mat4,
    pub model: nalgebra_glm::Mat4,
    pub material_id: u32,
    pub _pad: [u32; 3],
}

impl Default for GBufferPushConstants {
    fn default() -> Self {
        Self {
            view_projection: nalgebra_glm::identity(),
            model: nalgebra_glm::identity(),
            material_id: 0,
            _pad: [0; 3],
        }
    }
}

/// GBuffer pass struct implementing the RenderPass trait.
pub struct GBufferPass {
    pub bindless_set: u64,

    // Resolved handles (updated in record)
    depth_buffer: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive: ImageHandle,
    material_buffer: BufferHandle,

    // Persistence
    shader: Option<ShaderModule>,
    pipeline: Option<BackendPipeline>,
    draw_commands: Vec<DrawCommand>,
    is_baked: bool,
}

impl GBufferPass {
    pub fn new() -> Self {
        let dummy_image = ImageHandle(SymbolId(0));
        let dummy_buffer = BufferHandle(SymbolId(0));
        Self {
            depth_buffer: dummy_image,
            gbuffer_albedo: dummy_image,
            gbuffer_normal: dummy_image,
            gbuffer_roughmetal: dummy_image,
            gbuffer_emissive: dummy_image,
            material_buffer: dummy_buffer,
            bindless_set: 0,
            shader: None,
            pipeline: None,
            draw_commands: Vec::new(),
            is_baked: false,
        }
    }

    pub fn init_from_baked(
        &mut self,
        backend: &mut dyn RenderBackend,
        asset: &i3_io::pipeline_asset::PipelineAsset,
    ) {
        if self.pipeline.is_some() {
            return;
        }

        let state = asset.state.as_ref().expect("GBuffer asset missing state");
        self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
            state,
            &asset.reflection_data,
            &asset.bytecode,
        ));
        self.is_baked = true;
    }
    
    // ... existing init ...

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
                        format: VertexFormat::Float2,
                        offset: 24,
                    },
                    VertexInputAttribute {
                        location: 3,
                        binding: 0,
                        format: VertexFormat::Float4,
                        offset: 32,
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
        self.is_baked = false;
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        // Resolve target handles by name
        self.gbuffer_albedo = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive = builder.resolve_image("GBuffer_Emissive");
        self.depth_buffer = builder.resolve_image("DepthBuffer");
        self.material_buffer = builder.resolve_buffer("MaterialBuffer");

        // Resolve bindless descriptor set from blackboard
        self.bindless_set = *builder.consume::<u64>("BindlessSet");

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

        builder.read_buffer(self.material_buffer, ResourceUsage::READ);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let pipeline = self.pipeline.expect("GBufferPass pipeline not initialized");
        ctx.bind_pipeline_raw(pipeline);

        // Bind Material SSBO at set 1
        let mat_set = ctx.create_descriptor_set(
            pipeline,
            1,
            &[DescriptorWrite::buffer(0, self.material_buffer)],
        );
        ctx.bind_descriptor_set(1, mat_set);

        // Bind Bindless Set at set 2
        ctx.bind_descriptor_set_raw(2, self.bindless_set);

        let expected_stride = std::mem::size_of::<GBufferVertex>() as u32;

        for cmd in &self.draw_commands {
            if cmd.mesh.stride != expected_stride {
                tracing::warn!(
                    "GBufferPass: Skipping mesh with incompatible stride {} (expected {})",
                    cmd.mesh.stride,
                    expected_stride
                );
                continue;
            }

            ctx.push_constant_data(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                &cmd.push_constants,
            );

            let vb = BufferHandle(SymbolId(cmd.mesh.vertex_buffer.0));
            let ib = BufferHandle(SymbolId(cmd.mesh.index_buffer.0));

            ctx.bind_vertex_buffer(0, vb);
            ctx.bind_index_buffer(ib, cmd.mesh.index_type);
            ctx.draw_indexed(cmd.mesh.index_count, 0, 0);
        }
    }
}
