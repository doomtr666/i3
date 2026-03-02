use i3_gfx::prelude::*;
use i3_slang::prelude::*;

/// Which GBuffer channel to display in the debug visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugChannel {
    Lit,
    LightDensity,
    ClusterGrid,
    Albedo,
    Normal,
    Roughness,
    Metallic,
    Emissive,
    Depth,
}

/// Push constants for the debug visualization pass.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DebugVizPushConstants {
    /// Channel selector (maps to DebugChannel enum as u32).
    pub channel: u32,
    /// Padding to align to 4 bytes.
    pub _pad: [u32; 3],
}

/// Records the debug visualization pass into the FrameGraph.
///
/// Draws a fullscreen triangle that samples the selected GBuffer channel
/// and writes to the backbuffer.
/// Debug visualization pass struct implementing the RenderPass trait.
pub struct DebugVizPass {
    pub backbuffer: ImageHandle,
    pub gbuffer_albedo: ImageHandle,
    pub gbuffer_normal: ImageHandle,
    pub gbuffer_roughmetal: ImageHandle,
    pub gbuffer_emissive: ImageHandle,
    pub sampler: SamplerHandle,
    pub channel: DebugChannel,

    // Persistence
    shader: Option<ShaderModule>,
    pipeline: Option<BackendPipeline>,
}

impl DebugVizPass {
    pub fn new(
        backbuffer: ImageHandle,
        gbuffer_albedo: ImageHandle,
        gbuffer_normal: ImageHandle,
        gbuffer_roughmetal: ImageHandle,
        gbuffer_emissive: ImageHandle,
        sampler: SamplerHandle,
        channel: DebugChannel,
    ) -> Self {
        Self {
            backbuffer,
            gbuffer_albedo,
            gbuffer_normal,
            gbuffer_roughmetal,
            gbuffer_emissive,
            sampler,
            channel,
            shader: None,
            pipeline: None,
        }
    }

    pub fn create_pipeline_info(&self) -> GraphicsPipelineCreateInfo {
        GraphicsPipelineCreateInfo {
            shader_module: self.shader.clone().expect("Shader not compiled"),
            vertex_input: VertexInputState::default(),
            render_targets: RenderTargetsInfo {
                color_targets: vec![RenderTargetInfo {
                    format: Format::B8G8R8A8_SRGB, // Backbuffer format
                    ..Default::default()
                }],
                depth_stencil_format: None,
                ..Default::default()
            },
            rasterization_state: RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

impl RenderPass for DebugVizPass {
    fn init(&mut self, backend: &mut dyn RenderBackend) {
        if self.pipeline.is_some() {
            return;
        }

        // 1. Compile Shader
        let slang = SlangCompiler::new().expect("Failed to create Slang compiler");
        let shader_dir = "crates/i3_renderer/shaders";

        self.shader = Some(
            slang
                .compile_file("debug_viz", ShaderTarget::Spirv, &[shader_dir])
                .expect("Failed to compile DebugViz shader"),
        );

        // 2. Create Pipeline
        let info = self.create_pipeline_info();
        self.pipeline = Some(backend.create_graphics_pipeline(&info));
    }

    fn name(&self) -> &str {
        "DebugVizPass"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        // Read GBuffer targets
        builder.read_image(self.gbuffer_albedo, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_emissive, ResourceUsage::SHADER_READ);

        // Write to backbuffer
        builder.write_image(self.backbuffer, ResourceUsage::COLOR_ATTACHMENT);

        // Bind GBuffer textures via push descriptors
        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.gbuffer_albedo,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(self.sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.gbuffer_normal,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(self.sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 2,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.gbuffer_roughmetal,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(self.sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 3,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.gbuffer_emissive,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(self.sampler),
                    }),
                },
            ],
        );
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let pipeline = self
            .pipeline
            .expect("DebugVizPass pipeline not initialized");
        ctx.bind_pipeline_raw(pipeline);
        let push = DebugVizPushConstants {
            channel: self.channel as u32,
            _pad: [0; 3],
        };
        ctx.push_constant_data(
            ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
            0,
            &push,
        );
        ctx.draw(3, 0); // Fullscreen triangle
        ctx.present(self.backbuffer);
    }
}
