use i3_gfx::prelude::*;

/// Which GBuffer channel to display in the debug visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugChannel {
    Albedo = 0,
    Normal = 1,
    Roughness = 2,
    Metallic = 3,
    Emissive = 4,
    Depth = 5,
    Lit = 10,
    LightDensity = 11,
    ClusterGrid = 12,
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
    pub sampler: SamplerHandle,
    pub channel: DebugChannel,
    pub backbuffer_name: String,

    // Resolved handles (updated in record)
    backbuffer: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive: ImageHandle,
    gbuffer_depth: ImageHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
}

impl DebugVizPass {
    pub fn new(sampler: SamplerHandle, channel: DebugChannel) -> Self {
        Self {
            backbuffer: ImageHandle::INVALID,
            gbuffer_albedo: ImageHandle::INVALID,
            gbuffer_normal: ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            gbuffer_emissive: ImageHandle::INVALID,
            gbuffer_depth: ImageHandle::INVALID,
            sampler,
            channel,
            backbuffer_name: "Backbuffer".to_string(),
            pipeline: None,
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

        let state = asset.state.as_ref().expect("DebugViz asset missing state");
        self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
            state,
            &asset.reflection_data,
            &asset.bytecode,
        ));
    }
}

impl RenderPass for DebugVizPass {
    fn init(&mut self, _backend: &mut dyn RenderBackend) {
        // Handled by init_from_baked
    }

    fn name(&self) -> &str {
        "DebugVizPass"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        // Resolve configuration from blackboard
        let channel_u32 = builder.consume::<u32>("DebugChannel");
        self.channel = match *channel_u32 {
            0 => DebugChannel::Albedo,
            1 => DebugChannel::Normal,
            2 => DebugChannel::Roughness,
            3 => DebugChannel::Metallic,
            4 => DebugChannel::Emissive,
            5 => DebugChannel::Depth,
            10 => DebugChannel::Lit,
            11 => DebugChannel::LightDensity,
            12 => DebugChannel::ClusterGrid,
            _ => DebugChannel::Lit,
        };

        // Resolve target handles by name
        self.gbuffer_albedo = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive = builder.resolve_image("GBuffer_Emissive");
        self.gbuffer_depth = builder.resolve_image("DepthBuffer");
        self.backbuffer = builder.resolve_image(&self.backbuffer_name);

        // Read GBuffer targets
        builder.read_image(self.gbuffer_albedo, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_emissive, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_depth, ResourceUsage::SHADER_READ);

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
                DescriptorWrite {
                    binding: 4,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.gbuffer_depth,
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
