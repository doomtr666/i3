use i3_gfx::prelude::*;

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
    pub pipeline: PipelineHandle,
    pub backbuffer: ImageHandle,
    pub gbuffer_albedo: ImageHandle,
    pub gbuffer_normal: ImageHandle,
    pub gbuffer_roughmetal: ImageHandle,
    pub gbuffer_emissive: ImageHandle,
    pub sampler: SamplerHandle,
    pub channel: DebugChannel,
}

impl RenderPass for DebugVizPass {
    fn name(&self) -> &str {
        "DebugVizPass"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.bind_pipeline(self.pipeline);

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
