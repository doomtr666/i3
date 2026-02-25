use i3_gfx::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ToneMapPushConstants {
    pub debug_mode: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// Tonemap pass struct implementing the RenderPass trait.
pub struct TonemapPass {
    pub pipeline: PipelineHandle,
    pub backbuffer: ImageHandle,
    pub hdr_target: ImageHandle,
    pub exposure_buffer: BufferHandle,
    pub sampler: SamplerHandle,
    pub push_constants: ToneMapPushConstants,
}

impl TonemapPass {
    pub fn new(
        pipeline: PipelineHandle,
        backbuffer: ImageHandle,
        hdr_target: ImageHandle,
        exposure_buffer: BufferHandle,
        sampler: SamplerHandle,
        push_constants: ToneMapPushConstants,
    ) -> Self {
        Self {
            pipeline,
            backbuffer,
            hdr_target,
            exposure_buffer,
            sampler,
            push_constants,
        }
    }
}

impl RenderPass for TonemapPass {
    fn name(&self) -> &str {
        "ToneMapPass"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.bind_pipeline(self.pipeline);

        // Read HDR target & ExposureBuffer
        builder.read_image(self.hdr_target, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.exposure_buffer, ResourceUsage::SHADER_READ);

        // Write to backbuffer
        builder.write_image(self.backbuffer, ResourceUsage::COLOR_ATTACHMENT);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.hdr_target,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(self.sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.exposure_buffer,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
            ],
        );
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        ctx.push_constant_data(
            ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
            0,
            &self.push_constants,
        );
        ctx.draw(3, 0); // Fullscreen triangle

        // This is the final pass targeting the backbuffer, so we must present it here.
        ctx.present(self.backbuffer);
    }
}
