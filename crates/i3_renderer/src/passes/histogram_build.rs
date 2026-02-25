use i3_gfx::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HistogramPushConstants {
    pub min_log_lum: f32,
    pub max_log_lum: f32,
    pub time_delta: f32,
    pub pad: u32,
}

/// Histogram build pass struct implementing the RenderPass trait.
pub struct HistogramBuildPass {
    pub pipeline: PipelineHandle,
    pub hdr_image: ImageHandle,
    pub histogram_buffer: BufferHandle,
    pub exposure_buffer: BufferHandle,
    pub width: u32,
    pub height: u32,
    pub push_constants: HistogramPushConstants,
}

impl RenderPass for HistogramBuildPass {
    fn name(&self) -> &str {
        "HistogramBuildPass"
    }

    fn domain(&self) -> PassDomain {
        PassDomain::Compute
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.bind_pipeline(self.pipeline);

        // Read HDR image and exposure buffer
        builder.read_image(self.hdr_image, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.exposure_buffer, ResourceUsage::SHADER_READ);

        // Write to histogram buffer
        builder.write_buffer(self.histogram_buffer, ResourceUsage::SHADER_WRITE);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::Texture,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.hdr_image,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: None,
                    }),
                },
                DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.histogram_buffer,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 2,
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
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &self.push_constants);

        // Assuming GROUP_SIZE = 16
        let group_count_x = (self.width + 15) / 16;
        let group_count_y = (self.height + 15) / 16;
        ctx.dispatch(group_count_x, group_count_y, 1);
    }
}
