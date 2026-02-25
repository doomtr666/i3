use i3_gfx::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AverageLuminancePushConstants {
    pub min_log_lum: f32,
    pub max_log_lum: f32,
    pub time_delta: f32,
    pub adaptation_rate: f32,
    pub pixel_count: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// Average luminance pass struct implementing the RenderPass trait.
pub struct AverageLuminancePass {
    pub pipeline: PipelineHandle,
    pub histogram_buffer: BufferHandle,
    pub exposure_buffer: BufferHandle,
    pub push_constants: AverageLuminancePushConstants,
}

impl RenderPass for AverageLuminancePass {
    fn name(&self) -> &str {
        "AverageLuminancePass"
    }

    fn domain(&self) -> PassDomain {
        PassDomain::Compute
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.bind_pipeline(self.pipeline);

        // Read histogram, read/write exposure
        builder.read_buffer(self.histogram_buffer, ResourceUsage::SHADER_READ);
        builder.write_buffer(self.exposure_buffer, ResourceUsage::SHADER_WRITE);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
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
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &self.push_constants);

        // Only 1 workgroup needed to process the 256-bin histogram
        ctx.dispatch(1, 1, 1);
    }
}
