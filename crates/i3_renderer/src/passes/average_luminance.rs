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

pub fn record_average_luminance_pass(
    builder: &mut PassBuilder,
    pipeline: PipelineHandle,
    histogram_buffer: BufferHandle,
    exposure_buffer: BufferHandle,
    push_constants: &AverageLuminancePushConstants,
) {
    let pc = *push_constants;

    builder.add_node("AverageLuminancePass", move |builder| {
        builder.bind_pipeline(pipeline);

        // Read histogram, read/write exposure
        builder.read_buffer(histogram_buffer, ResourceUsage::SHADER_READ);
        builder.write_buffer(exposure_buffer, ResourceUsage::SHADER_WRITE);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: histogram_buffer,
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
                        buffer: exposure_buffer,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
            ],
        );

        move |ctx: &mut dyn PassContext| {
            let pc_bytes = unsafe {
                std::slice::from_raw_parts(
                    &pc as *const _ as *const u8,
                    std::mem::size_of::<AverageLuminancePushConstants>(),
                )
            };
            ctx.push_constants(ShaderStageFlags::Compute, 0, pc_bytes);

            // Only 1 workgroup needed to process the 256-bin histogram
            ctx.dispatch(1, 1, 1);
        }
    });
}
