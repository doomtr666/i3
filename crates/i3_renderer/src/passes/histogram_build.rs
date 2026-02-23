use i3_gfx::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HistogramPushConstants {
    pub min_log_lum: f32,
    pub max_log_lum: f32,
    pub time_delta: f32,
    pub pad: u32,
}

pub fn record_histogram_build_pass(
    builder: &mut PassBuilder,
    pipeline: PipelineHandle,
    hdr_image: ImageHandle,
    histogram_buffer: BufferHandle,
    exposure_buffer: BufferHandle,
    width: u32,
    height: u32,
    push_constants: &HistogramPushConstants,
) {
    let pc = *push_constants;

    builder.add_node("HistogramBuildPass", move |builder| {
        builder.bind_pipeline(pipeline);

        // Read HDR image and exposure buffer
        builder.read_image(hdr_image, ResourceUsage::SHADER_READ);
        builder.read_buffer(exposure_buffer, ResourceUsage::SHADER_READ);

        // Write to histogram buffer
        builder.write_buffer(histogram_buffer, ResourceUsage::SHADER_WRITE);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::Texture,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: hdr_image,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: None,
                    }),
                },
                DescriptorWrite {
                    binding: 1,
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
                    binding: 2,
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
                    std::mem::size_of::<HistogramPushConstants>(),
                )
            };
            ctx.push_constants(ShaderStageFlags::Compute, 0, pc_bytes);
            ctx.clear_buffer(histogram_buffer, 0);

            // Assuming GROUP_SIZE = 16
            let group_count_x = (width + 15) / 16;
            let group_count_y = (height + 15) / 16;
            ctx.dispatch(group_count_x, group_count_y, 1);
        }
    });
}
