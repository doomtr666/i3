use i3_gfx::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ToneMapPushConstants {
    pub debug_mode: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

pub fn record_tonemap_pass(
    builder: &mut PassBuilder,
    pipeline: PipelineHandle,
    backbuffer: ImageHandle,
    hdr_target: ImageHandle,
    exposure_buffer: BufferHandle,
    sampler: SamplerHandle,
    push_constants: &ToneMapPushConstants,
) {
    let pc = *push_constants;
    builder.add_node("ToneMapPass", move |builder| {
        builder.bind_pipeline(pipeline);

        // Read HDR target & ExposureBuffer
        builder.read_image(hdr_target, ResourceUsage::SHADER_READ);
        builder.read_buffer(exposure_buffer, ResourceUsage::SHADER_READ);

        // Write to backbuffer
        builder.write_image(backbuffer, ResourceUsage::COLOR_ATTACHMENT);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: hdr_target,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(sampler),
                    }),
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
                    std::mem::size_of::<ToneMapPushConstants>(),
                )
            };
            ctx.push_constants(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                pc_bytes,
            );
            ctx.draw(3, 0); // Fullscreen triangle

            // This is the final pass targeting the backbuffer, so we must present it here.
            ctx.present(backbuffer);
        }
    });
}
