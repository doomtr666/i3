use i3_gfx::prelude::*;
use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeferredResolvePushConstants {
    pub inv_view_proj: glm::Mat4,
    pub inv_projection: glm::Mat4,
    pub camera_pos: glm::Vec3,
    pub near_plane: f32,
    pub grid_size: [u32; 3],
    pub far_plane: f32,
    pub screen_dimensions: [f32; 2],
    pub debug_mode: u32,
    pub _pad: u32,
}

pub fn record_deferred_resolve_pass(
    builder: &mut PassBuilder,
    pipeline: PipelineHandle,
    backbuffer: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive: ImageHandle,
    depth_buffer: ImageHandle,
    lights: BufferHandle,
    cluster_grid: BufferHandle,
    cluster_light_indices: BufferHandle,
    sampler: SamplerHandle,
    push_constants: &DeferredResolvePushConstants,
) {
    let pc = *push_constants;
    builder.add_node("DeferredResolvePass", move |sub| {
        sub.bind_pipeline(pipeline);

        // Read GBuffers and buffers
        sub.read_image(gbuffer_albedo, ResourceUsage::SHADER_READ);
        sub.read_image(gbuffer_normal, ResourceUsage::SHADER_READ);
        sub.read_image(gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        sub.read_image(gbuffer_emissive, ResourceUsage::SHADER_READ);
        sub.read_image(depth_buffer, ResourceUsage::SHADER_READ);

        sub.read_buffer(lights, ResourceUsage::SHADER_READ);
        sub.read_buffer(cluster_grid, ResourceUsage::SHADER_READ);
        sub.read_buffer(cluster_light_indices, ResourceUsage::SHADER_READ);

        // Write to backbuffer
        sub.write_image(backbuffer, ResourceUsage::COLOR_ATTACHMENT);

        sub.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: gbuffer_albedo,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: gbuffer_normal,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 2,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: gbuffer_roughmetal,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 3,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: gbuffer_emissive,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 4,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: depth_buffer,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 5,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: lights,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 6,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: cluster_grid,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 7,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: cluster_light_indices,
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
                    std::mem::size_of::<DeferredResolvePushConstants>(),
                )
            };
            ctx.push_constants(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                pc_bytes,
            );
            ctx.draw(3, 0); // Fullscreen triangle
            ctx.present(backbuffer); // Wait, backbuffer usually presented externally? Well, debug_viz does this
        }
    });
}
