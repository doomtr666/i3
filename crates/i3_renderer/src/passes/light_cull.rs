use i3_gfx::prelude::*;
use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LightCullPushConstants {
    pub view_matrix: glm::Mat4, // 64 bytes
    pub grid_size: [u32; 3],    // 12 bytes
    pub light_count: u32,       // 4 bytes -> total 80 bytes
}

pub fn record_light_cull_pass(
    builder: &mut PassBuilder,
    pipeline: PipelineHandle,
    cluster_aabbs: BufferHandle,
    lights: BufferHandle,
    cluster_grid: BufferHandle,
    cluster_light_indices: BufferHandle,
    push_constants: &LightCullPushConstants,
) {
    let pc = *push_constants;
    builder.add_node("LightCull", move |pass_node| {
        pass_node.read_buffer(cluster_aabbs, ResourceUsage::SHADER_READ);
        pass_node.read_buffer(lights, ResourceUsage::SHADER_READ);
        pass_node.write_buffer(cluster_grid, ResourceUsage::SHADER_WRITE);
        pass_node.write_buffer(cluster_light_indices, ResourceUsage::SHADER_WRITE);

        pass_node.bind_pipeline(pipeline);
        pass_node.bind_descriptor_set(
            0,
            vec![
                i3_gfx::graph::backend::DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                    buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                        buffer: cluster_aabbs,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                i3_gfx::graph::backend::DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                    buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                        buffer: lights,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                i3_gfx::graph::backend::DescriptorWrite {
                    binding: 2,
                    array_element: 0,
                    descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                    buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                        buffer: cluster_grid,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                i3_gfx::graph::backend::DescriptorWrite {
                    binding: 3,
                    array_element: 0,
                    descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                    buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                        buffer: cluster_light_indices,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
            ],
        );

        move |ctx| {
            let pc_bytes = unsafe {
                std::slice::from_raw_parts(
                    &pc as *const _ as *const u8,
                    std::mem::size_of::<LightCullPushConstants>(),
                )
            };
            ctx.push_constants(
                i3_gfx::graph::pipeline::ShaderStageFlags::Compute,
                0,
                pc_bytes,
            );

            ctx.dispatch(pc.grid_size[0], pc.grid_size[1], pc.grid_size[2]);
        }
    });
}
