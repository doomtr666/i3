use i3_gfx::prelude::*;
use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ClusterBuildPushConstants {
    pub inv_projection: glm::Mat4,
    pub grid_size: [u32; 3],
    pub near_plane: f32,
    pub far_plane: f32,
    pub screen_dimensions: [f32; 2],
    pub pad: u32,
}

pub fn record_cluster_build_pass(
    builder: &mut PassBuilder,
    pipeline: PipelineHandle,
    cluster_aabbs: BufferHandle,
    push_constants: &ClusterBuildPushConstants,
) {
    let pc = *push_constants;
    builder.add_node("ClusterBuild", move |pass_node| {
        pass_node.write_buffer(cluster_aabbs, ResourceUsage::SHADER_WRITE);
        pass_node.bind_pipeline(pipeline);
        pass_node.bind_descriptor_set(
            0,
            vec![i3_gfx::graph::backend::DescriptorWrite {
                binding: 0,
                array_element: 0,
                descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                    buffer: cluster_aabbs,
                    offset: 0,
                    range: 0,
                }),
                image_info: None,
            }],
        );

        move |ctx| {
            let pc_bytes = unsafe {
                std::slice::from_raw_parts(
                    &pc as *const _ as *const u8,
                    std::mem::size_of::<ClusterBuildPushConstants>(),
                )
            };
            ctx.push_constants(
                i3_gfx::graph::pipeline::ShaderStageFlags::Compute,
                0,
                pc_bytes,
            );

            // Our shader uses [numthreads(1, 1, 1)] for now. Wait, I wrote [numthreads(8, 8, 1)]? No, [numthreads(1, 1, 1)]
            // I should just dispatch one thread per cluster.
            ctx.dispatch(pc.grid_size[0], pc.grid_size[1], pc.grid_size[2]);
        }
    });
}
