use i3_gfx::prelude::*;
use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LightCullPushConstants {
    pub view_matrix: glm::Mat4, // 64 bytes
    pub grid_size: [u32; 3],    // 12 bytes
    pub light_count: u32,       // 4 bytes -> total 80 bytes
}

/// Light cull pass struct implementing the RenderPass trait.
pub struct LightCullPass {
    pub pipeline: PipelineHandle,
    pub cluster_aabbs: BufferHandle,
    pub lights: BufferHandle,
    pub cluster_grid: BufferHandle,
    pub cluster_light_indices: BufferHandle,
    pub push_constants: LightCullPushConstants,
}

impl RenderPass for LightCullPass {
    fn name(&self) -> &str {
        "LightCull"
    }

    fn domain(&self) -> PassDomain {
        PassDomain::Compute
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.read_buffer(self.cluster_aabbs, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.lights, ResourceUsage::SHADER_READ);
        builder.write_buffer(self.cluster_grid, ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.cluster_light_indices, ResourceUsage::SHADER_WRITE);

        builder.bind_pipeline(self.pipeline);
        builder.bind_descriptor_set(
            0,
            vec![
                i3_gfx::graph::backend::DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                    buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                        buffer: self.cluster_aabbs,
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
                        buffer: self.lights,
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
                        buffer: self.cluster_grid,
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
                        buffer: self.cluster_light_indices,
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
            i3_gfx::graph::pipeline::ShaderStageFlags::Compute,
            0,
            &self.push_constants,
        );

        ctx.dispatch(
            self.push_constants.grid_size[0],
            self.push_constants.grid_size[1],
            self.push_constants.grid_size[2],
        );
    }
}
