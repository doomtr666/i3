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

/// Cluster build pass struct implementing the RenderPass trait.
pub struct ClusterBuildPass {
    pub pipeline: PipelineHandle,
    pub cluster_aabbs: BufferHandle,
    pub push_constants: ClusterBuildPushConstants,
}

impl RenderPass for ClusterBuildPass {
    fn name(&self) -> &str {
        "ClusterBuild"
    }

    fn domain(&self) -> PassDomain {
        PassDomain::Compute
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.write_buffer(self.cluster_aabbs, ResourceUsage::SHADER_WRITE);
        builder.bind_pipeline(self.pipeline);
        builder.bind_descriptor_set(
            0,
            vec![i3_gfx::graph::backend::DescriptorWrite {
                binding: 0,
                array_element: 0,
                descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                    buffer: self.cluster_aabbs,
                    offset: 0,
                    range: 0,
                }),
                image_info: None,
            }],
        );
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &self.push_constants);

        ctx.dispatch(
            self.push_constants.grid_size[0],
            self.push_constants.grid_size[1],
            self.push_constants.grid_size[2],
        );
    }
}
