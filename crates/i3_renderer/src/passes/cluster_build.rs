use std::sync::Arc;
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

pub struct ClusterBuildPass {
    pub push_constants: ClusterBuildPushConstants,

    // Resolved handles (updated in declare)
    cluster_aabbs: BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
}

impl ClusterBuildPass {
    pub fn new() -> Self {
        Self {
            cluster_aabbs: BufferHandle::INVALID,
            push_constants: ClusterBuildPushConstants {
                inv_projection: nalgebra_glm::identity(),
                grid_size: [0, 0, 0],
                near_plane: 0.0,
                far_plane: 0.0,
                screen_dimensions: [0.0, 0.0],
                pad: 0,
            },
            pipeline: None,
        }
    }

}

impl RenderPass for ClusterBuildPass {
    fn name(&self) -> &str {
        "ClusterBuild"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("cluster_build").wait_loaded() {
            self.pipeline = Some(backend.create_compute_pipeline_from_baked(
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        if builder.is_setup() {
            return;
        }
        self.cluster_aabbs = builder.resolve_buffer("ClusterAABBs");

        // Consume CommonData to compute push constants
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        let grid_x = (common.screen_width + 63) / 64;
        let grid_y = (common.screen_height + 63) / 64;
        let grid_z: u32 = 16;

        self.push_constants = ClusterBuildPushConstants {
            inv_projection: common.inv_projection,
            grid_size: [grid_x, grid_y, grid_z],
            near_plane: common.near_plane,
            far_plane: common.far_plane,
            screen_dimensions: [common.screen_width as f32, common.screen_height as f32],
            pad: 0,
        };

        builder.write_buffer(self.cluster_aabbs, ResourceUsage::SHADER_WRITE);
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
        if let Some(pipeline) = self.pipeline {
            ctx.bind_pipeline_raw(pipeline);
            ctx.push_constant_data(ShaderStageFlags::Compute, 0, &self.push_constants);
            ctx.dispatch(
                self.push_constants.grid_size[0],
                self.push_constants.grid_size[1],
                self.push_constants.grid_size[2],
            );
        } else {
            tracing::error!("ClusterBuildPass::execute: pipeline not initialized!");
        }
    }
}
