use crate::constants::{div_ceil, CLUSTER_GRID_Z, CLUSTER_TILE_SIZE};
use i3_gfx::prelude::*;
use std::sync::Arc;

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
    // Resolved handles (updated in declare)
    cluster_aabbs: BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
}

impl ClusterBuildPass {
    pub fn new() -> Self {
        Self {
            cluster_aabbs: BufferHandle::INVALID,
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
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("cluster_build")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.cluster_aabbs = builder.resolve_buffer("ClusterAABBs");

        builder.write_buffer(self.cluster_aabbs, ResourceUsage::SHADER_WRITE);
        builder.descriptor_set(0, |d| {
            d.storage_buffer(self.cluster_aabbs);
        });
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("ClusterBuildPass::execute: pipeline not initialized!");
            return;
        };
        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let grid_x = div_ceil(common.screen_width, CLUSTER_TILE_SIZE);
        let grid_y = div_ceil(common.screen_height, CLUSTER_TILE_SIZE);
        let grid_z: u32 = CLUSTER_GRID_Z;
        let push_constants = ClusterBuildPushConstants {
            inv_projection: common.inv_projection,
            grid_size: [grid_x, grid_y, grid_z],
            near_plane: common.near_plane,
            far_plane: common.far_plane,
            screen_dimensions: [common.screen_width as f32, common.screen_height as f32],
            pad: 0,
        };
        ctx.bind_pipeline_raw(pipeline);
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &push_constants);
        ctx.dispatch(grid_x, grid_y, grid_z);
    }
}
