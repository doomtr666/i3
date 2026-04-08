use i3_gfx::prelude::*;
use std::sync::Arc;

use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LightCullPushConstants {
    pub view_matrix: glm::Mat4, // 64 bytes
    pub grid_size: [u32; 3],    // 12 bytes
    pub light_count: u32,       // 4 bytes -> total 80 bytes
}

pub struct LightCullPass {
    pub light_buffer_name: String,

    // Resolved handles (updated in declare)
    cluster_aabbs: BufferHandle,
    lights: BufferHandle,
    cluster_grid: BufferHandle,
    cluster_light_indices: BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
}

impl LightCullPass {
    pub fn new() -> Self {
        Self {
            cluster_aabbs: BufferHandle::INVALID,
            lights: BufferHandle::INVALID,
            cluster_grid: BufferHandle::INVALID,
            cluster_light_indices: BufferHandle::INVALID,
            light_buffer_name: "LightBuffer".to_string(),
            pipeline: None,
        }
    }
}

impl RenderPass for LightCullPass {
    fn name(&self) -> &str {
        "LightCull"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("light_cull")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.cluster_aabbs = builder.resolve_buffer("ClusterAABBs");
        self.lights = builder.resolve_buffer(&self.light_buffer_name);
        self.cluster_grid = builder.resolve_buffer("ClusterGrid");
        self.cluster_light_indices = builder.resolve_buffer("ClusterLightIndices");

        builder.read_buffer(self.cluster_aabbs, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.lights, ResourceUsage::SHADER_READ);
        builder.write_buffer(self.cluster_grid, ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.cluster_light_indices, ResourceUsage::SHADER_WRITE);

        builder.descriptor_set(0, |d| {
            d.storage_buffer(self.cluster_aabbs)
                .storage_buffer(self.lights)
                .storage_buffer(self.cluster_grid)
                .storage_buffer(self.cluster_light_indices);
        });
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("LightCullPass::execute: pipeline not initialized!");
            return;
        };
        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let grid_x = (common.screen_width + crate::constants::CLUSTER_TILE_SIZE - 1)
            / crate::constants::CLUSTER_TILE_SIZE;
        let grid_y = (common.screen_height + crate::constants::CLUSTER_TILE_SIZE - 1)
            / crate::constants::CLUSTER_TILE_SIZE;
        let grid_z: u32 = crate::constants::CLUSTER_GRID_Z;
        let push_constants = LightCullPushConstants {
            view_matrix: common.view,
            grid_size: [grid_x, grid_y, grid_z],
            light_count: common.light_count,
        };
        ctx.bind_pipeline_raw(pipeline);
        ctx.push_constant_data(
            i3_gfx::graph::pipeline::ShaderStageFlags::Compute,
            0,
            &push_constants,
        );
        ctx.dispatch(grid_x, grid_y, grid_z);
    }
}
