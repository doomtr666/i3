use i3_gfx::graph::pass::{PassBuilder, RenderPass};
use i3_gfx::graph::backend::{PassContext, BackendPipeline, DescriptorImageLayout, DescriptorWrite, RenderBackend};
use i3_gfx::graph::types::*;
use i3_gfx::graph::pipeline::ShaderStageFlags;
use i3_gfx::graph::compiler::FrameBlackboard;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HiZPushConstants {
    src_size: [u32; 2],
    hiz_size: [u32; 2],
    src_mip: u32,
    _pad: u32,
}

/// Root pass for Hi-Z Pyramid construction.
/// This pass acts as a coordinator, adding sub-passes for each pyramid level.
pub struct HiZBuildPass {
    blit_pipeline: Option<BackendPipeline>,
    reduce_pipeline: Option<BackendPipeline>,
}

impl HiZBuildPass {
    pub fn new() -> Self {
        Self {
            blit_pipeline: None,
            reduce_pipeline: None,
        }
    }
}

impl RenderPass for HiZBuildPass {
    fn name(&self) -> &str { "HiZBuild" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        
        // Load Blit Pipeline
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("hiz_build_blit").wait_loaded() {
            self.blit_pipeline = Some(backend.create_compute_pipeline_from_baked(
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }

        // Load Reduce Pipeline
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("hiz_build_reduce").wait_loaded() {
            self.reduce_pipeline = Some(backend.create_compute_pipeline_from_baked(
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let depth_buffer = builder.resolve_image("DepthBuffer");
        let hiz_pyramid = builder.resolve_image("HiZPyramid");

        let hiz_desc = builder.get_image_desc(hiz_pyramid);
        let depth_desc = builder.get_image_desc(depth_buffer);
        let mips = hiz_desc.mip_levels;

        if let Some(blit) = self.blit_pipeline {
            builder.add_owned_pass(HiZBlitSubPass {
                depth_buffer,
                hiz_pyramid,
                pipeline: blit,
                src_size: [depth_desc.width, depth_desc.height],
                dst_size: [hiz_desc.width, hiz_desc.height],
            });
        }

        if let Some(reduce) = self.reduce_pipeline {
            for mip in 0..(mips - 1) {
                let src_w = (hiz_desc.width >> mip).max(1);
                let src_h = (hiz_desc.height >> mip).max(1);
                let dst_w = (src_w >> 1).max(1);
                let dst_h = (src_h >> 1).max(1);

                builder.add_owned_pass(HiZReduceSubPass {
                    hiz_pyramid,
                    pipeline: reduce,
                    src_mip: mip,
                    hiz_size: [hiz_desc.width, hiz_desc.height],
                    dst_size: [dst_w, dst_h],
                });
            }
        }
    }

    fn execute(&self, _ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {}
}

struct HiZBlitSubPass {
    depth_buffer: ImageHandle,
    hiz_pyramid: ImageHandle,
    pipeline: BackendPipeline,
    src_size: [u32; 2],
    dst_size: [u32; 2],
}

impl RenderPass for HiZBlitSubPass {
    fn name(&self) -> &str { "HiZBlit" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.read_image(self.depth_buffer, ResourceUsage::SHADER_READ);
        builder.write_image(self.hiz_pyramid, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        ctx.bind_pipeline_raw(self.pipeline);
        
        let pc = HiZPushConstants {
            src_size: self.src_size,
            hiz_size: self.dst_size,
            src_mip: 0,
            _pad: 0,
        };
        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        let descriptor_set = ctx.create_descriptor_set(self.pipeline, 0, &[
            DescriptorWrite::sampled_image(0, 0, self.depth_buffer, DescriptorImageLayout::ShaderReadOnlyOptimal),
            DescriptorWrite::storage_image_mip(1, 0, self.hiz_pyramid, DescriptorImageLayout::General, 0),
        ]);
        ctx.bind_descriptor_set(0, descriptor_set);

        let groups_x = (self.dst_size[0] + 7) / 8;
        let groups_y = (self.dst_size[1] + 7) / 8;
        ctx.dispatch(groups_x, groups_y, 1);
    }
}

struct HiZReduceSubPass {
    hiz_pyramid: ImageHandle,
    pipeline: BackendPipeline,
    src_mip: u32,
    hiz_size: [u32; 2],
    dst_size: [u32; 2],
}

impl RenderPass for HiZReduceSubPass {
    fn name(&self) -> &str { "HiZReduce" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.read_image(self.hiz_pyramid, ResourceUsage::SHADER_READ);
        builder.write_image(self.hiz_pyramid, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        ctx.bind_pipeline_raw(self.pipeline);

        let pc = HiZPushConstants {
            src_size: [0, 0],
            hiz_size: self.hiz_size,
            src_mip: self.src_mip,
            _pad: 0,
        };
        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        let descriptor_set = ctx.create_descriptor_set(self.pipeline, 0, &[
            DescriptorWrite::sampled_image_mip(0, 0, self.hiz_pyramid, DescriptorImageLayout::General, self.src_mip, 1),
            DescriptorWrite::storage_image_mip(1, 0, self.hiz_pyramid, DescriptorImageLayout::General, self.src_mip + 1),
        ]);
        ctx.bind_descriptor_set(0, descriptor_set);

        let groups_x = (self.dst_size[0] + 7) / 8;
        let groups_y = (self.dst_size[1] + 7) / 8;
        ctx.dispatch(groups_x, groups_y, 1);
    }
}
