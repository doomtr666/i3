use bytemuck::{Pod, Zeroable};
use i3_gfx::graph::backend::{
    BackendPipeline, DescriptorImageLayout, DescriptorSetHandle, DescriptorWrite, PassContext,
    RenderBackend,
};
use i3_gfx::graph::compiler::FrameBlackboard;
use i3_gfx::graph::pass::{PassBuilder, RenderPass};
use i3_gfx::graph::pipeline::ShaderStageFlags;
use i3_gfx::graph::types::*;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HdrMipsPushConstants {
    src_size: [u32; 2],
    src_mip:  u32,
    _pad:     u32,
}

/// Generates the full mip chain for `HDR_Target` using a 2×2 box average.
///
/// SSR uses these mips to sample blurred scene colour on glossy surfaces:
/// higher roughness → higher mip → softer reflection.
///
/// One `HdrMipsReduceSubPass` is registered per mip transition (mip N → mip N+1),
/// same pattern as `HiZBuildPass`. The passes are owned by `HdrMipsPass::declare()`.
pub struct HdrMipsPass {
    reduce_pipeline: Option<BackendPipeline>,
}

impl HdrMipsPass {
    pub fn new() -> Self {
        Self { reduce_pipeline: None }
    }
}

impl RenderPass for HdrMipsPass {
    fn name(&self) -> &str { "HdrMipsPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("hdr_mips_reduce")
            .wait_loaded()
        {
            self.reduce_pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let hdr = builder.resolve_image("HDR_Target");
        let desc = builder.get_image_desc(hdr);

        let Some(pipeline) = self.reduce_pipeline else { return; };

        // Register one sub-pass per mip transition: mip i → mip i+1.
        for src_mip in 0..(desc.mip_levels.saturating_sub(1)) {
            let src_w = (desc.width  >> src_mip).max(1);
            let src_h = (desc.height >> src_mip).max(1);

            builder.add_owned_pass(HdrMipsReduceSubPass {
                hdr_target: hdr,
                pipeline,
                src_mip,
                src_size: [src_w, src_h],
            });
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────

struct HdrMipsReduceSubPass {
    hdr_target: ImageHandle,
    pipeline:   BackendPipeline,
    src_mip:    u32,
    src_size:   [u32; 2],
}

impl RenderPass for HdrMipsReduceSubPass {
    fn name(&self) -> &str { "HdrMipsReduce" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.read_image(self.hdr_target,  ResourceUsage::SHADER_READ);
        builder.write_image(self.hdr_target, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        ctx.bind_pipeline_raw(self.pipeline);

        let pc = HdrMipsPushConstants {
            src_size: self.src_size,
            src_mip:  self.src_mip,
            _pad:     0,
        };
        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let ds = ctx.create_descriptor_set(
            self.pipeline,
            0,
            &[
                DescriptorWrite::sampled_image_mip(
                    0, 0,
                    self.hdr_target,
                    DescriptorImageLayout::General,
                    self.src_mip,
                    1,
                ),
                DescriptorWrite::storage_image_mip(
                    1, 0,
                    self.hdr_target,
                    DescriptorImageLayout::General,
                    self.src_mip + 1,
                ),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        let dst_w = (self.src_size[0] / 2).max(1);
        let dst_h = (self.src_size[1] / 2).max(1);
        ctx.dispatch((dst_w + 7) / 8, (dst_h + 7) / 8, 1);
    }
}
