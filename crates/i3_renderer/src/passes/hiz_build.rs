use crate::constants::div_ceil;
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
struct HiZSpdPc {
    mip0_size: [u32; 2],
}

/// Hi-Z pyramid builder using a single SPD dispatch.
///
/// GBufferFillPass writes mip 0 of HiZFinal as SV_Target4.
/// This pass reads mip 0 and generates mips 1..5 in one dispatch via LDS
/// reductions (16×16 threads, 32×32 tile per group, max reduction for Reverse-Z).
pub struct HiZBuildPass {
    output_hiz_name: &'static str,
    pipeline: Option<BackendPipeline>,
}

impl HiZBuildPass {
    pub fn new_final() -> Self {
        Self { output_hiz_name: "HiZFinal", pipeline: None }
    }
}

impl RenderPass for HiZBuildPass {
    fn name(&self) -> &str { "HiZBuild" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("hiz_spd")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let Some(pipeline) = self.pipeline else { return; };

        // HiZFinal is declared by GBufferPass (mip 0 filled via MRT).
        // We resolve it here and generate mips 1..5.
        let hiz_pyramid = builder.resolve_image(self.output_hiz_name);
        let hiz_desc    = builder.get_image_desc(hiz_pyramid);

        // Only run SPD if image has enough mip levels.
        if hiz_desc.mip_levels < 6 { return; }

        builder.add_owned_pass(HiZSpdSubPass {
            hiz_pyramid,
            pipeline,
            mip0_size: [hiz_desc.width, hiz_desc.height],
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────

struct HiZSpdSubPass {
    hiz_pyramid: ImageHandle,
    pipeline:    BackendPipeline,
    mip0_size:   [u32; 2],
}

impl RenderPass for HiZSpdSubPass {
    fn name(&self) -> &str { "HiZSpd" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // Mip 0 was written by GBufferFillPass as COLOR_ATTACHMENT.
        // We read it as SHADER_READ and write mips 1..5 as SHADER_WRITE.
        builder.read_image(self.hiz_pyramid,  ResourceUsage::SHADER_READ);
        builder.write_image(self.hiz_pyramid, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        ctx.bind_pipeline_raw(self.pipeline);

        let pc = HiZSpdPc { mip0_size: self.mip0_size };
        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let ds = ctx.create_descriptor_set(
            self.pipeline,
            0,
            &[
                DescriptorWrite::sampled_image_mip(
                    0, 0,
                    self.hiz_pyramid,
                    DescriptorImageLayout::General,
                    0, 1,
                ),
                DescriptorWrite::storage_image_mip(1, 0, self.hiz_pyramid, DescriptorImageLayout::General, 1),
                DescriptorWrite::storage_image_mip(2, 0, self.hiz_pyramid, DescriptorImageLayout::General, 2),
                DescriptorWrite::storage_image_mip(3, 0, self.hiz_pyramid, DescriptorImageLayout::General, 3),
                DescriptorWrite::storage_image_mip(4, 0, self.hiz_pyramid, DescriptorImageLayout::General, 4),
                DescriptorWrite::storage_image_mip(5, 0, self.hiz_pyramid, DescriptorImageLayout::General, 5),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        // Each group handles a 32×32 tile.
        ctx.dispatch(div_ceil(self.mip0_size[0], 32), div_ceil(self.mip0_size[1], 32), 1);
    }
}
