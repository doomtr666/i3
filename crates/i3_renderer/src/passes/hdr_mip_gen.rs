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

/// Number of mip levels in HDR_Mips (mip 0 = full-res copy, mips 1..5 = downsampled).
pub const HDR_MIP_COUNT: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HdrSpdPc {
    src_size: [u32; 2],
}

/// Generates the HDR_Mips resource from HDR_Target in a single SPD dispatch.
///
/// HDR_Mips is used by SssrSamplePass to approximate blurry reflections:
/// a mirror ray hit is sampled at mip = roughness * max_mip, matching the
/// IBL pre-filter philosophy.
///
/// Insert in the render graph AFTER deferred_resolve (HDR_Target fully written)
/// and BEFORE sssr_sample_pass.
pub struct HdrMipGenPass {
    pipeline:   Option<BackendPipeline>,
    hdr_target: ImageHandle,
    hdr_mips:   ImageHandle,
}

impl HdrMipGenPass {
    pub fn new() -> Self {
        Self {
            pipeline:   None,
            hdr_target: ImageHandle::INVALID,
            hdr_mips:   ImageHandle::INVALID,
        }
    }
}

impl RenderPass for HdrMipGenPass {
    fn name(&self) -> &str { "HdrMipGen" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("hdr_spd")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let Some(pipeline) = self.pipeline else { return; };

        self.hdr_target = builder.resolve_image("HDR_Target");
        let hdr_desc    = builder.get_image_desc(self.hdr_target);
        let (w, h)      = (hdr_desc.width, hdr_desc.height);

        self.hdr_mips = builder.declare_image_output(
            "HDR_Mips",
            ImageDesc {
                width:        w,
                height:       h,
                depth:        1,
                format:       Format::R16G16B16A16_SFLOAT,
                mip_levels:   HDR_MIP_COUNT,
                array_layers: 1,
                usage:        ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
                view_type:    ImageViewType::Type2D,
                swizzle:      ComponentMapping::default(),
                clear_value:  None,
            },
        );

        builder.add_owned_pass(HdrSpdSubPass {
            hdr_target: self.hdr_target,
            hdr_mips:   self.hdr_mips,
            pipeline,
            src_size:   [w, h],
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────

struct HdrSpdSubPass {
    hdr_target: ImageHandle,
    hdr_mips:   ImageHandle,
    pipeline:   BackendPipeline,
    src_size:   [u32; 2],
}

impl RenderPass for HdrSpdSubPass {
    fn name(&self) -> &str { "HdrSpd" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.read_image(self.hdr_target, ResourceUsage::SHADER_READ);
        builder.write_image(self.hdr_mips,  ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        ctx.bind_pipeline_raw(self.pipeline);

        let pc = HdrSpdPc { src_size: self.src_size };
        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let ds = ctx.create_descriptor_set(
            self.pipeline,
            0,
            &[
                DescriptorWrite::sampled_image(
                    0, 0,
                    self.hdr_target,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                // Bindings 1..6 = HDR_Mips mips 0..5 (match shader [[vk::binding(1..6, 0)]])
                DescriptorWrite::storage_image_mip(1, 0, self.hdr_mips, DescriptorImageLayout::General, 0),
                DescriptorWrite::storage_image_mip(2, 0, self.hdr_mips, DescriptorImageLayout::General, 1),
                DescriptorWrite::storage_image_mip(3, 0, self.hdr_mips, DescriptorImageLayout::General, 2),
                DescriptorWrite::storage_image_mip(4, 0, self.hdr_mips, DescriptorImageLayout::General, 3),
                DescriptorWrite::storage_image_mip(5, 0, self.hdr_mips, DescriptorImageLayout::General, 4),
                DescriptorWrite::storage_image_mip(6, 0, self.hdr_mips, DescriptorImageLayout::General, 5),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        // Each group handles a 32×32 tile.
        ctx.dispatch(div_ceil(self.src_size[0], 32), div_ceil(self.src_size[1], 32), 1);
    }
}
