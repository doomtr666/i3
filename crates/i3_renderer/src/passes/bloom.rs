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

// ─── Constants ───────────────────────────────────────────────────────────────

/// Number of mip levels in Bloom_Buffer (half-res base, down to 1/32).
const BLOOM_MIP_COUNT: u32 = 5;

// ─── Push constants ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomPrefilterPc {
    src_size:  [u32; 2],
    threshold: f32,
    knee:      f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomMipPc {
    src_size: [u32; 2],
    src_mip:  u32,
    _pad:     u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomCompositePc {
    intensity: f32,
    _pad:      [f32; 3],
}

// ─── BloomPass (parent) ───────────────────────────────────────────────────────

/// Bloom post-FX pass.
///
/// Bright-pass prefilter (half-res) → 13-tap downsample chain →
/// tent upsample chain (accumulative) → additive composite onto HDR_Target.
///
/// Declares `Bloom_Buffer` (R16G16B16A16_SFLOAT, BLOOM_MIP_COUNT mip levels)
/// as a transient image owned by the frame graph.
///
/// Sub-passes injected via `add_owned_pass` (pattern: HdrMipsPass):
///   BloomPrefilterSubPass
///   BloomDownSubPass × (BLOOM_MIP_COUNT − 1)
///   BloomUpSubPass   × (BLOOM_MIP_COUNT − 1)
///   BloomCompositeSubPass
pub struct BloomPass {
    pub enabled:   bool,
    pub threshold: f32,
    pub knee:      f32,
    pub intensity: f32,

    prefilter_pipeline:  Option<BackendPipeline>,
    downsample_pipeline: Option<BackendPipeline>,
    upsample_pipeline:   Option<BackendPipeline>,
    composite_pipeline:  Option<BackendPipeline>,
}

impl BloomPass {
    pub fn new() -> Self {
        Self {
            enabled:   true,
            threshold: 1.0,
            knee:      0.5,
            intensity: 0.1,

            prefilter_pipeline:  None,
            downsample_pipeline: None,
            upsample_pipeline:   None,
            composite_pipeline:  None,
        }
    }
}

impl RenderPass for BloomPass {
    fn name(&self) -> &str { "BloomPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");

        macro_rules! load_pipeline {
            ($name:expr) => {
                if let Ok(asset) = loader
                    .load::<i3_io::pipeline_asset::PipelineAsset>($name)
                    .wait_loaded()
                {
                    Some(backend.create_compute_pipeline_from_baked(
                        &asset.reflection_data,
                        &asset.bytecode,
                    ))
                } else {
                    None
                }
            };
        }

        self.prefilter_pipeline  = load_pipeline!("bloom_prefilter");
        self.downsample_pipeline = load_pipeline!("bloom_downsample");
        self.upsample_pipeline   = load_pipeline!("bloom_upsample");
        self.composite_pipeline  = load_pipeline!("bloom_composite");
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // All four pipelines must be ready before we inject sub-passes.
        let (Some(pre_pl), Some(down_pl), Some(up_pl), Some(comp_pl)) = (
            self.prefilter_pipeline,
            self.downsample_pipeline,
            self.upsample_pipeline,
            self.composite_pipeline,
        ) else {
            return;
        };

        if !self.enabled { return; }

        let hdr = builder.resolve_image("HDR_Target");
        let hdr_desc = builder.get_image_desc(hdr);

        // Bloom_Buffer: half-resolution, BLOOM_MIP_COUNT mip levels.
        let bloom_w = (hdr_desc.width  / 2).max(1);
        let bloom_h = (hdr_desc.height / 2).max(1);

        let bloom_buf = builder.declare_image_output(
            "Bloom_Buffer",
            ImageDesc {
                width:        bloom_w,
                height:       bloom_h,
                depth:        1,
                format:       Format::R16G16B16A16_SFLOAT,
                mip_levels:   BLOOM_MIP_COUNT,
                array_layers: 1,
                usage:        ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
                view_type:    ImageViewType::Type2D,
                swizzle:      ComponentMapping::default(),
                clear_value:  None,
            },
        );

        // Capture parameters for sub-passes.
        let threshold = self.threshold;
        let knee      = self.knee;
        let intensity = self.intensity;

        // ── Prefilter ─────────────────────────────────────────────────────────
        builder.add_owned_pass(BloomPrefilterSubPass {
            hdr_target: hdr,
            bloom_buf,
            pipeline:  pre_pl,
            hdr_size:  [hdr_desc.width, hdr_desc.height],
            threshold,
            knee,
        });

        // ── Downsample chain ──────────────────────────────────────────────────
        for src_mip in 0..(BLOOM_MIP_COUNT - 1) {
            let src_w = (bloom_w >> src_mip).max(1);
            let src_h = (bloom_h >> src_mip).max(1);
            builder.add_owned_pass(BloomDownSubPass {
                bloom_buf,
                pipeline: down_pl,
                src_mip,
                src_size: [src_w, src_h],
            });
        }

        // ── Upsample chain (bottom → top) ─────────────────────────────────────
        // src_mip goes from (N-1) down to 1; dst_mip = src_mip - 1.
        for src_mip in (1..BLOOM_MIP_COUNT).rev() {
            let src_w = (bloom_w >> src_mip).max(1);
            let src_h = (bloom_h >> src_mip).max(1);
            builder.add_owned_pass(BloomUpSubPass {
                bloom_buf,
                pipeline: up_pl,
                src_mip,
                src_size: [src_w, src_h],
            });
        }

        // ── Composite ─────────────────────────────────────────────────────────
        builder.add_owned_pass(BloomCompositeSubPass {
            hdr_target: hdr,
            bloom_buf,
            pipeline: comp_pl,
            intensity,
        });
    }
}

// ─── Sub-passes ───────────────────────────────────────────────────────────────

struct BloomPrefilterSubPass {
    hdr_target: ImageHandle,
    bloom_buf:  ImageHandle,
    pipeline:   BackendPipeline,
    hdr_size:   [u32; 2],
    threshold:  f32,
    knee:       f32,
}

impl RenderPass for BloomPrefilterSubPass {
    fn name(&self) -> &str { "BloomPrefilter" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.read_image(self.hdr_target, ResourceUsage::SHADER_READ);
        builder.write_image(self.bloom_buf, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        ctx.bind_pipeline_raw(self.pipeline);

        let pc = BloomPrefilterPc {
            src_size:  self.hdr_size,
            threshold: self.threshold,
            knee:      self.knee,
        };
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
                DescriptorWrite::storage_image_mip(
                    1, 0,
                    self.bloom_buf,
                    DescriptorImageLayout::General,
                    0,
                ),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        let dst_w = (self.hdr_size[0] / 2).max(1);
        let dst_h = (self.hdr_size[1] / 2).max(1);
        ctx.dispatch((dst_w + 7) / 8, (dst_h + 7) / 8, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────

struct BloomDownSubPass {
    bloom_buf: ImageHandle,
    pipeline:  BackendPipeline,
    src_mip:   u32,
    src_size:  [u32; 2],
}

impl RenderPass for BloomDownSubPass {
    fn name(&self) -> &str { "BloomDown" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.read_image(self.bloom_buf,  ResourceUsage::SHADER_READ);
        builder.write_image(self.bloom_buf, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        ctx.bind_pipeline_raw(self.pipeline);

        let pc = BloomMipPc { src_size: self.src_size, src_mip: self.src_mip, _pad: 0 };
        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let ds = ctx.create_descriptor_set(
            self.pipeline,
            0,
            &[
                DescriptorWrite::sampled_image_mip(
                    0, 0,
                    self.bloom_buf,
                    DescriptorImageLayout::General,
                    self.src_mip,
                    1,
                ),
                DescriptorWrite::storage_image_mip(
                    1, 0,
                    self.bloom_buf,
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

// ─────────────────────────────────────────────────────────────────────────────

struct BloomUpSubPass {
    bloom_buf: ImageHandle,
    pipeline:  BackendPipeline,
    src_mip:   u32,       // coarser mip (source)
    src_size:  [u32; 2],  // dimensions of src_mip
}

impl RenderPass for BloomUpSubPass {
    fn name(&self) -> &str { "BloomUp" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.read_image(self.bloom_buf,  ResourceUsage::SHADER_READ);
        builder.write_image(self.bloom_buf, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        ctx.bind_pipeline_raw(self.pipeline);

        let pc = BloomMipPc { src_size: self.src_size, src_mip: self.src_mip, _pad: 0 };
        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let dst_mip = self.src_mip - 1;
        let ds = ctx.create_descriptor_set(
            self.pipeline,
            0,
            &[
                DescriptorWrite::sampled_image_mip(
                    0, 0,
                    self.bloom_buf,
                    DescriptorImageLayout::General,
                    self.src_mip,
                    1,
                ),
                DescriptorWrite::storage_image_mip(
                    1, 0,
                    self.bloom_buf,
                    DescriptorImageLayout::General,
                    dst_mip,
                ),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        // Destination mip is one level finer (2× each dim).
        let dst_w = (self.src_size[0] * 2).max(1);
        let dst_h = (self.src_size[1] * 2).max(1);
        ctx.dispatch((dst_w + 7) / 8, (dst_h + 7) / 8, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────

struct BloomCompositeSubPass {
    hdr_target: ImageHandle,
    bloom_buf:  ImageHandle,
    pipeline:   BackendPipeline,
    intensity:  f32,
}

impl RenderPass for BloomCompositeSubPass {
    fn name(&self) -> &str { "BloomComposite" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.read_image(self.bloom_buf,   ResourceUsage::SHADER_READ);
        builder.read_image(self.hdr_target,  ResourceUsage::SHADER_READ);
        builder.write_image(self.hdr_target, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        ctx.bind_pipeline_raw(self.pipeline);

        let pc = BloomCompositePc { intensity: self.intensity, _pad: [0.0; 3] };
        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let ds = ctx.create_descriptor_set(
            self.pipeline,
            0,
            &[
                // bloom_buf is only READ here — frame graph transitions to
                // ShaderReadOnlyOptimal, so the descriptor must match.
                DescriptorWrite::sampled_image_mip(
                    0, 0,
                    self.bloom_buf,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                    0,
                    1,
                ),
                DescriptorWrite::storage_image_mip(
                    1, 0,
                    self.hdr_target,
                    DescriptorImageLayout::General,
                    0,
                ),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let w = common.screen_width;
        let h = common.screen_height;
        ctx.dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }
}
