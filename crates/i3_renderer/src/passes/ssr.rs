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

// ─── Push constants (must match SsrPushConstants in ssr_main.slang) ──────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SsrPushConstants {
    pub inv_projection:   [[f32; 4]; 4],
    pub projection:       [[f32; 4]; 4],
    pub view:             [[f32; 4]; 4],
    pub screen_size:      [f32; 2],
    pub max_distance:     f32,
    pub thickness:        f32,
    pub roughness_cutoff: f32,
    pub max_steps:        u32,
    pub max_mip:          u32,
    pub frame_index:      u32,
    pub enabled:          u32,
    pub _pad:             u32,
}

// ─── SsrMainPass ──────────────────────────────────────────────────────────────

/// SSR main compute pass.
///
/// Declares `SSR_Raw` (R16G16B16A16_SFLOAT) as a transient output.
/// RGB = reflected colour, A = hit confidence [0, 1].
///
/// When `enabled = false` (or pipeline not yet loaded) the pass writes
/// `float4(0)` — `SSR_Resolved` will then be transparent black, ensuring
/// downstream passes (debug viz, composite) always find the resource.
#[allow(dead_code)]
pub struct SsrMainPass {
    pub enabled:          bool,
    pub max_steps:        u32,
    pub thickness:        f32,
    pub max_distance:     f32,
    pub roughness_cutoff: f32,

    pipeline:    Option<BackendPipeline>,
    frame_index: u32,

    depth_buffer:      ImageHandle,
    gbuffer_normal:    ImageHandle,
    gbuffer_roughmetal:ImageHandle,
    hiz_final:         ImageHandle,
    hdr_target:        ImageHandle,
    ssr_raw:           ImageHandle,
}

impl SsrMainPass {
    pub fn new() -> Self {
        Self {
            enabled:          true,
            max_steps:        48,
            thickness:        0.15,
            max_distance:     50.0,
            roughness_cutoff: 0.75,
            pipeline:         None,
            frame_index:      0,
            depth_buffer:       ImageHandle::INVALID,
            gbuffer_normal:     ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            hiz_final:          ImageHandle::INVALID,
            hdr_target:         ImageHandle::INVALID,
            ssr_raw:            ImageHandle::INVALID,
        }
    }

    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

impl RenderPass for SsrMainPass {
    fn name(&self) -> &str { "SsrMainPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("ssr_main")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.depth_buffer       = builder.resolve_image("DepthBuffer");
        self.gbuffer_normal     = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.hiz_final          = builder.resolve_image("HiZFinal");
        self.hdr_target         = builder.resolve_image("HDR_Target");

        let depth_desc = builder.get_image_desc(self.depth_buffer);
        let (w, h) = (depth_desc.width, depth_desc.height);

        builder.declare_image_output(
            "SSR_Raw",
            ImageDesc {
                width:        w,
                height:       h,
                depth:        1,
                format:       Format::R16G16B16A16_SFLOAT,
                mip_levels:   1,
                array_layers: 1,
                usage:        ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
                view_type:    ImageViewType::Type2D,
                swizzle:      ComponentMapping::default(),
                clear_value:  None,
            },
        );
        self.ssr_raw = builder.resolve_image("SSR_Raw");

        builder.read_image(self.depth_buffer,       ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal,     ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        builder.read_image(self.hiz_final,          ResourceUsage::SHADER_READ);
        builder.read_image(self.hdr_target,         ResourceUsage::SHADER_READ);
        builder.write_image(self.ssr_raw,           ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else { return; };

        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let w = common.screen_width;
        let h = common.screen_height;

        let inv_proj: [[f32; 4]; 4] = *common.inv_projection.as_ref();
        let proj:     [[f32; 4]; 4] = *common.projection.as_ref();
        let view:     [[f32; 4]; 4] = *common.view.as_ref();

        let mip_levels = {
            let max_dim = w.max(h);
            if max_dim > 0 { (31 - max_dim.leading_zeros()) + 1 } else { 1 }
        };
        let max_mip = (mip_levels - 1).min(10); // cap to avoid over-large steps

        let pc = SsrPushConstants {
            inv_projection:   inv_proj,
            projection:       proj,
            view,
            screen_size:      [w as f32, h as f32],
            max_distance:     self.max_distance,
            thickness:        self.thickness,
            roughness_cutoff: self.roughness_cutoff,
            max_steps:        self.max_steps,
            max_mip,
            frame_index:      self.frame_index,
            enabled:          self.enabled as u32,
            _pad:             0,
        };

        ctx.bind_pipeline_raw(pipeline);

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let ds = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::sampled_image(0, 0, self.depth_buffer,       DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(1, 0, self.gbuffer_normal,     DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(2, 0, self.gbuffer_roughmetal, DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(3, 0, self.hiz_final,          DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(4, 0, self.hdr_target,         DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::storage_image(5, 0, self.ssr_raw,            DescriptorImageLayout::General),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        ctx.dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }
}

// ─── SsrTemporalPass ─────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsrTemporalPushConstants {
    inv_view_proj:  [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    screen_size:    [f32; 2],
    alpha:          f32,
    frame_index:    u32,
}



/// SSR temporal accumulation pass.
///
/// Declares `SSR_Resolved` as a double-buffered history image.
/// Reads `SSR_Raw` (current frame) + `SSR_Resolved` history (N-1).
/// Writes the blended result back to `SSR_Resolved` (N).
#[allow(dead_code)]
pub struct SsrTemporalPass {
    pub alpha: f32,

    pipeline:    Option<BackendPipeline>,
    frame_index: u32,

    depth_buffer: ImageHandle,
    ssr_raw:      ImageHandle,
    ssr_history:  ImageHandle,
    ssr_resolved: ImageHandle,
}

impl SsrTemporalPass {
    pub fn new() -> Self {
        Self {
            alpha:        0.92,
            pipeline:     None,
            frame_index:  0,
            depth_buffer: ImageHandle::INVALID,
            ssr_raw:      ImageHandle::INVALID,
            ssr_history:  ImageHandle::INVALID,
            ssr_resolved: ImageHandle::INVALID,
        }
    }

    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

impl RenderPass for SsrTemporalPass {
    fn name(&self) -> &str { "SsrTemporalPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("ssr_temporal")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.depth_buffer = builder.resolve_image("DepthBuffer");
        self.ssr_raw      = builder.resolve_image("SSR_Raw");

        let raw_desc = builder.get_image_desc(self.ssr_raw);
        let hist_desc = ImageDesc {
            width:        raw_desc.width,
            height:       raw_desc.height,
            depth:        1,
            format:       raw_desc.format,
            mip_levels:   1,
            array_layers: 1,
            usage:        ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
            view_type:    ImageViewType::Type2D,
            swizzle:      ComponentMapping::default(),
            clear_value:  None,
        };

        self.ssr_resolved = builder.declare_image_history_output("SSR_Resolved", hist_desc);
        self.ssr_history  = builder.read_image_history("SSR_Resolved");

        builder.read_image(self.depth_buffer, ResourceUsage::SHADER_READ);
        builder.read_image(self.ssr_raw,      ResourceUsage::SHADER_READ);
        builder.read_image(self.ssr_history,  ResourceUsage::SHADER_READ);
        builder.write_image(self.ssr_resolved,ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else { return; };

        let common   = frame.consume::<crate::render_graph::CommonData>("Common");
        let prev_vp  = frame.consume::<nalgebra_glm::Mat4>("PrevViewProjection");
        let w = common.screen_width;
        let h = common.screen_height;

        let inv_vp: [[f32; 4]; 4] = *common.inv_view_projection.as_ref();
        let prev:   [[f32; 4]; 4] = *prev_vp.as_ref();

        let pc = SsrTemporalPushConstants {
            inv_view_proj:  inv_vp,
            prev_view_proj: prev,
            screen_size:    [w as f32, h as f32],
            alpha:          self.alpha,
            frame_index:    self.frame_index,
        };

        ctx.bind_pipeline_raw(pipeline);

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let ds = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::sampled_image(0, 0, self.depth_buffer, DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(1, 0, self.ssr_raw,      DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(2, 0, self.ssr_history,  DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::storage_image(3, 0, self.ssr_resolved, DescriptorImageLayout::General),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        ctx.dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }
}

// ─── SsrCompositePass ────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsrCompositePushConstants {
    intensity: f32,
    _pad: [f32; 3],
}

/// SSR composite — blends `SSR_Resolved` onto `HDR_Target` (mip 0, in-place).
///
/// Weight = F0 × (1 − roughness) × confidence × intensity.
/// No Fresnel: the quintic Schlick at grazing causes pathological over-reflection
/// on dielectrics. Plain F0 is predictable; intensity slider handles calibration.
#[allow(dead_code)]
pub struct SsrCompositePass {
    pub intensity: f32,

    pipeline:           Option<BackendPipeline>,
    gbuffer_albedo:     ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    ssr_resolved:       ImageHandle,
    hdr_target:         ImageHandle,
}

impl SsrCompositePass {
    pub fn new() -> Self {
        Self {
            intensity:          1.0,
            pipeline:           None,
            gbuffer_albedo:     ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            ssr_resolved:       ImageHandle::INVALID,
            hdr_target:         ImageHandle::INVALID,
        }
    }
}

impl RenderPass for SsrCompositePass {
    fn name(&self) -> &str { "SsrCompositePass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("ssr_composite")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.gbuffer_albedo     = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.ssr_resolved       = builder.resolve_image("SSR_Resolved");
        self.hdr_target         = builder.resolve_image("HDR_Target");

        builder.read_image(self.gbuffer_albedo,     ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        builder.read_image(self.ssr_resolved,       ResourceUsage::SHADER_READ);
        builder.read_image(self.hdr_target,  ResourceUsage::SHADER_READ);
        builder.write_image(self.hdr_target, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else { return; };

        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let w = common.screen_width;
        let h = common.screen_height;

        let pc = SsrCompositePushConstants { intensity: self.intensity, _pad: [0.0; 3] };

        ctx.bind_pipeline_raw(pipeline);

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let ds = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::sampled_image(0, 0, self.gbuffer_albedo,     DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(1, 0, self.gbuffer_roughmetal, DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(2, 0, self.ssr_resolved,       DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::storage_image(3, 0, self.hdr_target,         DescriptorImageLayout::General),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));
        ctx.dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }
}
