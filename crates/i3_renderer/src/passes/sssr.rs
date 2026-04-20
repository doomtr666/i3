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

// ─── Push constants ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SssrSamplePc {
    pub inv_projection:    [[f32; 4]; 4],
    pub projection:        [[f32; 4]; 4],
    pub view:              [[f32; 4]; 4],
    pub screen_size:       [f32; 2],
    pub max_distance:      f32,
    pub thickness:         f32,
    pub roughness_cutoff:  f32,
    pub max_steps:         u32,
    pub frame_index:       u32,
    pub enabled:           u32,
    pub blue_noise_index:  u32, // bindless index of the 64×64 RG8 blue-noise texture
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SssrTemporalPc {
    inv_view_proj:  [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    screen_size:    [f32; 2],
    alpha:          f32,
    frame_index:    u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SssrCompositePc {
    inv_view_proj:  [[f32; 4]; 4],
    camera_pos:     [f32; 3],
    intensity:      f32,
    screen_size:    [f32; 2],
    ibl_lut_index:  u32,
    ibl_pref_index: u32,
    ibl_intensity:  f32,
    _pad:           f32,
}

// ─── SssrSamplePass ──────────────────────────────────────────────────────────

/// SSSR sample pass: one GGX-importance-sampled ray per pixel.
///
/// Declares `SSR_Raw` as a transient output (RGBA16F, full resolution).
/// RGB = hit colour, A = confidence.
pub struct SssrSamplePass {
    pub enabled:           bool,
    pub max_steps:         u32,
    pub thickness:         f32,
    pub max_distance:      f32,
    pub roughness_cutoff:  f32,
    /// Bindless index of the 64×64 RG8 blue-noise texture (set by the render graph).
    pub blue_noise_index:  u32,

    pipeline:    Option<BackendPipeline>,
    frame_index: u32,

    depth_buffer:       ImageHandle,
    gbuffer_normal:     ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    hdr_target:         ImageHandle,
    ssr_raw:            ImageHandle,
}

impl SssrSamplePass {
    pub fn new() -> Self {
        Self {
            enabled:           true,
            max_steps:         32,
            thickness:         0.15,
            max_distance:      30.0,
            roughness_cutoff:  0.75,
            blue_noise_index:  0,
            pipeline:          None,
            frame_index:       0,
            depth_buffer:       ImageHandle::INVALID,
            gbuffer_normal:     ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            hdr_target:         ImageHandle::INVALID,
            ssr_raw:            ImageHandle::INVALID,
        }
    }

    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

impl RenderPass for SssrSamplePass {
    fn name(&self) -> &str { "SssrSamplePass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("sssr_sample")
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
        builder.read_image(self.hdr_target,         ResourceUsage::SHADER_READ);
        builder.write_image(self.ssr_raw,           ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else { return; };

        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let w = common.screen_width;
        let h = common.screen_height;

        let pc = SssrSamplePc {
            inv_projection:   *common.inv_projection.as_ref(),
            projection:       *common.projection.as_ref(),
            view:             *common.view.as_ref(),
            screen_size:      [w as f32, h as f32],
            max_distance:     self.max_distance,
            thickness:        self.thickness,
            roughness_cutoff: self.roughness_cutoff,
            max_steps:        self.max_steps,
            frame_index:      self.frame_index,
            enabled:          self.enabled as u32,
            blue_noise_index: self.blue_noise_index,
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
                DescriptorWrite::sampled_image(3, 0, self.hdr_target,         DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::storage_image(4, 0, self.ssr_raw,            DescriptorImageLayout::General),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));
        ctx.dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }
}

// ─── SssrTemporalPass ────────────────────────────────────────────────────────

/// SSSR temporal accumulation: EMA over SSR_Raw → SSR_Resolved.
pub struct SssrTemporalPass {
    pub alpha: f32,

    pipeline:    Option<BackendPipeline>,
    frame_index: u32,

    depth_buffer: ImageHandle,
    ssr_raw:      ImageHandle,
    ssr_history:  ImageHandle,
    ssr_resolved: ImageHandle,
}

impl SssrTemporalPass {
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

impl RenderPass for SssrTemporalPass {
    fn name(&self) -> &str { "SssrTemporalPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("sssr_temporal")
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

        builder.read_image(self.depth_buffer,  ResourceUsage::SHADER_READ);
        builder.read_image(self.ssr_raw,       ResourceUsage::SHADER_READ);
        builder.read_image(self.ssr_history,   ResourceUsage::SHADER_READ);
        builder.write_image(self.ssr_resolved, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else { return; };

        let common  = frame.consume::<crate::render_graph::CommonData>("Common");
        let prev_vp = frame.consume::<nalgebra_glm::Mat4>("PrevViewProjection");
        let w = common.screen_width;
        let h = common.screen_height;

        let pc = SssrTemporalPc {
            inv_view_proj:  *common.inv_view_projection.as_ref(),
            prev_view_proj: *prev_vp.as_ref(),
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

// ─── SssrCompositePass ───────────────────────────────────────────────────────

/// SSSR composite: replaces the IBL specular with the SSR estimate where confident.
///
/// Formula: hdr += conf × (ssr_spec − ibl_spec)
///   where ssr_spec = ssr_resolved.rgb × brdf_weight × ibl_intensity
///   and   ibl_spec = prefiltered_env  × brdf_weight × ibl_intensity
pub struct SssrCompositePass {
    pub intensity: f32,

    pipeline:           Option<BackendPipeline>,
    depth_buffer:       ImageHandle,
    gbuffer_normal:     ImageHandle,
    gbuffer_albedo:     ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    ssr_resolved:       ImageHandle,
    ao_resolved:        ImageHandle,
    hdr_target:         ImageHandle,
}

impl SssrCompositePass {
    pub fn new() -> Self {
        Self {
            intensity:          1.0,
            pipeline:           None,
            depth_buffer:       ImageHandle::INVALID,
            gbuffer_normal:     ImageHandle::INVALID,
            gbuffer_albedo:     ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            ssr_resolved:       ImageHandle::INVALID,
            ao_resolved:        ImageHandle::INVALID,
            hdr_target:         ImageHandle::INVALID,
        }
    }
}

impl RenderPass for SssrCompositePass {
    fn name(&self) -> &str { "SssrCompositePass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("sssr_composite")
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
        self.gbuffer_albedo     = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.ssr_resolved       = builder.resolve_image("SSR_Resolved");
        self.ao_resolved        = builder.resolve_image("AO_Resolved");
        self.hdr_target         = builder.resolve_image("HDR_Target");

        builder.read_image(self.depth_buffer,       ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal,     ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_albedo,     ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        builder.read_image(self.ssr_resolved,       ResourceUsage::SHADER_READ);
        builder.read_image(self.ao_resolved,        ResourceUsage::SHADER_READ);
        builder.read_image(self.hdr_target,         ResourceUsage::SHADER_READ);
        builder.write_image(self.hdr_target,        ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else { return; };

        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let ibl    = frame.consume::<crate::render_graph::IblIndices>("IblIndices");
        let w = common.screen_width;
        let h = common.screen_height;

        let cam = common.camera_pos;
        let pc = SssrCompositePc {
            inv_view_proj:  *common.inv_view_projection.as_ref(),
            camera_pos:     [cam.x, cam.y, cam.z],
            intensity:      self.intensity,
            screen_size:    [w as f32, h as f32],
            ibl_lut_index:  ibl.lut_index,
            ibl_pref_index: ibl.pref_index,
            ibl_intensity:  ibl.intensity_scale,
            _pad:           0.0,
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
                DescriptorWrite::sampled_image(2, 0, self.gbuffer_albedo,     DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(3, 0, self.gbuffer_roughmetal, DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(4, 0, self.ssr_resolved,       DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(5, 0, self.ao_resolved,        DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::storage_image(6, 0, self.hdr_target,         DescriptorImageLayout::General),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));
        ctx.dispatch((w + 7) / 8, (h + 7) / 8, 1);
    }
}
