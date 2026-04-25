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

// ─── Push constants ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct SssrSamplePc {
    pub max_distance: f32,
    pub thickness: f32,
    pub roughness_cutoff: f32,
    pub max_steps: u32,
    pub enabled: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct SssrTemporalPc {
    alpha: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct SssrCompositePc {
    intensity: f32,
}

// ─── SssrSamplePass ──────────────────────────────────────────────────────────

/// SSSR sample pass: one GGX-importance-sampled ray per pixel.
///
/// Declares `SSR_Raw` as a transient output (RGBA16F, full resolution).
/// RGB = hit colour, A = confidence.
pub struct SssrSamplePass {
    pub enabled: bool,
    pub max_steps: u32,
    pub thickness: f32,
    pub max_distance: f32,
    pub roughness_cutoff: f32,
    /// Bindless index of the 64×64 RG8 blue-noise texture (set by the render graph).
    pub blue_noise_index: u32,

    pipeline: Option<BackendPipeline>,
    frame_index: u32,

    depth_buffer: ImageHandle,
    hiz_buffer: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    hdr_lit: ImageHandle,
    ssr_raw: ImageHandle,
    common_buffer: BufferHandle,
}

impl SssrSamplePass {
    pub fn new() -> Self {
        Self {
            enabled: true,
            max_steps: 32,
            thickness: 0.15,
            max_distance: 30.0,
            roughness_cutoff: 0.75,
            blue_noise_index: 0,
            pipeline: None,
            frame_index: 0,
            depth_buffer: ImageHandle::INVALID,
            hiz_buffer: ImageHandle::INVALID,
            gbuffer_albedo: ImageHandle::INVALID,
            gbuffer_normal: ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            hdr_lit: ImageHandle::INVALID,
            ssr_raw: ImageHandle::INVALID,
            common_buffer: BufferHandle::INVALID,
        }
    }

    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

impl RenderPass for SssrSamplePass {
    fn name(&self) -> &str {
        "SssrSamplePass"
    }

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
        self.depth_buffer = builder.resolve_image("DepthBuffer");
        self.hiz_buffer = builder.resolve_image("HiZFinal");
        self.gbuffer_albedo = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.hdr_lit = builder.resolve_image("HDR_Target");

        let depth_desc = builder.get_image_desc(self.depth_buffer);
        let (w, h) = (depth_desc.width, depth_desc.height);

        builder.declare_image_output(
            "SSR_Raw",
            ImageDesc {
                width: w,
                height: h,
                depth: 1,
                format: Format::R16G16B16A16_SFLOAT,
                mip_levels: 1,
                array_layers: 1,
                usage: ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
                view_type: ImageViewType::Type2D,
                swizzle: ComponentMapping::default(),
                clear_value: None,
            },
        );
        self.ssr_raw = builder.resolve_image("SSR_Raw");
        self.common_buffer = builder.resolve_buffer("CommonBuffer");

        builder.read_buffer(self.common_buffer, ResourceUsage::SHADER_READ);
        builder.read_image(self.depth_buffer, ResourceUsage::SHADER_READ);
        builder.read_image(self.hiz_buffer, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_albedo, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        builder.read_image(self.hdr_lit, ResourceUsage::SHADER_READ);
        builder.write_image(self.ssr_raw, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            return;
        };

        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let w = common.screen_width;
        let h = common.screen_height;

        let pc = SssrSamplePc {
            max_distance: self.max_distance,
            thickness: self.thickness,
            roughness_cutoff: self.roughness_cutoff,
            max_steps: self.max_steps,
            enabled: self.enabled as u32,
        };

        let common_set = ctx.create_descriptor_set(
            pipeline,
            1,
            &[DescriptorWrite::uniform_buffer(0, 0, self.common_buffer)],
        );

        ctx.bind_pipeline_raw(pipeline);

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(1, common_set);
        ctx.bind_descriptor_set(2, bindless_set);

        let ds = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::sampled_image(
                    0,
                    0,
                    self.depth_buffer,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    1,
                    0,
                    self.hiz_buffer,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    2,
                    0,
                    self.gbuffer_albedo,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    3,
                    0,
                    self.gbuffer_normal,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    4,
                    0,
                    self.gbuffer_roughmetal,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    5,
                    0,
                    self.hdr_lit,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::storage_image(6, 0, self.ssr_raw, DescriptorImageLayout::General),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));
        ctx.dispatch(div_ceil(w, 8), div_ceil(h, 8), 1);
    }
}

// ─── SssrTemporalPass ────────────────────────────────────────────────────────

/// SSSR temporal accumulation: EMA over SSR_Raw → SSR_Resolved.
pub struct SssrTemporalPass {
    pub alpha: f32,

    pipeline: Option<BackendPipeline>,
    frame_index: u32,

    depth_buffer: ImageHandle,
    ssr_raw: ImageHandle,
    ssr_history: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    hdr_lit: ImageHandle,
    ssr_resolved: ImageHandle,
    common_buffer: BufferHandle,
}

impl SssrTemporalPass {
    pub fn new() -> Self {
        Self {
            alpha: 0.92,
            pipeline: None,
            frame_index: 0,
            depth_buffer: ImageHandle::INVALID,
            ssr_raw: ImageHandle::INVALID,
            ssr_history: ImageHandle::INVALID,
            gbuffer_normal: ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            hdr_lit: ImageHandle::INVALID,
            ssr_resolved: ImageHandle::INVALID,
            common_buffer: BufferHandle::INVALID,
        }
    }

    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

impl RenderPass for SssrTemporalPass {
    fn name(&self) -> &str {
        "SssrTemporalPass"
    }

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
        self.ssr_raw = builder.resolve_image("SSR_Raw");
        self.common_buffer = builder.resolve_buffer("CommonBuffer");

        let raw_desc = builder.get_image_desc(self.ssr_raw);
        let hist_desc = ImageDesc {
            width: raw_desc.width,
            height: raw_desc.height,
            depth: 1,
            format: raw_desc.format,
            mip_levels: 1,
            array_layers: 1,
            usage: ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
            view_type: ImageViewType::Type2D,
            swizzle: ComponentMapping::default(),
            clear_value: None,
        };
        self.hdr_lit = builder.resolve_image("HDR_Target");
        self.ssr_resolved = builder.declare_image_history_output("SSR_Resolved", hist_desc);
        self.ssr_history = builder.read_image_history("SSR_Resolved");
        self.gbuffer_normal = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");

        builder.read_image(self.depth_buffer, ResourceUsage::SHADER_READ);
        builder.read_image(self.ssr_raw, ResourceUsage::SHADER_READ);
        builder.read_image(self.ssr_history, ResourceUsage::SHADER_READ);
        builder.read_image(self.hdr_lit, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.common_buffer, ResourceUsage::SHADER_READ);
        builder.write_image(self.ssr_resolved, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            return;
        };

        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let w = common.screen_width;
        let h = common.screen_height;

        let pc = SssrTemporalPc { alpha: self.alpha };

        let common_set = ctx.create_descriptor_set(
            pipeline,
            1,
            &[DescriptorWrite::uniform_buffer(0, 0, self.common_buffer)],
        );

        ctx.bind_pipeline_raw(pipeline);

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(1, common_set);
        ctx.bind_descriptor_set(2, bindless_set);

        let ds = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::sampled_image(
                    0,
                    0,
                    self.depth_buffer,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    1,
                    0,
                    self.ssr_raw,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    2,
                    0,
                    self.ssr_history,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    3,
                    0,
                    self.gbuffer_normal,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    4,
                    0,
                    self.gbuffer_roughmetal,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    5,
                    0,
                    self.hdr_lit,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::storage_image(
                    6,
                    0,
                    self.ssr_resolved,
                    DescriptorImageLayout::General,
                ),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));
        ctx.dispatch(div_ceil(w, 8), div_ceil(h, 8), 1);
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

    pipeline: Option<BackendPipeline>,
    depth_buffer: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    ssr_resolved: ImageHandle,
    ao_resolved: ImageHandle,
    hdr_target: ImageHandle,
    common_buffer: BufferHandle,
}

impl SssrCompositePass {
    pub fn new() -> Self {
        Self {
            intensity: 1.0,
            pipeline: None,
            depth_buffer: ImageHandle::INVALID,
            gbuffer_normal: ImageHandle::INVALID,
            gbuffer_albedo: ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            ssr_resolved: ImageHandle::INVALID,
            ao_resolved: ImageHandle::INVALID,
            hdr_target: ImageHandle::INVALID,
            common_buffer: BufferHandle::INVALID,
        }
    }
}

impl RenderPass for SssrCompositePass {
    fn name(&self) -> &str {
        "SssrCompositePass"
    }

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
        self.depth_buffer = builder.resolve_image("DepthBuffer");
        self.gbuffer_normal = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_albedo = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.ssr_resolved = builder.resolve_image("SSR_Resolved");
        self.ao_resolved = builder.resolve_image("AO_Resolved");
        self.hdr_target = builder.resolve_image("HDR_Target");
        self.common_buffer = builder.resolve_buffer("CommonBuffer");

        builder.read_image(self.depth_buffer, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_albedo, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        builder.read_image(self.ssr_resolved, ResourceUsage::SHADER_READ);
        builder.read_image(self.ao_resolved, ResourceUsage::SHADER_READ);
        builder.read_image(self.hdr_target, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.common_buffer, ResourceUsage::SHADER_READ);
        builder.write_image(self.hdr_target, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            return;
        };

        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let w = common.screen_width;
        let h = common.screen_height;

        let pc = SssrCompositePc {
            intensity: self.intensity,
        };

        let common_set = ctx.create_descriptor_set(
            pipeline,
            1,
            &[DescriptorWrite::uniform_buffer(0, 0, self.common_buffer)],
        );

        ctx.bind_pipeline_raw(pipeline);

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(1, common_set);
        ctx.bind_descriptor_set(2, bindless_set);

        let ds = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::sampled_image(
                    0,
                    0,
                    self.depth_buffer,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    1,
                    0,
                    self.gbuffer_normal,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    2,
                    0,
                    self.gbuffer_albedo,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    3,
                    0,
                    self.gbuffer_roughmetal,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    4,
                    0,
                    self.ssr_resolved,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    5,
                    0,
                    self.ao_resolved,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::storage_image(
                    6,
                    0,
                    self.hdr_target,
                    DescriptorImageLayout::General,
                ),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));
        ctx.dispatch(div_ceil(w, 8), div_ceil(h, 8), 1);
    }
}
