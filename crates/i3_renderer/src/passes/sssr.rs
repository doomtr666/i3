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
    pub thickness: f32,
    pub enabled: u32,
    pub downsample_factor: u32,
    pub max_mip_level: u32,
    /// 0 = normal output, 1 = debug viz (R=hit, G=iter/96, B=thickness_ratio)
    pub debug_mode: u32,
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
    pub thickness: f32,
    pub blue_noise_index: u32,
    pub downsample_factor: u32,
    /// Highest mip to sample from HDR_Mips (roughness=1 → this mip).
    pub max_mip_level: u32,
    /// 0 = normal, 1 = debug viz (R=hit, G=iter/96, B=thickness_ratio).
    pub debug_mode: u32,

    pipeline: Option<BackendPipeline>,
    frame_index: u32,

    depth_buffer: ImageHandle,
    hiz_buffer: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    hdr_mips: ImageHandle,
    ssr_raw: ImageHandle,
    common_buffer: BufferHandle,
}

impl SssrSamplePass {
    pub fn new() -> Self {
        Self {
            enabled: true,
            thickness: 0.15,
            blue_noise_index: 0,
            downsample_factor: 1,
            max_mip_level: 5,
            debug_mode: 0,
            pipeline: None,
            frame_index: 0,
            depth_buffer: ImageHandle::INVALID,
            hiz_buffer: ImageHandle::INVALID,
            gbuffer_albedo: ImageHandle::INVALID,
            gbuffer_normal: ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            hdr_mips: ImageHandle::INVALID,
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
        self.hdr_mips = builder.resolve_image("HDR_Mips");

        let depth_desc = builder.get_image_desc(self.depth_buffer);
        let (w, h) = (
            depth_desc.width / self.downsample_factor,
            depth_desc.height / self.downsample_factor,
        );

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
        builder.read_image(self.hdr_mips, ResourceUsage::SHADER_READ);
        builder.write_image(self.ssr_raw, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            return;
        };

        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let w = common.screen_width;
        let h = common.screen_height;
        let factor = self.downsample_factor;

        let pc = SssrSamplePc {
            thickness: self.thickness,
            enabled: self.enabled as u32,
            downsample_factor: factor,
            max_mip_level: self.max_mip_level,
            debug_mode: self.debug_mode,
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
                    self.hdr_mips,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::storage_image(6, 0, self.ssr_raw, DescriptorImageLayout::General),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));
        ctx.dispatch(div_ceil(w / factor, 8), div_ceil(h / factor, 8), 1);
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
    ssr_upsampled: ImageHandle,
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
            ssr_upsampled: ImageHandle::INVALID,
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
        self.ssr_upsampled = builder.resolve_image("SSR_Upsampled");
        self.ao_resolved = builder.resolve_image("AO_Resolved");
        self.hdr_target = builder.resolve_image("HDR_Target");
        self.common_buffer = builder.resolve_buffer("CommonBuffer");

        builder.read_image(self.depth_buffer, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_albedo, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        builder.read_image(self.ssr_upsampled, ResourceUsage::SHADER_READ);
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
                    self.ssr_upsampled,
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

// ─── SssrBilateralUpsamplePass ───────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct SssrBilateralUpsamplePc {
    downsample_factor: u32,
}

/// SSSR bilateral upsampling: reconstructs full-res SSR_Upsampled from low-res SSR_Raw.
///
/// Joint bilateral filter using full-res depth and normals as edge-stopping criteria.
/// Only useful when `downsample_factor > 1`; degenerates to a plain copy at factor = 1.
pub struct SssrBilateralUpsamplePass {
    pub downsample_factor: u32,

    pipeline: Option<BackendPipeline>,
    depth_buffer: ImageHandle,
    gbuffer_normal: ImageHandle,
    ssr_resolved: ImageHandle,
    ssr_upsampled: ImageHandle,
    common_buffer: BufferHandle,
}

impl SssrBilateralUpsamplePass {
    pub fn new() -> Self {
        Self {
            downsample_factor: 1,
            pipeline: None,
            depth_buffer: ImageHandle::INVALID,
            gbuffer_normal: ImageHandle::INVALID,
            ssr_resolved: ImageHandle::INVALID,
            ssr_upsampled: ImageHandle::INVALID,
            common_buffer: BufferHandle::INVALID,
        }
    }
}

impl RenderPass for SssrBilateralUpsamplePass {
    fn name(&self) -> &str {
        "SssrBilateralUpsamplePass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("sssr_bilateral_upsample")
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
        self.ssr_resolved = builder.resolve_image("SSR_Raw");
        self.common_buffer = builder.resolve_buffer("CommonBuffer");

        let depth_desc = builder.get_image_desc(self.depth_buffer);
        builder.declare_image_output(
            "SSR_Upsampled",
            ImageDesc {
                width: depth_desc.width,
                height: depth_desc.height,
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
        self.ssr_upsampled = builder.resolve_image("SSR_Upsampled");

        builder.read_buffer(self.common_buffer, ResourceUsage::SHADER_READ);
        builder.read_image(self.depth_buffer, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal, ResourceUsage::SHADER_READ);
        builder.read_image(self.ssr_resolved, ResourceUsage::SHADER_READ);
        builder.write_image(self.ssr_upsampled, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            return;
        };

        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let w = common.screen_width;
        let h = common.screen_height;

        let pc = SssrBilateralUpsamplePc {
            downsample_factor: self.downsample_factor,
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
                    self.ssr_resolved,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::storage_image(
                    3,
                    0,
                    self.ssr_upsampled,
                    DescriptorImageLayout::General,
                ),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));
        ctx.dispatch(div_ceil(w, 8), div_ceil(h, 8), 1);
    }
}
