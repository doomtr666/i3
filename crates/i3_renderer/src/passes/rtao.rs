use crate::constants::div_ceil;
use bytemuck::{Pod, Zeroable};
use i3_gfx::graph::backend::{
    BackendPipeline, DescriptorImageLayout, DescriptorSetHandle, DescriptorWrite, PassContext,
    RenderBackend,
};
use i3_gfx::graph::types::AccelerationStructureHandle;
use i3_gfx::graph::compiler::FrameBlackboard;
use i3_gfx::graph::pass::{PassBuilder, RenderPass};
use i3_gfx::graph::pipeline::ShaderStageFlags;
use i3_gfx::graph::types::*;
use std::sync::Arc;

// ─── RtaoPass ────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RtaoPushConstants {
    inv_view_proj:    [[f32; 4]; 4],
    screen_size:      [f32; 2],
    radius:           f32,
    frame_index:      u32,
    blue_noise_index: u32,
}

/// Shoots one cosine-weighted visibility ray per pixel, outputs raw noisy AO to RTAO_Raw.
/// Temporal accumulation is handled by RtaoTemporalPass.
pub struct RtaoPass {
    pub radius:           f32,
    pub blue_noise_index: u32,

    pipeline:    Option<BackendPipeline>,
    frame_index: u32,

    depth_buffer:   ImageHandle,
    gbuffer_normal: ImageHandle,
    rtao_raw:       ImageHandle,
    tlas_handle:    AccelerationStructureHandle,
}

impl RtaoPass {
    pub fn new() -> Self {
        Self {
            radius:           1.0,
            blue_noise_index: 0,
            pipeline:         None,
            frame_index:      0,
            depth_buffer:     ImageHandle::INVALID,
            gbuffer_normal:   ImageHandle::INVALID,
            rtao_raw:         ImageHandle::INVALID,
            tlas_handle:      AccelerationStructureHandle::INVALID,
        }
    }

    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

impl RenderPass for RtaoPass {
    fn name(&self) -> &str {
        "RtaoPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("rtao_main")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.depth_buffer   = builder.resolve_image("DepthBuffer");
        self.gbuffer_normal = builder.resolve_image("GBuffer_Normal");

        let depth_desc = builder.get_image_desc(self.depth_buffer);
        let img_desc = ImageDesc {
            width:        depth_desc.width,
            height:       depth_desc.height,
            depth:        1,
            format:       Format::R32_SFLOAT,
            mip_levels:   1,
            array_layers: 1,
            usage:        ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
            view_type:    ImageViewType::Type2D,
            swizzle:      ComponentMapping::default(),
            clear_value:  None,
        };

        builder.declare_image_output("RTAO_Raw", img_desc);
        self.rtao_raw = builder.resolve_image("RTAO_Raw");

        builder.read_image(self.depth_buffer,   ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal, ResourceUsage::SHADER_READ);
        builder.write_image(self.rtao_raw,      ResourceUsage::SHADER_WRITE);

        self.tlas_handle = builder
            .try_resolve_acceleration_structure("TLAS")
            .unwrap_or(AccelerationStructureHandle::INVALID);
        if self.tlas_handle != AccelerationStructureHandle::INVALID {
            builder.read_acceleration_structure(self.tlas_handle, ResourceUsage::SHADER_READ);
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else { return; };

        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let width  = common.screen_width;
        let height = common.screen_height;

        ctx.bind_pipeline_raw(pipeline);

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let pc = RtaoPushConstants {
            inv_view_proj:    *common.inv_view_projection.as_ref(),
            screen_size:      [width as f32, height as f32],
            radius:           self.radius,
            frame_index:      self.frame_index,
            blue_noise_index: self.blue_noise_index,
        };
        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        let ds = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::sampled_image(0, 0, self.depth_buffer,   DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(1, 0, self.gbuffer_normal, DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::storage_image(2, 0, self.rtao_raw,       DescriptorImageLayout::General),
                DescriptorWrite::acceleration_structure(3, 0, self.tlas_handle),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.dispatch(div_ceil(width, 8), div_ceil(height, 8), 1);
    }
}

// ─── RtaoTemporalPass ────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RtaoTemporalPushConstants {
    inv_view_proj:  [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    screen_size:    [f32; 2],
    alpha:          f32,
    frame_index:    u32,
}

/// EMA temporal accumulation for RTAO with reprojection.
/// Blends RTAO_Raw with reprojected AO_Resolved history — no reset on camera movement.
pub struct RtaoTemporalPass {
    pub alpha: f32,

    pipeline:     Option<BackendPipeline>,
    frame_index:  u32,

    depth_buffer: ImageHandle,
    rtao_raw:     ImageHandle,
    ao_history:   ImageHandle,
    ao_resolved:  ImageHandle,
}

impl RtaoTemporalPass {
    pub fn new() -> Self {
        Self {
            alpha:        0.05,
            pipeline:     None,
            frame_index:  0,
            depth_buffer: ImageHandle::INVALID,
            rtao_raw:     ImageHandle::INVALID,
            ao_history:   ImageHandle::INVALID,
            ao_resolved:  ImageHandle::INVALID,
        }
    }

    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

impl RenderPass for RtaoTemporalPass {
    fn name(&self) -> &str {
        "RtaoTemporalPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("rtao_temporal")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.depth_buffer = builder.resolve_image("DepthBuffer");
        self.rtao_raw     = builder.resolve_image("RTAO_Raw");

        let raw_desc  = builder.get_image_desc(self.rtao_raw);
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

        self.ao_resolved = builder.declare_image_history_output("AO_Resolved", hist_desc);
        self.ao_history  = builder.read_image_history("AO_Resolved");

        builder.read_image(self.depth_buffer, ResourceUsage::SHADER_READ);
        builder.read_image(self.rtao_raw,     ResourceUsage::SHADER_READ);
        builder.read_image(self.ao_history,   ResourceUsage::SHADER_READ);
        builder.write_image(self.ao_resolved, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else { return; };

        let common   = frame.consume::<crate::render_graph::CommonData>("Common");
        let prev_vp  = frame.consume::<nalgebra_glm::Mat4>("PrevViewProjection");
        let width    = common.screen_width;
        let height   = common.screen_height;

        ctx.bind_pipeline_raw(pipeline);

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let pc = RtaoTemporalPushConstants {
            inv_view_proj:  *common.inv_view_projection.as_ref(),
            prev_view_proj: *prev_vp.as_ref(),
            screen_size:    [width as f32, height as f32],
            alpha:          self.alpha,
            frame_index:    self.frame_index,
        };
        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        let ds = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::sampled_image(0, 0, self.depth_buffer, DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(1, 0, self.rtao_raw,     DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::sampled_image(2, 0, self.ao_history,   DescriptorImageLayout::ShaderReadOnlyOptimal),
                DescriptorWrite::storage_image(3, 0, self.ao_resolved,  DescriptorImageLayout::General),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        ctx.dispatch(div_ceil(width, 8), div_ceil(height, 8), 1);
    }
}
