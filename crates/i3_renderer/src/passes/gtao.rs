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

// ─── Push constants (must match gtao_main.slang) ──────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GtaoPushConstants {
    inv_projection: [[f32; 4]; 4],
    view:           [[f32; 4]; 4],
    screen_size:    [f32; 2],
    radius:         f32,
    falloff:        f32,
    slice_count:    u32,
    step_count:     u32,
    frame_index:    u32,
    enabled:        u32,
}

// ─── GtaoPass ─────────────────────────────────────────────────────────────────

/// GTAO main compute pass.
///
/// Declares `AO_Raw` (R8_UNORM) as a screen-sized transient output.
/// Reads `DepthBuffer` and `GBuffer_Normal`.
///
/// When `enabled = false` the dispatch is skipped and the pass writes 1.0
/// (no occlusion) via the shader's early-out branch — the resource is still
/// declared so that `DeferredResolvePass` can always find `AO_Raw`.
pub struct GtaoPass {
    pub enabled:     bool,
    pub radius:      f32,
    pub falloff:     f32,
    pub slice_count: u32,
    pub step_count:  u32,

    pipeline:    Option<BackendPipeline>,
    frame_index: u32,

    // Handles resolved in declare(), used in execute()
    depth_buffer:   ImageHandle,
    gbuffer_normal: ImageHandle,
    ao_raw:         ImageHandle,
}

impl GtaoPass {
    pub fn new() -> Self {
        Self {
            enabled:     true,
            radius:      0.5,
            falloff:     2.0,
            slice_count: 2,
            step_count:  5,
            pipeline:    None,
            frame_index: 0,
            depth_buffer:   ImageHandle::INVALID,
            gbuffer_normal: ImageHandle::INVALID,
            ao_raw:         ImageHandle::INVALID,
        }
    }
}

impl RenderPass for GtaoPass {
    fn name(&self) -> &str {
        "GtaoPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("gtao_main")
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
        let (w, h) = (depth_desc.width, depth_desc.height);

        // Declare the transient AO_Raw output
        builder.declare_image_output(
            "AO_Raw",
            ImageDesc {
                width:        w,
                height:       h,
                depth:        1,
                format:       Format::R32_SFLOAT,
                mip_levels:   1,
                array_layers: 1,
                usage:        ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
                view_type:    ImageViewType::Type2D,
                swizzle:      ComponentMapping::default(),
                clear_value:  None,
            },
        );
        self.ao_raw = builder.resolve_image("AO_Raw");

        builder.read_image(self.depth_buffer,   ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal, ResourceUsage::SHADER_READ);
        builder.write_image(self.ao_raw,        ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else { return; };

        let common  = frame.consume::<crate::render_graph::CommonData>("Common");
        let width   = common.screen_width;
        let height  = common.screen_height;

        ctx.bind_pipeline_raw(pipeline);

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        // mat4 → [[f32;4];4] column-major (nalgebra is column-major)
        let inv_proj: [[f32; 4]; 4] = *common.inv_projection.as_ref();
        let view:     [[f32; 4]; 4] = *common.view.as_ref();

        let pc = GtaoPushConstants {
            inv_projection: inv_proj,
            view,
            screen_size:    [width as f32, height as f32],
            radius:         self.radius,
            falloff:        self.falloff,
            slice_count:    self.slice_count,
            step_count:     self.step_count,
            frame_index:    self.frame_index,
            enabled:        self.enabled as u32,
        };
        ctx.push_bytes(ShaderStageFlags::Compute, 0, bytemuck::bytes_of(&pc));

        let ds = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::sampled_image(
                    0, 0,
                    self.depth_buffer,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    1, 0,
                    self.gbuffer_normal,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::storage_image(
                    2, 0,
                    self.ao_raw,
                    DescriptorImageLayout::General,
                ),
            ],
        );
        ctx.bind_descriptor_set(0, ds);

        let gx = (width  + 7) / 8;
        let gy = (height + 7) / 8;
        ctx.dispatch(gx, gy, 1);
    }
}

impl GtaoPass {
    /// Called each frame from `DefaultRenderGraph::render()` to advance the temporal counter.
    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}
