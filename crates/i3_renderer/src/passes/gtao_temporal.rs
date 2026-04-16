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
struct TemporalPushConstants {
    inv_view_proj:  [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    screen_size:    [f32; 2],
    alpha:          f32,
    frame_index:    u32,
}

/// GTAO Temporal Accumulation pass with reprojection.
///
/// For each pixel:
///   1. Reconstruct world position from depth + inv(VP_current).
///   2. Project into the previous frame with VP_prev to find the history UV.
///   3. Sample `AO_History` at the reprojected UV (bilinear).
///   4. If the reprojected UV falls outside [0,1] (disocclusion / newly visible),
///      skip history and write the raw sample directly.
///   5. Otherwise blend: `resolved = lerp(history, current, alpha)`.
///
/// `AO_Resolved` is declared as a temporal history image so it automatically
/// becomes the history input on the next frame (ping-pong).
pub struct GtaoTemporalPass {
    /// Target blend weight for the incoming frame (0.1 = 10% new, 90% history).
    pub alpha: f32,

    pipeline:    Option<BackendPipeline>,
    frame_index: u32,

    depth_buffer: ImageHandle,
    ao_raw:       ImageHandle,
    ao_history:   ImageHandle,
    ao_resolved:  ImageHandle,
}

impl GtaoTemporalPass {
    pub fn new() -> Self {
        Self {
            alpha:        0.1,
            pipeline:     None,
            frame_index:  0,
            depth_buffer: ImageHandle::INVALID,
            ao_raw:       ImageHandle::INVALID,
            ao_history:   ImageHandle::INVALID,
            ao_resolved:  ImageHandle::INVALID,
        }
    }

    pub fn tick(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}

impl RenderPass for GtaoTemporalPass {
    fn name(&self) -> &str {
        "GtaoTemporalPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("gtao_temporal")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.depth_buffer = builder.resolve_image("DepthBuffer");
        self.ao_raw       = builder.resolve_image("AO_Raw");

        let raw_desc  = builder.get_image_desc(self.ao_raw);
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
        builder.read_image(self.ao_raw,       ResourceUsage::SHADER_READ);
        builder.read_image(self.ao_history,   ResourceUsage::SHADER_READ);
        builder.write_image(self.ao_resolved, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else { return; };

        let common         = frame.consume::<crate::render_graph::CommonData>("Common");
        let prev_vp        = frame.consume::<nalgebra_glm::Mat4>("PrevViewProjection");
        let width          = common.screen_width;
        let height         = common.screen_height;

        ctx.bind_pipeline_raw(pipeline);

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);

        let inv_vp: [[f32; 4]; 4] = *common.inv_view_projection.as_ref();
        let prev:   [[f32; 4]; 4] = *prev_vp.as_ref();

        let pc = TemporalPushConstants {
            inv_view_proj:  inv_vp,
            prev_view_proj: prev,
            screen_size:    [width as f32, height as f32],
            alpha:          self.alpha,
            frame_index:    self.frame_index,
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
                    self.ao_raw,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::sampled_image(
                    2, 0,
                    self.ao_history,
                    DescriptorImageLayout::ShaderReadOnlyOptimal,
                ),
                DescriptorWrite::storage_image(
                    3, 0,
                    self.ao_resolved,
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
