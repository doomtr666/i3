use i3_gfx::prelude::*;
use std::sync::Arc;

use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SkyPushConstants {
    pub inv_view_proj: glm::Mat4,
    pub camera_pos: glm::Vec3,
    pub _pad0: f32,
    pub sun_direction: glm::Vec3,
    pub sun_intensity: f32,
    pub sun_color: glm::Vec3,
    pub ibl_env_index: u32, // bindless index, !0 = no IBL
}

pub struct SkyPass {
    // Resolved handles (updated in declare)
    hdr_target: ImageHandle,
    depth_buffer: ImageHandle,

    pipeline: Option<BackendPipeline>,
}

impl SkyPass {
    pub fn new() -> Self {
        Self {
            hdr_target: ImageHandle::INVALID,
            depth_buffer: ImageHandle::INVALID,
            pipeline: None,
        }
    }
}

impl RenderPass for SkyPass {
    fn name(&self) -> &str {
        "SkyPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("sky")
            .wait_loaded()
        {
            let state = asset.state.as_ref().expect("Sky asset missing state");
            self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                state,
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        let (w, h) = (common.screen_width, common.screen_height);
        self.hdr_target = builder.declare_image_output(
            "HDR_Target",
            ImageDesc::new(w, h, Format::R16G16B16A16_SFLOAT),
        );
        self.depth_buffer = builder.resolve_image("DepthBuffer");
        builder.write_image(self.hdr_target, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer, ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("SkyPass::execute: pipeline not initialized!");
            return;
        };
        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let sun_dir = *frame.consume::<glm::Vec3>("SunDirection");
        let sun_int = *frame.consume::<f32>("SunIntensity");
        let sun_col = *frame.consume::<glm::Vec3>("SunColor");
        let ibl = *frame.consume::<crate::render_graph::IblIndices>("IblIndices");
        let push_constants = SkyPushConstants {
            inv_view_proj: common.inv_view_projection,
            camera_pos: common.camera_pos,
            _pad0: 0.0,
            sun_direction: sun_dir,
            sun_intensity: sun_int,
            sun_color: sun_col,
            ibl_env_index: ibl.env_index,
        };

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");

        ctx.bind_pipeline_raw(pipeline);
        ctx.bind_descriptor_set(2, bindless_set);
        ctx.push_constant_data(
            ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
            0,
            &push_constants,
        );
        ctx.draw(3, 0); // Fullscreen triangle
    }
}
