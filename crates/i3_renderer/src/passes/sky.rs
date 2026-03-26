use std::sync::Arc;
use i3_gfx::prelude::*;

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
    pub _pad1: f32,
}

/// Sky pass struct implementing the RenderPass trait.
pub struct SkyPass {
    // Resolved handles (updated in record)
    hdr_target: ImageHandle,
    depth_buffer: ImageHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
    push_constants: Option<SkyPushConstants>,
}

impl SkyPass {
    pub fn new() -> Self {
        Self {
            hdr_target: ImageHandle::INVALID,
            depth_buffer: ImageHandle::INVALID,
            pipeline: None,
            push_constants: None,
        }
    }

}

impl RenderPass for SkyPass {
    fn name(&self) -> &str {
        "SkyPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("sky").wait_loaded() {
            let state = asset.state.as_ref().expect("Sky asset missing state");
            self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                state,
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        if builder.is_setup() {
            return;
        }
        // Resolve target handles by name
        self.hdr_target = builder.resolve_image("HDR_Target");
        self.depth_buffer = builder.resolve_image("DepthBuffer");

        // Compute push constants from blackboard
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        let sun_dir = *builder.consume::<glm::Vec3>("SunDirection");
        let sun_int = *builder.consume::<f32>("SunIntensity");
        let sun_col = *builder.consume::<glm::Vec3>("SunColor");

        self.push_constants = Some(SkyPushConstants {
            inv_view_proj: common.inv_view_projection,
            camera_pos: common.camera_pos,
            _pad0: 0.0,
            sun_direction: sun_dir,
            sun_intensity: sun_int,
            sun_color: sun_col,
            _pad1: 0.0,
        });

        // Clears the HDR target and depth buffer (to 1.0)
        builder.write_image(self.hdr_target, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer, ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("SkyPass::execute: pipeline not initialized!");
            return;
        };
        ctx.bind_pipeline_raw(pipeline);

        if let Some(constants) = self.push_constants {
            ctx.push_constant_data(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                &constants,
            );
            ctx.draw(3, 0); // Fullscreen triangle
        }
    }
}
