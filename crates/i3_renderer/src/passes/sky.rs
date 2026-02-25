use i3_gfx::prelude::*;
use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
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
    pub pipeline: PipelineHandle,
    pub hdr_target: ImageHandle,
    pub depth_buffer: ImageHandle,
    pub push_constants: SkyPushConstants,
}

impl RenderPass for SkyPass {
    fn name(&self) -> &str {
        "SkyPass"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.bind_pipeline(self.pipeline);

        // Clears the HDR target and depth buffer (to 1.0)
        builder.write_image(self.hdr_target, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer, ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        ctx.push_constant_data(
            ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
            0,
            &self.push_constants,
        );
        ctx.draw(3, 0); // Fullscreen triangle
    }
}
