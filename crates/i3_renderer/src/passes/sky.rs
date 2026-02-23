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
}

pub fn record_sky_pass(
    builder: &mut PassBuilder,
    pipeline: PipelineHandle,
    hdr_target: ImageHandle,
    depth_buffer: ImageHandle,
    push_constants: &SkyPushConstants,
) {
    let pc = *push_constants;
    builder.add_node("SkyPass", move |builder| {
        builder.bind_pipeline(pipeline);

        // Clears the HDR target and depth buffer (to 1.0)
        builder.write_image(hdr_target, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(depth_buffer, ResourceUsage::DEPTH_STENCIL);

        move |ctx: &mut dyn PassContext| {
            let pc_bytes = unsafe {
                std::slice::from_raw_parts(
                    &pc as *const _ as *const u8,
                    std::mem::size_of::<SkyPushConstants>(),
                )
            };
            ctx.push_constants(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                pc_bytes,
            );
            ctx.draw(3, 0); // Fullscreen triangle
        }
    });
}
