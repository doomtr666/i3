use i3_gfx::prelude::*;

use crate::render_graph::DrawCommand;

/// GBuffer vertex layout: position + normal + color.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GBufferVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

/// Push constants for the GBuffer pass.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GBufferPushConstants {
    pub view_projection: nalgebra_glm::Mat4,
    pub model: nalgebra_glm::Mat4,
}

/// Records the GBuffer pass into the FrameGraph.
///
/// Writes to 4 color targets + depth, draws all objects from the draw command list.
pub fn record_gbuffer_pass(
    builder: &mut PassBuilder,
    pipeline: PipelineHandle,
    depth_buffer: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive: ImageHandle,
    draw_commands: &[DrawCommand],
) {
    // Clone draw commands so the closure owns them
    let commands: Vec<DrawCommand> = draw_commands.to_vec();

    builder.add_node("GBufferPass", move |sub| {
        sub.bind_pipeline(pipeline);

        // Declare write targets
        sub.write_image(gbuffer_albedo, ResourceUsage::COLOR_ATTACHMENT);
        sub.write_image(gbuffer_normal, ResourceUsage::COLOR_ATTACHMENT);
        sub.write_image(gbuffer_roughmetal, ResourceUsage::COLOR_ATTACHMENT);
        sub.write_image(gbuffer_emissive, ResourceUsage::COLOR_ATTACHMENT);
        sub.write_image(depth_buffer, ResourceUsage::DEPTH_STENCIL);

        move |ctx: &mut dyn PassContext| {
            for cmd in &commands {
                let pc_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &cmd.push_constants as *const GBufferPushConstants as *const u8,
                        std::mem::size_of::<GBufferPushConstants>(),
                    )
                };
                ctx.push_constants(
                    ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                    0,
                    pc_bytes,
                );

                let vb = BufferHandle(SymbolId(cmd.mesh.vertex_buffer.0));
                let ib = BufferHandle(SymbolId(cmd.mesh.index_buffer.0));

                ctx.bind_vertex_buffer(0, vb);
                ctx.bind_index_buffer(ib, IndexType::Uint16);
                ctx.draw_indexed(cmd.mesh.index_count, 0, 0);
            }
        }
    });
}
