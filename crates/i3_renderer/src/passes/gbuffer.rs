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

/// GBuffer pass struct implementing the RenderPass trait.
pub struct GBufferPass {
    pub pipeline: PipelineHandle,
    pub depth_buffer: ImageHandle,
    pub gbuffer_albedo: ImageHandle,
    pub gbuffer_normal: ImageHandle,
    pub gbuffer_roughmetal: ImageHandle,
    pub gbuffer_emissive: ImageHandle,
    pub draw_commands: Vec<DrawCommand>,
}

impl GBufferPass {
    pub fn new(
        pipeline: PipelineHandle,
        depth_buffer: ImageHandle,
        gbuffer_albedo: ImageHandle,
        gbuffer_normal: ImageHandle,
        gbuffer_roughmetal: ImageHandle,
        gbuffer_emissive: ImageHandle,
        draw_commands: Vec<DrawCommand>,
    ) -> Self {
        Self {
            pipeline,
            depth_buffer,
            gbuffer_albedo,
            gbuffer_normal,
            gbuffer_roughmetal,
            gbuffer_emissive,
            draw_commands,
        }
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &str {
        "GBufferPass"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.bind_pipeline(self.pipeline);

        // Declare write targets
        builder.write_image(self.gbuffer_albedo, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_normal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_roughmetal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_emissive, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer, ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        for cmd in &self.draw_commands {
            ctx.push_constant_data(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                &cmd.push_constants,
            );

            let vb = BufferHandle(SymbolId(cmd.mesh.vertex_buffer.0));
            let ib = BufferHandle(SymbolId(cmd.mesh.index_buffer.0));

            ctx.bind_vertex_buffer(0, vb);
            ctx.bind_index_buffer(ib, IndexType::Uint16);
            ctx.draw_indexed(cmd.mesh.index_count, 0, 0);
        }
    }
}
