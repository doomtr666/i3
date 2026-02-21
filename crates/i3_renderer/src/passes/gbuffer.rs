use i3_gfx::prelude::*;

/// Hardcoded cube vertex: position + normal.
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
/// Declares 4 color targets + depth, draws hardcoded cubes using push constants
/// for transform data.
pub fn record_gbuffer_pass(
    builder: &mut PassBuilder,
    pipeline: PipelineHandle,
    vertex_buffer: BackendBuffer,
    index_buffer: BackendBuffer,
    depth_buffer: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive: ImageHandle,
    push_constants: GBufferPushConstants,
) {
    builder.add_node("GBufferPass", move |sub| {
        sub.bind_pipeline(pipeline);

        // Declare write targets
        sub.write_image(gbuffer_albedo, ResourceUsage::COLOR_ATTACHMENT);
        sub.write_image(gbuffer_normal, ResourceUsage::COLOR_ATTACHMENT);
        sub.write_image(gbuffer_roughmetal, ResourceUsage::COLOR_ATTACHMENT);
        sub.write_image(gbuffer_emissive, ResourceUsage::COLOR_ATTACHMENT);
        sub.write_image(depth_buffer, ResourceUsage::DEPTH_STENCIL);

        let vb = BufferHandle(SymbolId(vertex_buffer.0));
        let ib = BufferHandle(SymbolId(index_buffer.0));

        move |ctx: &mut dyn PassContext| {
            // Push MVP
            let pc_bytes = unsafe {
                std::slice::from_raw_parts(
                    &push_constants as *const GBufferPushConstants as *const u8,
                    std::mem::size_of::<GBufferPushConstants>(),
                )
            };
            ctx.push_constants(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                pc_bytes,
            );

            ctx.bind_vertex_buffer(0, vb);
            ctx.bind_index_buffer(ib, IndexType::Uint16);

            // 36 indices = 12 triangles = 1 cube
            ctx.draw_indexed(36, 0, 0);
        }
    });
}

/// Generates vertices and indices for a unit cube centered at origin.
/// Returns (vertices, indices).
pub fn generate_cube() -> (Vec<GBufferVertex>, Vec<u16>) {
    // 24 vertices (4 per face, for unique normals)
    #[rustfmt::skip]
    let vertices = vec![
        // Front face (Z+) — normal (0,0,1)
        GBufferVertex { position: [-0.5, -0.5,  0.5], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },
        GBufferVertex { position: [ 0.5, -0.5,  0.5], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },
        GBufferVertex { position: [ 0.5,  0.5,  0.5], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },
        GBufferVertex { position: [-0.5,  0.5,  0.5], normal: [0.0, 0.0, 1.0], color: [1.0, 0.0, 0.0] },
        // Back face (Z-) — normal (0,0,-1)
        GBufferVertex { position: [ 0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], color: [0.0, 1.0, 0.0] },
        GBufferVertex { position: [-0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], color: [0.0, 1.0, 0.0] },
        GBufferVertex { position: [-0.5,  0.5, -0.5], normal: [0.0, 0.0, -1.0], color: [0.0, 1.0, 0.0] },
        GBufferVertex { position: [ 0.5,  0.5, -0.5], normal: [0.0, 0.0, -1.0], color: [0.0, 1.0, 0.0] },
        // Top face (Y+) — normal (0,1,0)
        GBufferVertex { position: [-0.5,  0.5,  0.5], normal: [0.0, 1.0, 0.0], color: [0.0, 0.0, 1.0] },
        GBufferVertex { position: [ 0.5,  0.5,  0.5], normal: [0.0, 1.0, 0.0], color: [0.0, 0.0, 1.0] },
        GBufferVertex { position: [ 0.5,  0.5, -0.5], normal: [0.0, 1.0, 0.0], color: [0.0, 0.0, 1.0] },
        GBufferVertex { position: [-0.5,  0.5, -0.5], normal: [0.0, 1.0, 0.0], color: [0.0, 0.0, 1.0] },
        // Bottom face (Y-) — normal (0,-1,0)
        GBufferVertex { position: [-0.5, -0.5, -0.5], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 0.0] },
        GBufferVertex { position: [ 0.5, -0.5, -0.5], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 0.0] },
        GBufferVertex { position: [ 0.5, -0.5,  0.5], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 0.0] },
        GBufferVertex { position: [-0.5, -0.5,  0.5], normal: [0.0, -1.0, 0.0], color: [1.0, 1.0, 0.0] },
        // Right face (X+) — normal (1,0,0)
        GBufferVertex { position: [ 0.5, -0.5,  0.5], normal: [1.0, 0.0, 0.0], color: [1.0, 0.0, 1.0] },
        GBufferVertex { position: [ 0.5, -0.5, -0.5], normal: [1.0, 0.0, 0.0], color: [1.0, 0.0, 1.0] },
        GBufferVertex { position: [ 0.5,  0.5, -0.5], normal: [1.0, 0.0, 0.0], color: [1.0, 0.0, 1.0] },
        GBufferVertex { position: [ 0.5,  0.5,  0.5], normal: [1.0, 0.0, 0.0], color: [1.0, 0.0, 1.0] },
        // Left face (X-) — normal (-1,0,0)
        GBufferVertex { position: [-0.5, -0.5, -0.5], normal: [-1.0, 0.0, 0.0], color: [0.0, 1.0, 1.0] },
        GBufferVertex { position: [-0.5, -0.5,  0.5], normal: [-1.0, 0.0, 0.0], color: [0.0, 1.0, 1.0] },
        GBufferVertex { position: [-0.5,  0.5,  0.5], normal: [-1.0, 0.0, 0.0], color: [0.0, 1.0, 1.0] },
        GBufferVertex { position: [-0.5,  0.5, -0.5], normal: [-1.0, 0.0, 0.0], color: [0.0, 1.0, 1.0] },
    ];

    #[rustfmt::skip]
    let indices: Vec<u16> = vec![
         0,  2,  1,  0,  3,  2, // Front
         4,  6,  5,  4,  7,  6, // Back
         8, 10,  9,  8, 11, 10, // Top
        12, 14, 13, 12, 15, 14, // Bottom
        16, 18, 17, 16, 19, 18, // Right
        20, 22, 21, 20, 23, 22, // Left
    ];

    (vertices, indices)
}
