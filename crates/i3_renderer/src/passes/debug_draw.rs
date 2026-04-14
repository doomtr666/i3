//! Debug line draw pass.
//!
//! CPU fills a Vec<DebugVertex> via push_line / push_aabb / push_frustum before
//! calling render_graph.render(). The pass copies the data to the GPU via a
//! transient staging buffer and draws it as LINE_LIST on top of the backbuffer.
//!
//! When no lines are queued the pass is a no-op (declare() returns early, the
//! frame graph skips it entirely).

use bytemuck::{Pod, Zeroable};
use i3_gfx::graph::backend::{BackendBuffer, BackendPipeline, PassContext, RenderBackend};
use i3_gfx::graph::compiler::FrameBlackboard;
use i3_gfx::graph::pass::{PassBuilder, RenderPass};
use i3_gfx::graph::pipeline::ShaderStageFlags;
use i3_gfx::graph::types::{
    BufferDesc, BufferHandle, BufferUsageFlags, ImageHandle, MemoryType, ResourceUsage,
};
use std::sync::Arc;

// ─── Vertex format ───────────────────────────────────────────────────────────

/// One endpoint of a debug line.  Must match the Slang `DebugVertex` struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct DebugVertex {
    pub pos: [f32; 3],
    pub _pad: f32,
    pub col: [f32; 4], // linear RGBA
}

// ─── Push constants ──────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DebugDrawPushConstants {
    view_proj: [[f32; 4]; 4],
}

// ─── Pass ────────────────────────────────────────────────────────────────────

const MAX_DEBUG_VERTICES: usize = 200_000; // 100 K line segments

pub struct DebugDrawPass {
    // Persistent GPU buffer (GpuOnly, STORAGE_BUFFER + TRANSFER_DST)
    vertex_buffer_physical: BackendBuffer,

    // CPU-side staging data — filled by the caller before render_graph.render()
    pub lines: Vec<DebugVertex>,

    // Resolved handles (valid only during declare → execute)
    vertex_buffer: BufferHandle,
    backbuffer:    ImageHandle,

    pipeline: Option<BackendPipeline>,
}

impl DebugDrawPass {
    pub fn new() -> Self {
        Self {
            vertex_buffer_physical: BackendBuffer::INVALID,
            lines:          Vec::new(),
            vertex_buffer:  BufferHandle::INVALID,
            backbuffer:     ImageHandle::INVALID,
            pipeline:       None,
        }
    }

    // ── Public drawing API ────────────────────────────────────────────────

    pub fn clear(&mut self) {
        self.lines.clear();
    }

    pub fn push_line(&mut self, a: [f32; 3], b: [f32; 3], col: [f32; 4]) {
        if self.lines.len() + 2 > MAX_DEBUG_VERTICES { return; }
        let zero = DebugVertex { pos: a, _pad: 0.0, col };
        self.lines.push(zero);
        let zero = DebugVertex { pos: b, _pad: 0.0, col };
        self.lines.push(zero);
    }

    /// Draw a wireframe AABB (12 edges = 24 vertices).
    pub fn push_aabb(&mut self, min: [f32; 3], max: [f32; 3], col: [f32; 4]) {
        let c = [
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [min[0], max[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [min[0], max[1], max[2]],
            [max[0], max[1], max[2]],
        ];
        // bottom face
        self.push_line(c[0], c[1], col); self.push_line(c[1], c[3], col);
        self.push_line(c[3], c[2], col); self.push_line(c[2], c[0], col);
        // top face
        self.push_line(c[4], c[5], col); self.push_line(c[5], c[7], col);
        self.push_line(c[7], c[6], col); self.push_line(c[6], c[4], col);
        // vertical edges
        self.push_line(c[0], c[4], col); self.push_line(c[1], c[5], col);
        self.push_line(c[2], c[6], col); self.push_line(c[3], c[7], col);
    }

    /// Draw the view frustum defined by `inv_vp` in world space (12 edges).
    /// Useful when the culling camera is decoupled from the render camera.
    pub fn push_frustum(&mut self, inv_vp: &nalgebra_glm::Mat4, col: [f32; 4]) {
        // 8 NDC corners (reverse-Z: near = 1.0, far = 0.0)
        let ndc: [[f32; 3]; 8] = [
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
            [-1.0,  1.0, 1.0], [1.0,  1.0, 1.0],
            [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
            [-1.0,  1.0, 0.0], [1.0,  1.0, 0.0],
        ];
        let corners: Vec<[f32; 3]> = ndc.iter().map(|&[x, y, z]| {
            let h = inv_vp * nalgebra_glm::vec4(x, y, z, 1.0);
            let p = h.xyz() / h.w;
            [p.x, p.y, p.z]
        }).collect();

        // near face
        self.push_line(corners[0], corners[1], col); self.push_line(corners[1], corners[3], col);
        self.push_line(corners[3], corners[2], col); self.push_line(corners[2], corners[0], col);
        // far face
        self.push_line(corners[4], corners[5], col); self.push_line(corners[5], corners[7], col);
        self.push_line(corners[7], corners[6], col); self.push_line(corners[6], corners[4], col);
        // lateral edges
        self.push_line(corners[0], corners[4], col); self.push_line(corners[1], corners[5], col);
        self.push_line(corners[2], corners[6], col); self.push_line(corners[3], corners[7], col);
    }
}

// ─── RenderPass impl ─────────────────────────────────────────────────────────

impl RenderPass for DebugDrawPass {
    fn name(&self) -> &str { "DebugDraw" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        // CpuToGpu vertex buffer — host-visible, written by CPU each frame via map_buffer.
        // No staging copy needed, which avoids vkCmdCopyBuffer inside an active render pass.
        self.vertex_buffer_physical = backend.create_buffer(&BufferDesc {
            size: (MAX_DEBUG_VERTICES * std::mem::size_of::<DebugVertex>()) as u64,
            usage: BufferUsageFlags::STORAGE_BUFFER,
            memory: MemoryType::CpuToGpu,
        });
        #[cfg(debug_assertions)]
        backend.set_buffer_name(self.vertex_buffer_physical, "DebugVertexBuffer");

        // Load pipeline from baked asset
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("debug_draw")
            .wait_loaded()
        {
            let state = asset.state.as_ref().expect("debug_draw asset missing pipeline state");
            self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                state,
                &asset.reflection_data,
                &asset.bytecode,
            ));
        } else {
            tracing::warn!("DebugDrawPass: 'debug_draw' pipeline asset not found — run the baker.");
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // No lines → completely skip this pass (frame graph ignores it)
        if self.lines.is_empty() {
            return;
        }

        // Import the CpuToGpu vertex buffer — written by CPU via map_buffer in execute().
        // No TRANSFER_WRITE declared; host writes to HOST_VISIBLE memory are CPU operations
        // and are valid even inside an active render pass.
        self.vertex_buffer = builder.import_buffer(
            "DebugVertexBuffer",
            self.vertex_buffer_physical,
        );

        // Expose vertex buffer to the draw shader as a storage buffer (set 0, binding 0)
        builder.descriptor_set(0, |d| {
            d.storage_buffer(self.vertex_buffer);
        });

        // Render to the backbuffer (after all post-processing)
        self.backbuffer = builder.resolve_image("Backbuffer");
        builder.write_image(self.backbuffer, ResourceUsage::COLOR_ATTACHMENT);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            return;
        };
        if self.lines.is_empty() {
            return;
        }

        // ── 1. Upload vertex data (host write — no GPU command, valid inside render pass) ──
        let data_size = self.lines.len() * std::mem::size_of::<DebugVertex>();
        let ptr = ctx.map_buffer(self.vertex_buffer);
        if ptr.is_null() {
            tracing::error!("DebugDrawPass: failed to map vertex buffer");
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.lines.as_ptr() as *const u8,
                ptr,
                data_size,
            );
        }
        ctx.unmap_buffer(self.vertex_buffer);

        // ── 2. Draw ──────────────────────────────────────────────────────
        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let vp = common.view_projection;
        // Convert nalgebra Mat4 (column-major) to [[f32;4];4] row-major for push constant
        let pc = DebugDrawPushConstants {
            view_proj: vp.into(),
        };

        ctx.bind_pipeline_raw(pipeline);
        ctx.push_bytes(ShaderStageFlags::Vertex | ShaderStageFlags::Fragment, 0, bytemuck::bytes_of(&pc));
        ctx.draw(self.lines.len() as u32, 0);
    }
}
