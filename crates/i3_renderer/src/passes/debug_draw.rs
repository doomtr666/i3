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

const MAX_DEBUG_VERTICES: usize = 4_000_000; // 2 M line segments

pub struct DebugDrawPass {
    // Persistent GPU buffer (GpuOnly, STORAGE_BUFFER + TRANSFER_DST)
    vertex_buffer_physical: BackendBuffer,

    // CPU-side staging data — filled by the caller before render_graph.render()
    pub lines: Vec<DebugVertex>,

    // Resolved handles (valid only during declare → execute)
    vertex_buffer: BufferHandle,
    backbuffer:    ImageHandle,
    depth_buffer:  ImageHandle,

    pipeline: Option<BackendPipeline>,
}

impl DebugDrawPass {
    pub fn new() -> Self {
        Self {
            vertex_buffer_physical: BackendBuffer::INVALID,
            lines:          Vec::new(),
            vertex_buffer:  BufferHandle::INVALID,
            backbuffer:     ImageHandle::INVALID,
            depth_buffer:   ImageHandle::INVALID,
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

    /// Draw a decimal integer as 7-segment digits in world space.
    ///
    /// Digits are rendered as line segments in the billboard plane defined by
    /// `cam_right` and `cam_up` (world-space unit vectors extracted from the view matrix).
    /// The label is centred at `pos`.
    ///
    /// `scale` is the height of one digit in world units.
    /// Useful to display instance indices for RenderDoc thread-level debugging.
    pub fn push_label_3d(
        &mut self,
        pos:       [f32; 3],
        n:         u32,
        col:       [f32; 4],
        scale:     f32,
        cam_right: [f32; 3], // first row of view matrix
        cam_up:    [f32; 3], // second row of view matrix
    ) {
        // 7-segment encoding: bits 0-6 = segments a,b,c,d,e,f,g
        //   aaa
        //  f   b
        //  f   b
        //   ggg
        //  e   c
        //  e   c
        //   ddd
        const SEGS: [u8; 10] = [
            0b0111111, // 0  a b c d e f
            0b0000110, // 1  b c
            0b1011011, // 2  a b d e g
            0b1001111, // 3  a b c d g
            0b1100110, // 4  b c f g
            0b1101101, // 5  a c d f g
            0b1111101, // 6  a c d e f g
            0b0000111, // 7  a b c
            0b1111111, // 8  a b c d e f g
            0b1101111, // 9  a b c d f g
        ];

        // Collect decimal digits (most-significant first).
        let mut digits = [0u8; 10];
        let mut ndigits = 0usize;
        let mut x = if n == 0 { 1 } else { n };
        if n == 0 { digits[0] = 0; ndigits = 1; }
        else {
            while x > 0 { digits[ndigits] = (x % 10) as u8; x /= 10; ndigits += 1; }
            digits[..ndigits].reverse();
        }
        let digits = &digits[..ndigits];

        let dw    = scale * 0.6;   // digit width
        let dh    = scale;         // digit height
        let gap   = scale * 0.2;   // inter-digit gap

        let total_w = ndigits as f32 * dw + (ndigits - 1) as f32 * gap;

        // pt(r, u): world position offset by r*dw along cam_right and u*dh along cam_up,
        // relative to the bottom-left corner of the current digit.
        let add3 = |a: [f32;3], b: [f32;3]| -> [f32;3] {
            [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
        };
        let scale3 = |v: [f32;3], s: f32| -> [f32;3] { [v[0]*s, v[1]*s, v[2]*s] };

        // Centre offset: start of first digit (bottom-left) relative to pos.
        let base_r = -total_w * 0.5;
        let base_u = -dh * 0.5;

        for (di, &digit) in digits.iter().enumerate() {
            let col_r = base_r + di as f32 * (dw + gap);

            let pt = |r: f32, u: f32| -> [f32; 3] {
                add3(pos, add3(scale3(cam_right, col_r + r * dw), scale3(cam_up, base_u + u * dh)))
            };

            let tl = pt(0.0, 1.0); let tr = pt(1.0, 1.0);
            let ml = pt(0.0, 0.5); let mr = pt(1.0, 0.5);
            let bl = pt(0.0, 0.0); let br = pt(1.0, 0.0);

            let s = SEGS[digit as usize];
            if s & (1 << 0) != 0 { self.push_line(tl, tr, col); } // a top
            if s & (1 << 1) != 0 { self.push_line(tr, mr, col); } // b top-right
            if s & (1 << 2) != 0 { self.push_line(mr, br, col); } // c bot-right
            if s & (1 << 3) != 0 { self.push_line(bl, br, col); } // d bottom
            if s & (1 << 4) != 0 { self.push_line(ml, bl, col); } // e bot-left
            if s & (1 << 5) != 0 { self.push_line(tl, ml, col); } // f top-left
            if s & (1 << 6) != 0 { self.push_line(ml, mr, col); } // g middle
        }
    }

    /// Draw a small 3D cross at `pos` (6 line segments along world ±X/±Y/±Z).
    /// Useful for visualising point clouds (e.g. dual-contouring vertices).
    /// `half_size` is the half-length of each arm in world units.
    pub fn push_cross(&mut self, pos: [f32; 3], half_size: f32, col: [f32; 4]) {
        let [x, y, z] = pos;
        let h = half_size;
        self.push_line([x - h, y, z], [x + h, y, z], col);
        self.push_line([x, y - h, z], [x, y + h, z], col);
        self.push_line([x, y, z - h], [x, y, z + h], col);
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

        // Read-only depth test against the scene depth buffer (reverse-Z, no write)
        self.depth_buffer = builder.resolve_image("DepthBuffer");
        builder.write_image(self.depth_buffer, ResourceUsage::DEPTH_STENCIL);
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
