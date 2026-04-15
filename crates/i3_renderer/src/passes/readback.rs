//! Generic GPU→CPU buffer readback pass.
//!
//! `BufferReadbackPass` records a `vkCmdCopyBuffer` from a named frame-graph buffer
//! to a host-visible (GpuToCpu) staging buffer. The frame graph's sync planner
//! automatically inserts the required barriers (e.g. SHADER_WRITE → TRANSFER_READ).
//!
//! # Usage
//! 1. Create a `GpuToCpu` staging buffer with `TRANSFER_DST` usage.
//! 2. Construct a `BufferReadbackPass::new(src_name, staging, size)` and add it
//!    to the render graph after the pass that writes the source buffer.
//! 3. After each `render()` call, call `try_download` on `DefaultRenderGraph`
//!    (non-blocking poll via `wait_for_timeline(T, 0)`).
//!
//! # Timing / safety
//! The copy is part of the frame's command buffer.  The render graph stores the
//! timeline semaphore value after each `execute()`.  `try_download` polls that
//! value without stalling: if the GPU hasn't finished yet the read is skipped and
//! the caller keeps the data from the previous frame.

use i3_gfx::graph::backend::{BackendBuffer, PassContext};
use i3_gfx::graph::compiler::FrameBlackboard;
use i3_gfx::graph::pass::{PassBuilder, RenderPass};
use i3_gfx::graph::types::{BufferHandle, ResourceUsage};

/// Copies a named frame-graph buffer to a host-visible staging buffer each frame.
///
/// One instance per source buffer.  The staging buffer is created externally
/// (by whoever owns the readback data) and passed in — this pass just records
/// the copy command.
pub struct BufferReadbackPass {
    /// Name of the source buffer in the frame graph (e.g. `"VisibilityBitset"`).
    pub src_name: &'static str,
    /// GpuToCpu staging buffer (TRANSFER_DST).  Owned by the caller.
    pub staging:  BackendBuffer,
    size:         u64,
    src_handle:   BufferHandle,
    dst_handle:   BufferHandle,
}

impl BufferReadbackPass {
    pub fn new(src_name: &'static str, staging: BackendBuffer, size: u64) -> Self {
        Self {
            src_name,
            staging,
            size,
            src_handle: BufferHandle::INVALID,
            dst_handle: BufferHandle::INVALID,
        }
    }
}

impl RenderPass for BufferReadbackPass {
    fn name(&self) -> &str { "BufferReadback" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.src_handle = builder.resolve_buffer(self.src_name);
        builder.read_buffer(self.src_handle, ResourceUsage::TRANSFER_READ);

        // Import the staging buffer under a unique name so the frame graph tracks it.
        let import_name = format!("Readback_{}", self.src_name);
        self.dst_handle = builder.import_buffer(&import_name, self.staging);
        builder.write_buffer(self.dst_handle, ResourceUsage::TRANSFER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        ctx.copy_buffer(self.src_handle, self.dst_handle, 0, 0, self.size);
    }
}
