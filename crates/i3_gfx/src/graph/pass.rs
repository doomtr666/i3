use crate::graph::types::{BufferHandle, ImageHandle, PassDomain, ResourceUsage};

/// Context provided to a pass during its declaration phase.
pub trait PassBuilder {
    fn read_image(&mut self, handle: ImageHandle, usage: ResourceUsage);
    fn write_image(&mut self, handle: ImageHandle, usage: ResourceUsage);

    fn read_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage);
    fn write_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage);

    // Add more specialized methods as needed (e.g., set_inline)
}

/// Interface for implementing a render pass in the frame graph.
pub trait RenderPass {
    /// Human-readable name for debugging and profiling labels.
    fn name(&self) -> &str;

    /// Target queue/domain for this pass.
    fn domain(&self) -> PassDomain;

    /// Declaration phase: tell the graph which resources this pass reads from or writes to.
    fn declare(&self, builder: &mut dyn PassBuilder);

    // Execution phase: recorded on a command buffer.
    // (Target implementation depends on PassContext, added in Phase 3/4)
}
