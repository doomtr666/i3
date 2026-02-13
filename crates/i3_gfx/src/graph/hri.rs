// use crate::graph::types::{ImageHandle, BufferHandle, ResourceUsage};

/// Handle representing a physically allocated image in the HRI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HriImage(pub u64);

/// Handle representing a physically allocated buffer in the HRI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HriBuffer(pub u64);

/// Hardware-specific context used to record commands during a pass.
pub trait PassContext {
    // Pipeline & Binding (Agnostic interfaces)
    // fn bind_resources(&mut self, ...);

    // Commands
    fn draw(&mut self, vertex_count: u32, first_vertex: u32);
    fn dispatch(&mut self, x: u32, y: u32, z: u32);
}

/// The main interface for hardware backends (Vulkan, DX12, Null).
pub trait HriBackend {
    // Resource Management
    // fn create_texture(&mut self, desc: &ResourceDesc) -> HriTexture;

    // Execution
    // fn submit(&mut self, batches: &[BarrierBatch]) -> PendingSubmission;
}
