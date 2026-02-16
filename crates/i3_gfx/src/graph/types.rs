use bitflags::bitflags;
use std::any::TypeId;

/// Unique identifier for any entry in the Scoped Symbol Table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub u64);

impl SymbolId {
    pub const INVALID: Self = Self(0);
}

/// Handle for a virtual image resource, backed by a Symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageHandle(pub SymbolId);

/// Handle for a virtual buffer resource, backed by a Symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub SymbolId);

/// Handle for a Window managed by the FrameGraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowHandle(pub u64);

/// Type-safe wrapper for an image acquired from a swapchain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwapChainImageHandle(pub ImageHandle);

/// Handle for a compiled Graphics or Compute pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineHandle(pub SymbolId);

impl std::ops::Deref for SwapChainImageHandle {
    type Target = ImageHandle;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The underlying type of a Symbol in the table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolType {
    Image(ImageDesc),
    Buffer(BufferDesc),
    CpuData(TypeId),
}

/// Defines the hardware or logical lifetime of a Symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolLifetime {
    /// Temporary within its scope (eligible for aliasing).
    Transient,
    /// Persists across frames (owned by the graph).
    Persistent,
    /// Injected from external system (e.g., Swapchain, Global Settings).
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum Format {
    R8G8B8A8_UNORM,
    B8G8R8A8_UNORM,
    B8G8R8A8_SRGB,
    R32_FLOAT,
    R32G32B32A32_FLOAT,
    D32_FLOAT,
    // Add others as needed
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageDesc {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: Format,
    pub mip_levels: u32,
    pub array_layers: u32,
}

impl ImageDesc {
    pub fn new(width: u32, height: u32, format: Format) -> Self {
        Self {
            width,
            height,
            depth: 1,
            format,
            mip_levels: 1,
            array_layers: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferDesc {
    pub size: u64,
}

impl ImageHandle {
    pub const INVALID: Self = Self(SymbolId::INVALID);
}

impl BufferHandle {
    pub const INVALID: Self = Self(SymbolId::INVALID);
}

impl PipelineHandle {
    pub const INVALID: Self = Self(SymbolId::INVALID);
}

/// Execution domain for a render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassDomain {
    Graphics,
    Compute,
    Transfer,
    Cpu,
}

bitflags! {
    /// Bitfield representing how a resource is used within a pass.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct ResourceUsage: u32 {
        const NONE = 0;
        const READ = 1 << 0;
        const WRITE = 1 << 1;

        // Specialized usages for barrier generation
        const COLOR_ATTACHMENT = 1 << 2;
        const DEPTH_STENCIL = 1 << 3;
        const SHADER_READ = 1 << 4;
        const SHADER_WRITE = 1 << 5;
        const TRANSFER_READ = 1 << 6;
        const TRANSFER_WRITE = 1 << 7;
    }
}

#[cfg(test)]
#[path = "types.tests.rs"]
mod tests;
