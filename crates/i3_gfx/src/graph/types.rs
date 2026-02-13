use bitflags::bitflags;

/// Opaque handle for a logical image within a single frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageHandle(pub u64);

/// Opaque handle for a logical buffer within a single frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u64);

impl ImageHandle {
    pub const INVALID: Self = Self(0);
    pub fn new(val: u64) -> Self {
        Self(val)
    }
}

impl BufferHandle {
    pub const INVALID: Self = Self(0);
    pub fn new(val: u64) -> Self {
        Self(val)
    }
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
