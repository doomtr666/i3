use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum GraphError {
    #[error("Symbol '{0}' not found in current or parent scope")]
    SymbolNotFound(String),

    #[error("Type mismatch for symbol '{0}'")]
    TypeMismatch(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Window minimized")]
    WindowMinimized,

    #[error("Validation error: {0}")]
    ValidationError(String),
}

/// Unique identifier for any entry in the Scoped Symbol Table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub u64);

impl SymbolId {
    pub const INVALID: Self = Self(0);
}

impl std::fmt::Display for SymbolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
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
    /// Manages an N and N-1 double buffering state for temporal algorithms.
    TemporalHistory,
    /// Injected from external system (e.g., Swapchain, Global Settings).
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum Format {
    Undefined,
    R8G8B8A8_UNORM,
    R8G8B8A8_SRGB,
    B8G8R8A8_UNORM,
    B8G8R8A8_SRGB,
    R8G8_UNORM,
    R16G16_SFLOAT,
    R16G16B16A16_SFLOAT,
    R11G11B10_UFLOAT,
    R32_FLOAT,
    R32G32B32A32_FLOAT,
    D32_FLOAT,
    R32_SINT,
    R32G32_SINT,
    R32G32B32_SINT,
    R32G32B32A32_SINT,
    R32_UINT,
    R32G32_UINT,
    R32G32B32_UINT,
    R32G32B32A32_UINT,
    R32_SFLOAT,
    R32G32_SFLOAT,
    R32G32B32_SFLOAT,
    R32G32B32A32_SFLOAT,
    // BCx Compressed formats
    BC1_RGB_UNORM,
    BC1_RGB_SRGB,
    BC1_RGBA_UNORM,
    BC1_RGBA_SRGB,
    BC3_UNORM,
    BC3_SRGB,
    BC4_UNORM,
    BC4_SNORM,
    BC5_UNORM,
    BC5_SNORM,
    BC7_UNORM,
    BC7_SRGB,
}

impl Format {
    pub fn is_depth(&self) -> bool {
        matches!(self, Format::D32_FLOAT)
    }

    pub fn is_srgb(&self) -> bool {
        matches!(
            self,
            Format::R8G8B8A8_SRGB
                | Format::B8G8R8A8_SRGB
                | Format::BC1_RGB_SRGB
                | Format::BC1_RGBA_SRGB
                | Format::BC3_SRGB
                | Format::BC7_SRGB
        )
    }

    pub fn aspect_mask(&self) -> ImageAspectFlags {
        if self.is_depth() {
            ImageAspectFlags::Depth
        } else {
            ImageAspectFlags::Color
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageType {
    Type1D,
    Type2D,
    Type3D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImageViewType {
    Type1D,
    Type2D,
    Type3D,
    TypeCube,
    Type1DArray,
    Type2DArray,
    TypeCubeArray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentSwizzle {
    Identity,
    Zero,
    One,
    R,
    G,
    B,
    A,
}

impl Default for ComponentSwizzle {
    fn default() -> Self {
        Self::Identity
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct ComponentMapping {
    pub r: ComponentSwizzle,
    pub g: ComponentSwizzle,
    pub b: ComponentSwizzle,
    pub a: ComponentSwizzle,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ImageUsageFlags: u32 {
        const TRANSFER_SRC = 0x1;
        const TRANSFER_DST = 0x2;
        const SAMPLED = 0x4;
        const STORAGE = 0x8;
        const COLOR_ATTACHMENT = 0x10;
        const DEPTH_STENCIL_ATTACHMENT = 0x20;
        const TRANSIENT_ATTACHMENT = 0x40;
        const INPUT_ATTACHMENT = 0x80;
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ImageAspectFlags: u32 {
        const Color = 0x1;
        const Depth = 0x2;
        const Stencil = 0x4;
        const Metadata = 0x8;
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BufferUsageFlags: u32 {
        const TRANSFER_SRC = 0x1;
        const TRANSFER_DST = 0x2;
        const UNIFORM_TEXEL_BUFFER = 0x4;
        const STORAGE_TEXEL_BUFFER = 0x8;
        const UNIFORM_BUFFER = 0x10;
        const STORAGE_BUFFER = 0x20;
        const INDEX_BUFFER = 0x40;
        const VERTEX_BUFFER = 0x80;
        const INDIRECT_BUFFER = 0x100;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageDesc {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: Format,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub usage: ImageUsageFlags,
    pub view_type: ImageViewType,
    pub swizzle: ComponentMapping,
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
            usage: ImageUsageFlags::SAMPLED
                | ImageUsageFlags::TRANSFER_DST
                | ImageUsageFlags::COLOR_ATTACHMENT, // Default usage
            view_type: ImageViewType::Type2D,
            swizzle: ComponentMapping::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    GpuOnly,
    CpuToGpu, // Host visible, coherent
    GpuToCpu, // Host visible, coherent
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: BufferUsageFlags,
    pub memory: MemoryType,
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

pub use crate::graph::pipeline::SampleCount; // Re-export if needed or redefine? 
// SampleCount is in pipeline.rs. I should probably move it here if it's used in ImageDesc?
// It's not in ImageDesc currently.
// types.rs had SampleCount commented out/missing in my view?
// No, I will import it in convert.rs from pipeline.rs.
// But wait, convert.rs imports `SampleCount` from `types`?
// In Step 7864: `use i3_gfx::graph::types::{..., SampleCount, ...};`
// So I MUST put SampleCount here or change convert.rs.
// It is better placed in pipeline.rs (MultisampleState).
// I will change verify convert.rs imports later.

// Added missing Enums for Sampler/ImageView
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
    MirrorClampToEdge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MipmapMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerDesc {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_mode: MipmapMode,
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mipmap_mode: MipmapMode::Linear,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BorderColor {
    FloatTransparentBlack,
    IntTransparentBlack,
    FloatOpaqueBlack,
    IntOpaqueBlack,
    FloatOpaqueWhite,
    IntOpaqueWhite,
}
