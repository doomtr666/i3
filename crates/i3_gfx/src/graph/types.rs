use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ImageHandle(pub SymbolId);

/// Handle for a virtual buffer resource, backed by a Symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferHandle(pub SymbolId);

/// Handle for a Window managed by the FrameGraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WindowHandle(pub u64);

/// Type-safe wrapper for an image acquired from a swapchain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SwapChainImageHandle(pub ImageHandle);

/// Handle for a compiled Graphics or Compute pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PipelineHandle(pub SymbolId);

/// Handle for a virtual acceleration structure, backed by a Symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AccelerationStructureHandle(pub SymbolId);

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
    AccelStruct(AccelerationStructureDesc),
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
    // Explicit discriminants — NEVER reorder or renumber existing entries.
    // Always append new formats at the end with the next available number.
    Undefined           = 0,
    R8G8B8A8_UNORM      = 1,
    R8G8B8A8_SRGB       = 2,
    B8G8R8A8_UNORM      = 3,
    B8G8R8A8_SRGB       = 4,
    R8G8_UNORM          = 5,
    R16G16B16A16_SFLOAT = 6,
    R32_FLOAT           = 7,
    R32G32B32A32_FLOAT  = 8,
    D32_FLOAT           = 9,
    R32_SINT            = 10,
    R32G32_SINT         = 11,
    R32G32B32_SINT      = 12,
    R32G32B32A32_SINT   = 13,
    R32_UINT            = 14,
    R32G32_UINT         = 15,
    R32G32B32_UINT      = 16,
    R32G32B32A32_UINT   = 17,
    R32_SFLOAT          = 18,
    R32G32_SFLOAT       = 19,
    R32G32B32_SFLOAT    = 20,
    R32G32B32A32_SFLOAT = 21,
    B10G11R11_UFLOAT_PACK32 = 22,
    BC1_RGB_UNORM       = 23,
    BC1_RGB_SRGB        = 24,
    BC1_RGBA_UNORM      = 25,
    BC1_RGBA_SRGB       = 26,
    BC3_UNORM           = 27,
    BC3_SRGB            = 28,
    BC4_UNORM           = 29,
    BC4_SNORM           = 30,
    BC5_UNORM           = 31,
    BC5_SNORM           = 32,
    BC7_UNORM           = 33,
    BC7_SRGB            = 34,
    R16G16_SFLOAT       = 35,
    R11G11B10_UFLOAT    = 36,
    BC6H_UF16           = 37,
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
                | Format::BC6H_UF16
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
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
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
        const SHADER_DEVICE_ADDRESS = 0x200;
        const ACCELERATION_STRUCTURE_BUILD_INPUT = 0x400;
        const ACCELERATION_STRUCTURE_STORAGE = 0x800;
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
    pub clear_value: Option<ClearValue>,
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
            clear_value: None,
        }
    }

    pub fn with_clear_value(mut self, value: ClearValue) -> Self {
        self.clear_value = Some(value);
        self
    }
}

impl Default for ImageDesc {
    fn default() -> Self {
        Self::new(0, 0, Format::Undefined)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClearValue {
    Color(u32, u32, u32, u32),
    DepthStencil(u32, u32),
}

impl ClearValue {
    pub fn color(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self::Color(r.to_bits(), g.to_bits(), b.to_bits(), a.to_bits())
    }

    pub fn depth_stencil(depth: f32, stencil: u32) -> Self {
        Self::DepthStencil(depth.to_bits(), stencil)
    }

    pub fn as_color(&self) -> [f32; 4] {
        match self {
            Self::Color(r, g, b, a) => [
                f32::from_bits(*r),
                f32::from_bits(*g),
                f32::from_bits(*b),
                f32::from_bits(*a),
            ],
            _ => [0.0; 4],
        }
    }

    pub fn as_depth_stencil(&self) -> (f32, u32) {
        match self {
            Self::DepthStencil(d, s) => (f32::from_bits(*d), *s),
            _ => (0.0, 0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryType {
    #[default]
    GpuOnly,
    CpuToGpu, // Host visible, coherent
    GpuToCpu, // Host visible, coherent
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: BufferUsageFlags,
    pub memory: MemoryType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AccelerationStructureDesc {
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

impl AccelerationStructureHandle {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QueueType {
    Graphics,
    AsyncCompute,
    Transfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputKind {
    /// Window presentation (swapchain).
    Present(WindowHandle),
    /// GPU-side texture intended for external sampling or later readback.
    Texture,
    /// Intended for CPU readback after a mapping or staging copy.
    Readback,
    /// Preserved across multiple execute() calls for iterative/temporal baking.
    AccumulationBuffer,
}

/// Description of a cross-queue resource transfer (ownership transfer).
/// Generated by the compiler when it detects a dependency crossing queue boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CrossQueueTransfer {
    pub image: Option<ImageHandle>,
    pub buffer: Option<BufferHandle>,
    pub src_queue: QueueType,
    pub dst_queue: QueueType,
    /// Usage on the source queue (for the Release barrier)
    pub src_usage: ResourceUsage,
    /// Usage on the destination queue (for the Acquire barrier)
    pub dst_usage: ResourceUsage,
}

/// A flattened pass extracted from the node tree.
#[derive(Debug, Clone)]
pub struct FlatPass {
    pub node_id: u64,
    pub name: String,
    pub domain: PassDomain,
    pub pipeline: Option<PipelineHandle>,
    pub image_reads: Vec<(ImageHandle, ResourceUsage)>,
    pub image_writes: Vec<(ImageHandle, ResourceUsage)>,
    pub buffer_reads: Vec<(BufferHandle, ResourceUsage)>,
    pub buffer_writes: Vec<(BufferHandle, ResourceUsage)>,
    pub data_reads: Vec<String>,
    pub data_writes: Vec<String>,
    pub prefer_async: bool,
    pub queue: QueueType,
    pub releases: Vec<CrossQueueTransfer>,
    pub acquires: Vec<CrossQueueTransfer>,
    /// Images to transition to PresentSrc after this pass executes (post-transition).
    pub present_images: Vec<ImageHandle>,
}

impl FlatPass {
    /// Infer the execution domain from resource usage intents.
    pub fn infer_domain_from_intents(
        image_reads: &[(ImageHandle, ResourceUsage)],
        image_writes: &[(ImageHandle, ResourceUsage)],
        buffer_reads: &[(BufferHandle, ResourceUsage)],
        buffer_writes: &[(BufferHandle, ResourceUsage)],
        has_pipeline: bool,
    ) -> PassDomain {
        let has_raster = image_writes.iter().any(|(_, u)| {
            u.intersects(ResourceUsage::COLOR_ATTACHMENT | ResourceUsage::DEPTH_STENCIL)
        });

        if has_raster {
            return PassDomain::Graphics;
        }

        let has_transfer = image_reads
            .iter()
            .any(|(_, u)| u.intersects(ResourceUsage::TRANSFER_READ))
            || image_writes
                .iter()
                .any(|(_, u)| u.intersects(ResourceUsage::TRANSFER_WRITE))
            || buffer_reads
                .iter()
                .any(|(_, u)| u.intersects(ResourceUsage::TRANSFER_READ))
            || buffer_writes
                .iter()
                .any(|(_, u)| u.intersects(ResourceUsage::TRANSFER_WRITE));

        if has_transfer && !has_pipeline {
            return PassDomain::Transfer;
        }

        let has_gpu_work = has_pipeline
            || !image_reads.is_empty()
            || !image_writes.is_empty()
            || !buffer_reads.is_empty()
            || !buffer_writes.is_empty();

        if has_gpu_work {
            return PassDomain::Compute;
        }

        PassDomain::Cpu
    }
}

bitflags! {
    /// Bitfield representing how a resource is used within a pass.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
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
        const PRESENT = 1 << 8;

        const INDIRECT_READ = 1 << 10;

        const ACCEL_STRUCT_READ = 1 << 11;
        const ACCEL_STRUCT_WRITE = 1 << 12;

        /// Explicit intent to clear a resource (Image or Buffer).
        /// For Images, this promotes the first-use load_op to CLEAR.
        const CLEAR = 1 << 13;
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
    pub anisotropy: u32, // Use discrete levels (1, 2, 4, 8, 16)
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
            anisotropy: 1,
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
