use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct ShaderStageFlags: u32 {
        const Vertex = 1 << 0;
        const Fragment = 1 << 1;
        const Compute = 1 << 2;
        const Geometry = 1 << 3;
        const All = Self::Vertex.bits() | Self::Fragment.bits() | Self::Compute.bits() | Self::Geometry.bits();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingType {
    Unknown,
    UniformBuffer,
    StorageBuffer,
    Texture,
    StorageTexture,
    Sampler,
    CombinedImageSampler,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub name: String,
    pub binding: u32,
    pub set: u32,
    pub count: u32,
    pub binding_type: BindingType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPointInfo {
    pub name: String,
    pub stage: String, // "vertex", "fragment", etc.
    pub thread_group_size: Option<[u64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushConstantRange {
    pub stage_flags: ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderReflection {
    pub entry_points: Vec<EntryPointInfo>,
    pub bindings: Vec<Binding>,
    pub push_constants: Vec<PushConstantRange>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderStageInfo {
    pub stage: ShaderStageFlags,
    pub entry_point: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderModule {
    pub bytecode: Vec<u8>,
    pub stages: Vec<ShaderStageInfo>,
    pub reflection: ShaderReflection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphicsPipelineDesc {
    pub shader: ShaderModule,
    pub name: String,
}

/// Handle representing a physically allocated image in the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendImage(pub u64);

/// Handle representing a physically allocated buffer in the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendBuffer(pub u64);

/// Handle representing a physically allocated pipeline in the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendPipeline(pub u64);

/// Hardware-specific context used to record commands during a pass.
pub trait PassContext {
    // Pipeline & Binding (Logical interfaces)
    fn bind_pipeline(&mut self, pipeline: crate::graph::types::PipelineHandle);
    fn bind_image(&mut self, slot: u32, handle: crate::graph::types::ImageHandle);
    fn bind_buffer(&mut self, slot: u32, handle: crate::graph::types::BufferHandle);

    // Commands
    fn draw(&mut self, vertex_count: u32, first_vertex: u32);
    fn dispatch(&mut self, x: u32, y: u32, z: u32);

    // Presentation
    fn present(&mut self, handle: crate::graph::types::ImageHandle);
}

pub use crate::graph::types::{BufferDesc, ImageDesc};

/// The main interface for hardware backends (Vulkan, DX12, Null).
pub trait RenderBackend {
    // Resource Management
    fn create_image(&mut self, desc: &ImageDesc) -> BackendImage;
    fn create_buffer(&mut self, desc: &BufferDesc) -> BackendBuffer;
    fn create_graphics_pipeline(&mut self, desc: &GraphicsPipelineDesc) -> BackendPipeline;

    // Window / Swapchain
    fn create_swapchain(&mut self, window_handle: u64, usages: u32) -> u64;
    fn present_swapchain(&mut self, sc_handle: u64, image: BackendImage);

    // Execution
    fn begin_pass(&mut self, name: &str, f: Box<dyn FnOnce(&mut dyn PassContext) + Send + Sync>);

    // Handle Resolution (Called by the FrameGraph during execution)
    fn resolve_image(&self, handle: crate::graph::types::ImageHandle) -> BackendImage;
    fn resolve_buffer(&self, handle: crate::graph::types::BufferHandle) -> BackendBuffer;
    fn resolve_pipeline(&self, handle: crate::graph::types::PipelineHandle) -> BackendPipeline;
    fn register_external_image(
        &mut self,
        handle: crate::graph::types::ImageHandle,
        physical: BackendImage,
    );
}
