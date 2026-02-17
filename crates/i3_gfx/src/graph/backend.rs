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
    pub color_formats: Vec<crate::graph::types::Format>,
    pub depth_format: Option<crate::graph::types::Format>,
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

pub use crate::graph::types::{BufferDesc, ImageDesc, ResourceUsage};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WindowDesc {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphicsPipelineDescDummy {
    // Renamed to avoid conflict with existing GraphicsPipelineDesc
    // This is hard to hash/eq genericly. For MVP, we might use name?
    // Or pointer?
    // For now, let's assume we don't pool pipelines this way or used differently.
    // Pipelines are usually cached by hash of state.
    pub dummy: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyCode {
    Escape,
    Space,
    W,
    A,
    S,
    D,
    // Add more as needed
}

#[derive(Debug, Clone)]
pub enum Event {
    Quit,
    KeyDown { key: KeyCode },
    Resize { width: u32, height: u32 },
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub id: u32,
    pub name: String,
    pub device_type: DeviceType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Any,
    Discrete,
    Integrated,
    Virtual,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwapchainConfig {
    pub vsync: bool,
    pub srgb: bool,
    pub min_image: u32,
}

#[derive(Debug, Clone, Default)]
pub struct CommandBatch {
    // This will be refined as we implement the execution engine
}

use crate::graph::types::{BufferHandle, ImageHandle, PipelineHandle};

#[derive(Debug, Clone)]
pub struct PassDescriptor {
    pub name: String,
    pub pipeline: Option<PipelineHandle>,
    pub image_reads: Vec<ImageHandle>,
    pub image_writes: Vec<ImageHandle>,
    pub buffer_reads: Vec<BufferHandle>,
    pub buffer_writes: Vec<BufferHandle>,
}

/// Hardware-specific context used to record commands during a pass.
pub trait PassContext {
    // Pipeline & Binding (Logical interfaces)
    fn bind_pipeline(&mut self, pipeline: crate::graph::types::PipelineHandle);
    fn bind_image(&mut self, slot: u32, handle: crate::graph::types::ImageHandle);
    fn bind_buffer(&mut self, slot: u32, handle: crate::graph::types::BufferHandle);

    // Commands
    fn draw(&mut self, vertex_count: u32, first_vertex: u32);
    fn dispatch(&mut self, x: u32, y: u32, z: u32);
    fn present(&mut self, image: crate::graph::types::ImageHandle);
}

/// The main interface for hardware backends (Vulkan, DX12, Null).
pub trait RenderBackend {
    // --- Lifecycle & Device Management ---

    /// Enumerate all compatible hardware devices on the system.
    fn enumerate_devices(&self) -> Vec<DeviceInfo>;

    /// Initialize the backend with a specific device.
    /// Should be called before any other operation.
    fn initialize(&mut self, device_id: u32) -> Result<(), String>;

    // --- Windowing & Events (Managed by Backend) ---

    /// Create a native window. The backend handles the connection (e.g. SDL2, Win32).
    fn create_window(
        &mut self,
        desc: WindowDesc,
    ) -> Result<crate::graph::types::WindowHandle, String>;

    fn destroy_window(&mut self, window: crate::graph::types::WindowHandle);

    fn configure_window(
        &mut self,
        window: crate::graph::types::WindowHandle,
        config: SwapchainConfig,
    ) -> Result<(), String>;

    /// Poll events from the windowing system.
    fn poll_events(&mut self) -> Vec<Event>;

    // --- Resource Management ---
    fn create_image(&mut self, desc: &ImageDesc) -> BackendImage;
    fn create_buffer(&mut self, desc: &BufferDesc) -> BackendBuffer;
    fn create_graphics_pipeline(&mut self, desc: &GraphicsPipelineDesc) -> BackendPipeline;

    fn destroy_image(&mut self, handle: BackendImage);
    fn destroy_buffer(&mut self, handle: BackendBuffer);

    // --- Transient Resource Management (Pooling) ---
    fn create_transient_image(&mut self, desc: &ImageDesc) -> BackendImage;
    fn create_transient_buffer(&mut self, desc: &BufferDesc) -> BackendBuffer;
    fn release_transient_image(&mut self, handle: BackendImage);
    fn release_transient_buffer(&mut self, handle: BackendBuffer);
    fn garbage_collect(&mut self);

    // --- Frame Control (Internal) ---

    /// Acquire the next available image from the swapchain associated with the window.
    /// Returns the image handle and a binary semaphore handle that will be signaled when the image is ready.
    fn acquire_swapchain_image(
        &mut self,
        window: crate::graph::types::WindowHandle,
    ) -> Result<(BackendImage, u64, u32), String>;

    // --- Execution & Sync ---

    /// Submit a batch of commands to the GPU.
    /// Returns a timeline semaphore value representing the completion of this batch.
    fn submit(
        &mut self,
        batch: CommandBatch,
        wait_sems: Vec<u64>,
        signal_sems: Vec<u64>,
    ) -> Result<u64, String>;

    fn begin_pass(
        &mut self,
        desc: PassDescriptor,
        f: Box<dyn FnOnce(&mut dyn PassContext) + Send + Sync>,
    ) -> u64;

    // Handle Resolution (Called by the FrameGraph during execution)
    fn resolve_image(&self, handle: crate::graph::types::ImageHandle) -> BackendImage;
    fn resolve_buffer(&self, handle: crate::graph::types::BufferHandle) -> BackendBuffer;
    fn resolve_pipeline(&self, handle: crate::graph::types::PipelineHandle) -> BackendPipeline;
    fn register_external_image(
        &mut self,
        handle: crate::graph::types::ImageHandle,
        physical: BackendImage,
    );

    /// Wait for the timeline semaphore to reach a specific value on the host (CPU).
    /// Timeout is in nanoseconds.
    fn wait_for_timeline(&self, value: u64, timeout_ns: u64) -> Result<(), String>;
}
