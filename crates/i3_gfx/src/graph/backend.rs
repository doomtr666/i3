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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyCode {
    Escape,
    Tab,
    Space,
    W,
    A,
    S,
    D,
    Z,
    Q,
    LShift,
    // Add more as needed
}

#[derive(Debug, Clone)]
pub enum Event {
    Quit,
    KeyDown { key: KeyCode },
    KeyUp { key: KeyCode },
    Resize { width: u32, height: u32 },
    MouseDown { button: u8, x: i32, y: i32 },
    MouseUp { button: u8, x: i32, y: i32 },
    MouseMove { x: i32, y: i32 },
    MouseWheel { x: i32, y: i32 },
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
pub struct PassDescriptor<'a> {
    pub name: &'a str,
    pub domain: crate::graph::types::PassDomain,
    pub pipeline: Option<PipelineHandle>,
    pub image_reads: &'a [(ImageHandle, ResourceUsage)],
    pub image_writes: &'a [(ImageHandle, ResourceUsage)],
    pub buffer_reads: &'a [(BufferHandle, ResourceUsage)],
    pub buffer_writes: &'a [(BufferHandle, ResourceUsage)],
    pub descriptor_sets: &'a [(u32, Vec<DescriptorWrite>)],
}

/// Hardware-specific context used to record commands during a pass.
pub trait PassContext {
    // Pipeline & Binding (Logical interfaces)
    fn bind_pipeline(&mut self, pipeline: crate::graph::types::PipelineHandle);
    fn bind_vertex_buffer(&mut self, binding: u32, handle: crate::graph::types::BufferHandle);
    fn bind_index_buffer(
        &mut self,
        handle: crate::graph::types::BufferHandle,
        index_type: crate::graph::pipeline::IndexType,
    );
    // Placeholder for descriptor binding - simplified for Phase 19
    // In a real engine, we'd bind specific sets or resources.
    // For now, let's assume specific sets are bound by their index.
    fn bind_descriptor_set(&mut self, set_index: u32, handle: DescriptorSetHandle);

    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32);
    fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32);

    // Commands
    fn draw(&mut self, vertex_count: u32, first_vertex: u32);
    fn draw_indexed(&mut self, index_count: u32, first_index: u32, vertex_offset: i32);
    fn push_constants(
        &mut self,
        stages: crate::graph::pipeline::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    );
    fn dispatch(&mut self, x: u32, y: u32, z: u32);
    fn clear_buffer(&mut self, buffer: crate::graph::types::BufferHandle, clear_value: u32);
    fn present(&mut self, image: crate::graph::types::ImageHandle);
}

/// Handle representing a physically allocated descriptor set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorSetHandle(pub u64);

/// The main interface for hardware backends (Vulkan, DX12, Null).
/// This trait exposes only user-facing operations: lifecycle, windowing, resource creation,
/// pipeline management, and data upload.
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
    fn create_sampler(&mut self, desc: &crate::graph::types::SamplerDesc) -> SamplerHandle;

    fn create_graphics_pipeline(
        &mut self,
        desc: &crate::graph::pipeline::GraphicsPipelineCreateInfo,
    ) -> BackendPipeline;

    fn create_compute_pipeline(
        &mut self,
        desc: &crate::graph::pipeline::ComputePipelineCreateInfo,
    ) -> BackendPipeline;

    fn destroy_image(&mut self, handle: BackendImage);
    fn destroy_buffer(&mut self, handle: BackendBuffer);
    fn destroy_sampler(&mut self, handle: SamplerHandle);

    // --- Data Upload ---
    /// Upload data to a buffer.
    fn upload_buffer(
        &mut self,
        handle: BackendBuffer,
        data: &[u8],
        offset: u64,
    ) -> Result<(), String>;
}

/// Internal trait consumed by the FrameGraph compiler and pass execution.
/// Backend implementations must implement this alongside `RenderBackend`.
/// User code should not call these methods directly.
pub trait RenderBackendInternal: RenderBackend {
    // --- Frame Control ---
    fn begin_frame(&mut self);
    fn end_frame(&mut self);

    /// Acquire the next available image from the swapchain associated with the window.
    fn acquire_swapchain_image(
        &mut self,
        window: crate::graph::types::WindowHandle,
    ) -> Result<Option<(BackendImage, u64, u32)>, String>;

    // --- Execution & Sync ---

    /// Submit a batch of commands to the GPU.
    fn submit(
        &mut self,
        batch: CommandBatch,
        wait_sems: &[u64],
        signal_sems: &[u64],
    ) -> Result<u64, String>;

    fn begin_pass(
        &mut self,
        desc: PassDescriptor,
        f: Box<dyn FnOnce(&mut dyn PassContext) + Send + Sync>,
    ) -> u64;

    // --- Handle Resolution ---
    fn resolve_image(&self, handle: crate::graph::types::ImageHandle) -> BackendImage;
    fn resolve_buffer(&self, handle: crate::graph::types::BufferHandle) -> BackendBuffer;
    fn resolve_pipeline(&self, handle: crate::graph::types::PipelineHandle) -> BackendPipeline;
    fn register_external_image(
        &mut self,
        handle: crate::graph::types::ImageHandle,
        physical: BackendImage,
    );
    fn register_external_buffer(
        &mut self,
        handle: crate::graph::types::BufferHandle,
        physical: BackendBuffer,
    );

    /// Wait for the timeline semaphore to reach a specific value on the host (CPU).
    fn wait_for_timeline(&self, value: u64, timeout_ns: u64) -> Result<(), String>;

    // --- Transient Resource Management (Pooling) ---
    fn create_transient_image(&mut self, desc: &ImageDesc) -> BackendImage;
    fn create_transient_buffer(&mut self, desc: &BufferDesc) -> BackendBuffer;
    fn release_transient_image(&mut self, handle: BackendImage);
    fn release_transient_buffer(&mut self, handle: BackendBuffer);
    fn garbage_collect(&mut self);

    // --- Descriptor Management (Internal) ---
    fn allocate_descriptor_set(
        &mut self,
        pipeline: crate::graph::types::PipelineHandle,
        set_index: u32,
    ) -> Result<DescriptorSetHandle, String>;

    fn update_descriptor_set(&mut self, set: DescriptorSetHandle, writes: &[DescriptorWrite]);
}

#[derive(Debug, Clone)]
pub struct DescriptorWrite {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_type: crate::graph::pipeline::BindingType, // Reusing from pipeline
    pub buffer_info: Option<DescriptorBufferInfo>,
    pub image_info: Option<DescriptorImageInfo>,
}

#[derive(Debug, Clone)]
pub struct DescriptorBufferInfo {
    pub buffer: crate::graph::types::BufferHandle,
    pub offset: u64,
    pub range: u64, // or whole size
}

#[derive(Debug, Clone)]
pub struct DescriptorImageInfo {
    pub image: crate::graph::types::ImageHandle,
    pub image_layout: DescriptorImageLayout, // New type needed?
    // Start with basic layout. Or reuse image layout from types?
    // Actually vk::DescriptorImageInfo needs sampler too.
    pub sampler: Option<SamplerHandle>, // Need SamplerHandle
}

// Temporary Sampler Handle until we have proper sampler resources
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerHandle(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorImageLayout {
    General,
    ShaderReadOnlyOptimal,
    // Add others as needed
}
