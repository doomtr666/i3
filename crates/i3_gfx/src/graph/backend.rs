/// Handle representing a physically allocated image in the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendImage(pub u64);

/// Handle representing a physically allocated buffer in the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendBuffer(pub u64);

/// Handle representing a physically allocated pipeline in the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendPipeline(pub u64);

/// Handle representing a command buffer recorded by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCommandBuffer(pub u64);

pub use crate::graph::types::{BufferDesc, ImageDesc, ResourceUsage};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WindowDesc {
    pub title: String,
    pub width: u32,
    pub height: u32,
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
    F11,
    Return,
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
    /// Stable index for selection (returned by enumerate_devices).
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
    pub command_buffers: Vec<BackendCommandBuffer>,
}

use crate::graph::types::{BufferHandle, ImageHandle, PipelineHandle};

#[derive(Debug, Clone)]
pub struct PassDescriptor<'a> {
    pub name: &'a str,
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

    /// Binds a raw backend pipeline. Used by passes that manage their own pipelines via `init`.
    fn bind_pipeline_raw(&mut self, pipeline: BackendPipeline);

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

    /// Creates a descriptor set for this pass.
    /// (Usually used for per-frame or per-pass bindings like Material SSBOs)
    fn create_descriptor_set(
        &mut self,
        pipeline: BackendPipeline,
        set_index: u32,
        writes: &[DescriptorWrite],
    ) -> DescriptorSetHandle;

    /// Binds a raw descriptor set handle (for global sets like bindless).
    fn bind_descriptor_set_raw(&mut self, set_index: u32, handle: u64);

    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32);
    fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32);

    // Commands
    fn draw(&mut self, vertex_count: u32, first_vertex: u32);
    fn draw_indexed(&mut self, index_count: u32, first_index: u32, vertex_offset: i32);
    fn push_bytes(
        &mut self,
        stages: crate::graph::pipeline::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    );
    fn dispatch(&mut self, x: u32, y: u32, z: u32);
    fn draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: BufferHandle,
        indirect_offset: u64,
        count_buffer:    BufferHandle,
        count_offset:    u64,
        max_draw_count:  u32,
        stride:          u32,
    );
    fn draw_indirect_count(
        &mut self,
        indirect_buffer: BufferHandle,
        indirect_offset: u64,
        count_buffer:    BufferHandle,
        count_offset:    u64,
        max_draw_count:  u32,
        stride:          u32,
    );
    fn clear_buffer(&mut self, buffer: crate::graph::types::BufferHandle, clear_value: u32);
    fn present(&mut self, image: crate::graph::types::ImageHandle);

    fn copy_buffer(
        &mut self,
        src: crate::graph::types::BufferHandle,
        dst: crate::graph::types::BufferHandle,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    );

    fn map_buffer(&mut self, handle: crate::graph::types::BufferHandle) -> *mut u8;
    fn unmap_buffer(&mut self, handle: crate::graph::types::BufferHandle);
}

/// Extension trait for [PassContext] to provide typed helpers.
pub trait PassContextExt: PassContext {
    /// Typed helper for push constants.
    fn push_constant_data<T: Sized>(
        &mut self,
        stages: crate::graph::pipeline::ShaderStageFlags,
        offset: u32,
        data: &T,
    ) {
        let bytes = unsafe {
            std::slice::from_raw_parts(data as *const T as *const u8, std::mem::size_of::<T>())
        };
        self.push_bytes(stages, offset, bytes);
    }
}

impl<T: PassContext + ?Sized> PassContextExt for T {}

/// Handle representing a physically allocated descriptor set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorSetHandle(pub u64);

/// GPU-side structure for indirect indexed drawing.
/// Matches VkDrawIndexedIndirectCommand.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DrawIndexedIndirectCommand {
    pub index_count:    u32,
    pub instance_count: u32,
    pub first_index:    u32,
    pub vertex_offset:  i32,
    pub first_instance: u32,
}

/// GPU-side structure for indirect drawing (non-indexed).
/// Matches VkDrawIndirectCommand.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DrawIndirectCommand {
    pub vertex_count:   u32,
    pub instance_count: u32,
    pub first_vertex:   u32,
    pub first_instance: u32,
}

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

    /// Returns the GPU device address for a buffer (requires BDA feature).
    fn get_buffer_device_address(&self, handle: BackendBuffer) -> u64;

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

    fn set_fullscreen(&mut self, window: crate::graph::types::WindowHandle, fullscreen: bool);

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

    fn create_graphics_pipeline_from_baked(
        &mut self,
        _desc: &i3_io::pipeline_asset::BakeableGraphicsPipeline,
        _reflection: &[u8],
        _bytecode: &[u8],
    ) -> BackendPipeline;

    fn create_compute_pipeline_from_baked(
        &mut self,
        _reflection: &[u8],
        _bytecode: &[u8],
    ) -> BackendPipeline;

    fn destroy_image(&mut self, handle: BackendImage);
    fn destroy_buffer(&mut self, handle: BackendBuffer);
    fn destroy_sampler(&mut self, handle: SamplerHandle);

    /// Upload raw bytes to a buffer.
    fn upload_buffer(
        &mut self,
        handle: BackendBuffer,
        data: &[u8],
        offset: u64,
    ) -> Result<(), String>;

    /// Upload raw bytes to an image using a staging buffer.
    fn upload_image(
        &mut self,
        handle: BackendImage,
        data: &[u8],
        offset_x: u32,
        offset_y: u32,
        data_width: u32,
        data_height: u32,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<(), String>;

    /// Returns the handle ID of the global bindless descriptor set.
    fn get_bindless_set_handle(&self) -> u64;

    // --- Resource Resolution ---
    fn resolve_image(&self, handle: crate::graph::types::ImageHandle) -> BackendImage;
    fn resolve_buffer(&self, handle: crate::graph::types::BufferHandle) -> BackendBuffer;
    fn resolve_pipeline(&self, handle: crate::graph::types::PipelineHandle) -> BackendPipeline;

    // --- Handle Registration ---
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

    // --- Descriptor Management ---
    /// Updates a specific index in an unbounded bindless texture array descriptor.
    fn update_bindless_texture(
        &mut self,
        texture: crate::graph::types::ImageHandle,
        sampler: SamplerHandle,
        index: u32,
        set: u64,
        binding: u32,
    );

    fn update_bindless_texture_raw(
        &mut self,
        texture: BackendImage,
        sampler: SamplerHandle,
        index: u32,
        set: u64,
        binding: u32,
    );

    /// Updates a specific binding in a bindless descriptor set with a sampler.
    fn update_bindless_sampler(
        &mut self,
        sampler: SamplerHandle,
        set: u64,
        binding: u32,
    );

    // --- Debug Utilities ---
    /// Nests a name for an image (no-op in release).
    fn set_image_name(&mut self, _image: BackendImage, _name: &str) {}
    /// Nests a name for a buffer (no-op in release).
    fn set_buffer_name(&mut self, _buffer: BackendBuffer, _name: &str) {}
}

/// Extension trait for [RenderBackend] to provide typed helpers.
pub trait RenderBackendExt: RenderBackend {
    /// Typed helper for uploading a single struct to a buffer.
    fn upload_buffer_data<T: Sized>(
        &mut self,
        handle: BackendBuffer,
        data: &T,
        offset: u64,
    ) -> Result<(), String> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data as *const T as *const u8, std::mem::size_of::<T>())
        };
        self.upload_buffer(handle, bytes, offset)
    }

    /// Typed helper for uploading a slice of structs to a buffer.
    fn upload_buffer_slice<T: Sized>(
        &mut self,
        handle: BackendBuffer,
        data: &[T],
        offset: u64,
    ) -> Result<(), String> {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.upload_buffer(handle, bytes, offset)
    }
}

impl<T: RenderBackend + ?Sized> RenderBackendExt for T {}

/// Internal trait consumed by the FrameGraph compiler and pass execution.
/// Backend implementations must implement this alongside `RenderBackend`.
/// User code should not call these methods directly.
pub trait RenderBackendInternal: RenderBackend + Send + Sync {
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

    type PreparedPass: Send + Sync;
    fn prepare_pass(&mut self, desc: PassDescriptor) -> Self::PreparedPass;

    /// Record barriers for a batch of prepared passes into a command buffer.
    fn record_barriers(&self, passes: &[&Self::PreparedPass]) -> Option<BackendCommandBuffer>;

    /// Record a pass into a command buffer.
    fn record_pass(
        &self,
        prepared: &Self::PreparedPass,
        pass: &dyn crate::graph::pass::RenderPass,
    ) -> (
        Option<u64>,
        Option<BackendCommandBuffer>,
        Option<crate::graph::types::ImageHandle>,
    );

    /// Forcefully updates an image state to PRESENT_SRC_KHR.
    /// Used after parallel recording to synchronize backend state.
    fn mark_image_as_presented(&mut self, handle: crate::graph::types::ImageHandle);

    // --- Descriptor Management (Internal) ---
    fn allocate_descriptor_set(
        &mut self,
        pipeline: crate::graph::types::PipelineHandle,
        set_index: u32,
    ) -> Result<DescriptorSetHandle, String>;

    fn update_descriptor_set(&mut self, set: DescriptorSetHandle, writes: &[DescriptorWrite]);

    // --- Debug Utilities (Internal) ---
    /// Begins a debug label for command buffer annotation (no-op in release).
    fn begin_debug_label(
        &self,
        _command_buffer: BackendCommandBuffer,
        _name: &str,
        _color: [f32; 4],
    ) {
    }
    /// Ends a debug label for command buffer annotation (no-op in release).
    fn end_debug_label(&self, _command_buffer: BackendCommandBuffer) {}
}

#[derive(Debug, Clone)]
pub struct DescriptorWrite {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_type: crate::graph::pipeline::BindingType, // Reusing from pipeline
    pub buffer_info: Option<DescriptorBufferInfo>,
    pub image_info: Option<DescriptorImageInfo>,
}

impl DescriptorWrite {
    pub fn buffer(binding: u32, buffer: crate::graph::types::BufferHandle) -> Self {
        Self {
            binding,
            array_element: 0,
            descriptor_type: crate::graph::pipeline::BindingType::StorageBuffer,
            buffer_info: Some(DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: 0,
            }),
            image_info: None,
        }
    }
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
