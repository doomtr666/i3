use ash::vk;
use ash::vk::Handle;
use i3_gfx::graph::backend::RenderBackendInternal;
use i3_gfx::graph::backend::*;
use i3_gfx::graph::pass::RenderPass;
use i3_gfx::graph::pipeline::*;
use i3_gfx::graph::types::*;
use std::collections::HashMap;

use std::sync::Arc;
use tracing::{debug, info};

use crate::resource_arena::{PhysicalBuffer, PhysicalImage, PhysicalPipeline, ResourceArena};
use crate::window_context::WindowContext;

pub(crate) use crate::commands::{
    PreparedDomain, ThreadCommandPool, VulkanFrameContext, VulkanPreparedPass,
};

use crate::sync;

/// Per-queue context for managing execution, synchronization and command pools.
pub struct QueueContext {
    pub(crate) queue: vk::Queue,
    #[allow(dead_code)]
    pub(crate) family: u32,
    pub(crate) timeline_sem: vk::Semaphore,
    pub(crate) cpu_timeline: u64,
    pub(crate) frame_contexts: Vec<VulkanFrameContext>,
}

/// Main Vulkan backend struct.
///
/// This struct contains all the state needed for rendering, including:
/// - Window management
/// - Resource arenas
/// - Synchronization primitives
/// - Descriptor pools
/// - Frame contexts
pub struct VulkanBackend {
    // Window Management
    pub(crate) windows: HashMap<u64, WindowContext>,
    pub(crate) next_window_id: u64,

    // SDL2 context
    pub sdl: sdl2::Sdl,
    pub video: sdl2::VideoSubsystem,
    pub event_pump: Option<sdl2::EventPump>,

    // Resource tracking for teardown
    pub pipeline_resources: ResourceArena<PhysicalPipeline>,
    pub shader_modules: Vec<vk::ShaderModule>,

    // Resources mapping (Arena based)
    pub images: ResourceArena<PhysicalImage>,
    pub buffers: ResourceArena<PhysicalBuffer>,
    pub external_to_physical: HashMap<u64, u64>, // Virtual ID -> Physical ID
    pub external_buffer_to_physical: HashMap<u64, u64>,

    pub frame_count: u64,
    pub dead_images: Vec<(u64, vk::Image, vk::ImageView, vk_mem::Allocation)>, // Frame, Image, View, Alloc
    pub dead_buffers: Vec<(u64, vk::Buffer, vk_mem::Allocation)>,
    pub dead_semaphores: Vec<(u64, u64, vk::Semaphore)>, // Frame, ID, Handle
    pub recycled_semaphores: Vec<vk::Semaphore>,

    pub accel_structs: ResourceArena<crate::accel_struct::PhysicalAccelerationStructure>,
    pub dead_accel_structs: Vec<(
        u64,
        vk::AccelerationStructureKHR,
        vk::Buffer,
        vk_mem::Allocation,
    )>,

    // Transient Pools
    pub(crate) transient_image_pool: HashMap<ImageDesc, Vec<u64>>,
    pub(crate) transient_buffer_pool: HashMap<BufferDesc, Vec<u64>>,

    // Resources
    pub samplers: ResourceArena<vk::Sampler>,
    pub dead_samplers: Vec<(u64, vk::Sampler)>, // Frame, Handle
    pub semaphores: ResourceArena<vk::Semaphore>,
    pub next_semaphore_id: u64,

    pub(crate) graphics: Option<QueueContext>,
    pub(crate) compute: Option<QueueContext>,
    pub(crate) transfer: Option<QueueContext>,

    pub static_descriptor_pool: vk::DescriptorPool,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub descriptor_sets: std::sync::Mutex<ResourceArena<vk::DescriptorSet>>,
    pub descriptor_pool_max_sets: u32,

    // Loaders
    pub(crate) swapchain_loader: Option<ash::khr::swapchain::Device>,
    pub dynamic_rendering: Option<ash::khr::dynamic_rendering::Device>,
    pub sync2: Option<ash::khr::synchronization2::Device>,

    pub accel_struct: Option<ash::khr::acceleration_structure::Device>,
    pub rt_pipeline: Option<ash::khr::ray_tracing_pipeline::Device>,

    pub rt_supported: bool,

    /* Removed oracle field - now uses i3_gfx::graph::oracle::SyncPlanner during analyze_frame */
    /// Current synchronization plan for the active frame.
    pub(crate) current_plan: Option<sync::SyncPlan>,

    // Global Frame Contexts
    pub(crate) frame_started: bool,
    pub(crate) global_frame_index: usize,
    pub next_resource_id: u64,
    pub bindless_set_layout: vk::DescriptorSetLayout,
    pub bindless_set_handle: u64,

    pub graphics_family: u32,
    pub compute_family: u32,
    pub transfer_family: u32,

    // Dependencies (Dropped Last)
    pub device: Option<Arc<crate::device::VulkanDevice>>,
    pub instance: Arc<crate::instance::VulkanInstance>,
}

unsafe impl Send for VulkanBackend {}
unsafe impl Sync for VulkanBackend {}

impl VulkanBackend {
    /// Create a new Vulkan backend instance.
    ///
    /// This function initializes:
    /// - Vulkan instance
    /// - SDL2 context
    /// - Resource arenas
    /// - Frame contexts (for frame-in-flight management)
    ///
    /// # Returns
    ///
    /// A new VulkanBackend instance, or an error if initialization fails
    pub fn new() -> Result<Self, String> {
        let instance = crate::instance::VulkanInstance::new()?;
        let sdl = sdl2::init()?;
        let video = sdl.video()?;

        Ok(VulkanBackend {
            instance,
            device: None,
            windows: HashMap::new(),
            next_window_id: 1,
            sdl,
            video,
            event_pump: None,
            pipeline_resources: ResourceArena::new(),
            shader_modules: Vec::new(),
            images: ResourceArena::new(),
            buffers: ResourceArena::new(),
            external_to_physical: HashMap::new(),
            external_buffer_to_physical: HashMap::new(),
            semaphores: ResourceArena::new(),
            samplers: ResourceArena::new(),
            next_semaphore_id: 1,
            frame_count: 0,
            dead_images: Vec::new(),
            dead_buffers: Vec::new(),
            dead_semaphores: Vec::new(),
            dead_samplers: Vec::new(),
            recycled_semaphores: Vec::new(),
            graphics_family: 0,
            compute_family: 0,
            transfer_family: 0,
            transient_image_pool: HashMap::new(),
            transient_buffer_pool: HashMap::new(),
            static_descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set_layouts: Vec::new(),
            descriptor_sets: std::sync::Mutex::new(ResourceArena::new()),
            descriptor_pool_max_sets: 1000,
            graphics: None,
            compute: None,
            transfer: None,
            swapchain_loader: None,
            dynamic_rendering: None,
            sync2: None,
            accel_struct: None,
            rt_pipeline: None,
            rt_supported: false,

            current_plan: None,

            frame_started: false,
            global_frame_index: 0,
            next_resource_id: 1,
            bindless_set_layout: vk::DescriptorSetLayout::null(),
            bindless_set_handle: 0,
            accel_structs: ResourceArena::new(),
            dead_accel_structs: Vec::new(),
        })
    }

    pub fn get_device(&self) -> &Arc<crate::device::VulkanDevice> {
        self.device.as_ref().expect("Backend not initialized")
    }

    pub fn get_image_barrier(
        &mut self,
        physical_id: u64,
        new_layout: vk::ImageLayout,
        dst_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        current_queue_family: u32,
    ) -> Option<vk::ImageMemoryBarrier2<'static>> {
        crate::sync::get_image_barrier(
            self,
            physical_id,
            new_layout,
            dst_access,
            dst_stage,
            current_queue_family,
        )
    }

    pub fn get_buffer_barrier(
        &mut self,
        physical_id: u64,
        dst_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        current_queue_family: u32,
    ) -> Option<vk::BufferMemoryBarrier2<'static>> {
        crate::sync::get_buffer_barrier(
            self,
            physical_id,
            dst_access,
            dst_stage,
            current_queue_family,
        )
    }

    pub fn get_image_state(
        &self,
        usage: ResourceUsage,
        bind_point: vk::PipelineBindPoint,
    ) -> (vk::ImageLayout, vk::AccessFlags2, vk::PipelineStageFlags2) {
        crate::sync::get_image_state(usage, bind_point)
    }

    pub fn get_buffer_state(
        &self,
        usage: ResourceUsage,
        bind_point: vk::PipelineBindPoint,
    ) -> (vk::AccessFlags2, vk::PipelineStageFlags2) {
        crate::sync::get_buffer_state(usage, bind_point)
    }

    #[allow(dead_code)]
    pub fn create_semaphore(&mut self, is_timeline: bool) -> u64 {
        crate::sync::create_semaphore(self, is_timeline)
    }

    pub fn next_id(&mut self) -> u64 {
        let id = self.next_resource_id;
        self.next_resource_id += 1;
        id
    }

    pub fn window_size(&self, window: WindowHandle) -> Option<(u32, u32)> {
        crate::window_context::window_size(self, window)
    }

    fn init_frame_contexts_for_family(
        &self,
        family_index: u32,
    ) -> Result<Vec<VulkanFrameContext>, String> {
        let mut frame_contexts = Vec::new();
        let device = self.get_device().clone();

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 4096,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 4096,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 4096,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 4096,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 4096,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 4096,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .pool_sizes(&pool_sizes)
            .max_sets(4096);

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        for _ in 0..3 {
            unsafe {
                let pool = device
                    .handle
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(family_index)
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )
                    .map_err(|e| e.to_string())?;

                let d_pool = device
                    .handle
                    .create_descriptor_pool(&pool_info, None)
                    .map_err(|e| e.to_string())?;

                let mut per_thread_pools = Vec::with_capacity(num_threads);
                for _ in 0..num_threads {
                    let tp = device
                        .handle
                        .create_command_pool(
                            &vk::CommandPoolCreateInfo::default()
                                .queue_family_index(family_index)
                                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                            None,
                        )
                        .map_err(|e| e.to_string())?;

                    let tp_d_pool = device
                        .handle
                        .create_descriptor_pool(&pool_info, None)
                        .map_err(|e| e.to_string())?;

                    per_thread_pools.push(std::sync::Mutex::new(ThreadCommandPool {
                        pool: tp,
                        descriptor_pool: tp_d_pool,
                        allocated: Vec::new(),
                        cursor: 0,
                    }));
                }

                frame_contexts.push(VulkanFrameContext {
                    command_pool: pool,
                    descriptor_pool: d_pool,
                    allocated_command_buffers: Vec::new(),
                    cursor: 0,
                    submitted_cursor: 0,
                    last_completion_value: 0,
                    per_thread_pools,
                });
            }
        }
        Ok(frame_contexts)
    }

    fn create_queue_context(&self, queue: vk::Queue, family: u32) -> Result<QueueContext, String> {
        let device = self.get_device();

        let mut type_info =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

        let timeline_sem = unsafe {
            device
                .handle
                .create_semaphore(&create_info, None)
                .map_err(|e| format!("Failed to create timeline semaphore: {}", e))?
        };

        let frame_contexts = self.init_frame_contexts_for_family(family)?;

        Ok(QueueContext {
            queue,
            family,
            timeline_sem,
            cpu_timeline: 0,
            frame_contexts,
        })
    }

    fn init_bindless(&mut self) -> Result<(), String> {
        let device = self.get_device().clone();
        unsafe {
            let layout_bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(4096)
                    .stage_flags(vk::ShaderStageFlags::ALL),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::ALL),
            ];

            let binding_flags = [
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                vk::DescriptorBindingFlags::empty(),
            ];

            let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&binding_flags);

            let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .bindings(&layout_bindings)
                .push_next(&mut flags_info);

            let layout = device
                .handle
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| format!("Failed to create bindless layout: {}", e))?;

            self.bindless_set_layout = layout;

            let layouts = [layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.static_descriptor_pool)
                .set_layouts(&layouts);

            let sets = device
                .handle
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| format!("Failed to allocate bindless set: {}", e))?;

            let set = sets[0];

            let handle_id = self.descriptor_sets.lock().unwrap().insert(set);
            self.bindless_set_handle = handle_id;
        }
        Ok(())
    }
}

impl RenderBackend for VulkanBackend {
    fn capabilities(&self) -> DeviceCapabilities {
        let device = self.get_device();
        DeviceCapabilities {
            ray_tracing: device.rt_supported,
            async_compute: device.compute_queue.is_some(),
            async_transfer: device.transfer_queue.is_some(),
        }
    }

    fn enumerate_devices(&self) -> Vec<DeviceInfo> {
        let pdevices =
            unsafe { self.instance.handle.enumerate_physical_devices() }.unwrap_or_default();

        pdevices
            .iter()
            .enumerate()
            .map(|(id, &p)| {
                let props = unsafe { self.instance.handle.get_physical_device_properties(p) };
                let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();

                let device_type = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::Discrete,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::Integrated,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceType::Virtual,
                    vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
                    _ => DeviceType::Any,
                };

                DeviceInfo {
                    id: id as u32,
                    name,
                    device_type,
                }
            })
            .collect()
    }

    fn initialize(&mut self, device_id: u32) -> Result<(), String> {
        let pdevices = unsafe { self.instance.handle.enumerate_physical_devices() }
            .map_err(|e| format!("Failed to enumerate physical devices: {}", e))?;

        if pdevices.is_empty() {
            return Err("No Vulkan physical devices found".to_string());
        }

        let physical_device = if (device_id as usize) < pdevices.len() {
            pdevices[device_id as usize]
        } else {
            tracing::warn!(
                "Requested GPU index {} is out of bounds (max {}). Falling back to GPU 0.",
                device_id,
                pdevices.len() - 1
            );
            pdevices[0]
        };

        let device =
            crate::device::VulkanDevice::new_with_physical(self.instance.clone(), physical_device)?;
        self.device = Some(Arc::new(device));
        let d = self.device.as_ref().unwrap();
        self.graphics_family = d.graphics_family;
        self.compute_family = d.compute_family.unwrap_or(d.graphics_family);
        self.transfer_family = d.transfer_family.unwrap_or(d.graphics_family);

        self.event_pump = Some(self.sdl.event_pump()?);

        // Create Queue Contexts
        let device_ptr = self.device.as_ref().unwrap();
        self.graphics =
            Some(self.create_queue_context(device_ptr.graphics_queue, device_ptr.graphics_family)?);

        if let (Some(q), Some(f)) = (device_ptr.compute_queue, device_ptr.compute_family) {
            self.compute = Some(self.create_queue_context(q, f)?);
        }
        if let (Some(q), Some(f)) = (device_ptr.transfer_queue, device_ptr.transfer_family) {
            self.transfer = Some(self.create_queue_context(q, f)?);
        }

        // Initialize Loaders
        let device_ptr = self.get_device().clone();
        self.dynamic_rendering = Some(device_ptr.dynamic_rendering.clone());
        self.sync2 = Some(device_ptr.sync2.clone());
        self.accel_struct = device_ptr.accel_struct.clone();
        self.rt_pipeline = device_ptr.rt_pipeline.clone();
        self.rt_supported = device_ptr.rt_supported;

        self.swapchain_loader = Some(ash::khr::swapchain::Device::new(
            &self.instance.handle,
            &self.get_device().handle,
        ));

        // Create static descriptor pool for bindless
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 4096,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 4096,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .pool_sizes(&pool_sizes)
            .max_sets(100);

        self.static_descriptor_pool = unsafe {
            self.get_device()
                .handle
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create static descriptor pool: {}", e))?
        };

        // Initialize Bindless Descriptor Set
        self.init_bindless()?;

        info!("Vulkan Backend Initialized");
        Ok(())
    }

    fn create_window(&mut self, desc: WindowDesc) -> Result<WindowHandle, String> {
        crate::window_context::create_window(self, desc)
    }

    fn destroy_window(&mut self, window: WindowHandle) {
        crate::window_context::destroy_window(self, window)
    }

    fn configure_window(
        &mut self,
        window: WindowHandle,
        config: SwapchainConfig,
    ) -> Result<(), String> {
        crate::window_context::configure_window(self, window, config)
    }

    fn set_fullscreen(&mut self, window: WindowHandle, fullscreen: bool) {
        crate::window_context::set_fullscreen(self, window, fullscreen);
    }

    fn poll_events(&mut self) -> Vec<Event> {
        crate::window_context::poll_events(self)
    }

    fn create_image(&mut self, desc: &ImageDesc) -> BackendImage {
        crate::resources::create_image(self, desc)
    }

    fn destroy_image(&mut self, handle: BackendImage) {
        crate::resources::destroy_image(self, handle)
    }

    fn create_buffer(&mut self, desc: &BufferDesc) -> BackendBuffer {
        crate::resources::create_buffer(self, desc)
    }

    fn destroy_buffer(&mut self, handle: BackendBuffer) {
        crate::resources::destroy_buffer(self, handle)
    }

    fn create_sampler(&mut self, desc: &SamplerDesc) -> SamplerHandle {
        crate::resources::create_sampler(self, desc)
    }

    fn destroy_sampler(&mut self, handle: SamplerHandle) {
        crate::resources::destroy_sampler(self, handle)
    }

    fn create_blas(&mut self, info: &BlasCreateInfo) -> BackendAccelerationStructure {
        crate::accel_struct::create_blas(self, info)
    }

    fn destroy_blas(&mut self, handle: BackendAccelerationStructure) {
        crate::accel_struct::destroy_blas(self, handle)
    }

    fn create_tlas(&mut self, info: &TlasCreateInfo) -> BackendAccelerationStructure {
        crate::accel_struct::create_tlas(self, info)
    }

    fn destroy_tlas(&mut self, handle: BackendAccelerationStructure) {
        crate::accel_struct::destroy_tlas(self, handle)
    }

    fn create_graphics_pipeline(&mut self, desc: &GraphicsPipelineCreateInfo) -> BackendPipeline {
        crate::pipeline_cache::create_graphics_pipeline(self, desc)
    }

    fn create_compute_pipeline(&mut self, desc: &ComputePipelineCreateInfo) -> BackendPipeline {
        crate::pipeline_cache::create_compute_pipeline(self, desc)
    }

    fn create_graphics_pipeline_from_baked(
        &mut self,
        baked: &i3_io::pipeline_asset::BakeableGraphicsPipeline,
        reflection: &[u8],
        bytecode: &[u8],
    ) -> BackendPipeline {
        crate::pipeline_cache::create_graphics_pipeline_from_baked(
            self, baked, reflection, bytecode,
        )
    }

    fn create_compute_pipeline_from_baked(
        &mut self,
        reflection: &[u8],
        bytecode: &[u8],
    ) -> BackendPipeline {
        crate::pipeline_cache::create_compute_pipeline_from_baked(self, reflection, bytecode)
    }

    fn upload_buffer(
        &mut self,
        handle: BackendBuffer,
        data: &[u8],
        offset: u64,
    ) -> Result<(), String> {
        crate::resources::upload_buffer(self, handle, data, offset)
    }

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
    ) -> Result<(), String> {
        crate::resources::upload_image(
            self,
            handle,
            data,
            offset_x,
            offset_y,
            data_width,
            data_height,
            mip_level,
            array_layer,
        )
    }

    fn get_bindless_set_handle(&self) -> u64 {
        self.bindless_set_handle
    }

    // --- Handle Registration ---
    fn register_external_image(&mut self, handle: ImageHandle, physical: BackendImage) {
        self.external_to_physical.insert(handle.0.0, physical.0);
    }

    fn register_external_buffer(&mut self, handle: BufferHandle, physical: BackendBuffer) {
        self.external_buffer_to_physical
            .insert(handle.0.0, physical.0);
    }

    /// Wait for the timeline semaphore to reach a specific value on the host (CPU).
    fn wait_for_timeline(&self, value: u64, timeout_ns: u64) -> Result<(), String> {
        let graphics = self.graphics.as_ref().unwrap();
        let semaphores = [graphics.timeline_sem];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe {
            self.get_device()
                .handle
                .wait_semaphores(&wait_info, timeout_ns)
                .map_err(|e| format!("Wait for timeline error: {}", e))
        }
    }

    // --- Transient Resource Management (Pooling) ---

    fn create_transient_image(&mut self, desc: &ImageDesc) -> BackendImage {
        if let Some(pool) = self.transient_image_pool.get_mut(desc) {
            if let Some(id) = pool.pop() {
                // Reset sync state so the oracle treats this as a fresh image each frame.
                // Transitioning from UNDEFINED tells the driver it can discard previous contents,
                // which is the correct semantic for transient render targets.
                if let Some(img) = self.images.get_mut(id) {
                    img.last_layout = ash::vk::ImageLayout::UNDEFINED;
                    img.last_access = ash::vk::AccessFlags2::empty();
                    img.last_stage = ash::vk::PipelineStageFlags2::NONE;
                }
                return BackendImage(id);
            }
        }
        self.create_image(desc)
    }

    fn create_transient_buffer(&mut self, desc: &BufferDesc) -> BackendBuffer {
        if let Some(pool) = self.transient_buffer_pool.get_mut(desc) {
            if let Some(id) = pool.pop() {
                return BackendBuffer(id);
            }
        }
        self.create_buffer(desc)
    }

    fn release_transient_image(&mut self, handle: BackendImage) {
        // Find desc... actually PhysicalImage needs to store desc for pooling
        if let Some(img) = self.images.get(handle.0) {
            let desc = img.desc; // Assuming PhysicalImage has desc
            self.transient_image_pool
                .entry(desc)
                .or_default()
                .push(handle.0);
        }
    }

    fn release_transient_buffer(&mut self, handle: BackendBuffer) {
        if let Some(buf) = self.buffers.get(handle.0) {
            let desc = buf.desc;
            self.transient_buffer_pool
                .entry(desc)
                .or_default()
                .push(handle.0);
        }
    }

    fn garbage_collect(&mut self) {
        if self.device.is_none() {
            return;
        }
        let safe_frame = self.frame_count.saturating_sub(4);
        let device = self.get_device().clone();

        // 1. Buffers
        if !self.dead_buffers.is_empty() {
            let allocator = device.allocator.lock().unwrap();
            let mut i = 0;
            while i < self.dead_buffers.len() {
                if self.dead_buffers[i].0 <= safe_frame {
                    let (_, buffer, mut alloc) = self.dead_buffers.remove(i);
                    unsafe {
                        allocator.destroy_buffer(buffer, &mut alloc);
                    }
                } else {
                    i += 1;
                }
            }
        }

        // 2. Images
        if !self.dead_images.is_empty() {
            let allocator = device.allocator.lock().unwrap();
            let mut i = 0;
            while i < self.dead_images.len() {
                if self.dead_images[i].0 <= safe_frame {
                    let (_, image, view, mut alloc) = self.dead_images.remove(i);
                    unsafe {
                        device.handle.destroy_image_view(view, None);
                        allocator.destroy_image(image, &mut alloc);
                    }
                } else {
                    i += 1;
                }
            }
        }

        // 3. Acceleration Structures
        if !self.dead_accel_structs.is_empty() {
            if let Some(as_loader) = &self.accel_struct {
                let allocator = device.allocator.lock().unwrap();
                let mut i = 0;
                while i < self.dead_accel_structs.len() {
                    if self.dead_accel_structs[i].0 <= safe_frame {
                        let (_, handle, buffer, mut alloc) = self.dead_accel_structs.remove(i);
                        unsafe {
                            as_loader.destroy_acceleration_structure(handle, None);
                            allocator.destroy_buffer(buffer, &mut alloc);
                        }
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // 4. Samplers
        if !self.dead_samplers.is_empty() {
            let mut i = 0;
            while i < self.dead_samplers.len() {
                if self.dead_samplers[i].0 <= safe_frame {
                    let (_, sampler) = self.dead_samplers.remove(i);
                    unsafe {
                        device.handle.destroy_sampler(sampler, None);
                    }
                } else {
                    i += 1;
                }
            }
        }

        // 5. Semaphores
        if !self.dead_semaphores.is_empty() {
            let mut i = 0;
            while i < self.dead_semaphores.len() {
                if self.dead_semaphores[i].0 <= safe_frame {
                    let (_, _id, sem) = self.dead_semaphores.remove(i);
                    unsafe {
                        device.handle.destroy_semaphore(sem, None);
                    }
                } else {
                    i += 1;
                }
            }
        }
    }

    // --- Resource Resolution ---
    fn resolve_image(&self, handle: ImageHandle) -> BackendImage {
        if let Some(&physical) = self.external_to_physical.get(&handle.0.0) {
            BackendImage(physical)
        } else {
            BackendImage(handle.0.0)
        }
    }

    fn resolve_buffer(&self, handle: BufferHandle) -> BackendBuffer {
        if let Some(&physical) = self.external_buffer_to_physical.get(&handle.0.0) {
            BackendBuffer(physical)
        } else {
            BackendBuffer(handle.0.0)
        }
    }

    fn resolve_pipeline(&self, handle: PipelineHandle) -> BackendPipeline {
        BackendPipeline(handle.0.0)
    }

    fn update_bindless_texture(
        &mut self,
        texture: ImageHandle,
        sampler: SamplerHandle,
        index: u32,
        set: u64,
        binding: u32,
    ) {
        crate::descriptors::update_bindless_texture(self, texture, sampler, index, set, binding);
    }

    fn update_bindless_texture_raw(
        &mut self,
        texture: BackendImage,
        sampler: SamplerHandle,
        index: u32,
        set: u64,
        binding: u32,
    ) {
        crate::descriptors::update_bindless_texture_raw(
            self, texture, sampler, index, set, binding,
        );
    }

    fn update_bindless_sampler(&mut self, sampler: SamplerHandle, set: u64, binding: u32) {
        crate::descriptors::update_bindless_sampler(self, sampler, set, binding);
    }

    #[cfg(debug_assertions)]
    fn set_image_name(&mut self, image: BackendImage, name: &str) {
        crate::debug::set_image_name(self, image, name);
    }

    #[cfg(debug_assertions)]
    fn set_buffer_name(&mut self, buffer: BackendBuffer, name: &str) {
        crate::debug::set_buffer_name(self, buffer, name);
    }

    fn get_buffer_address(&self, handle: BackendBuffer) -> u64 {
        if let Some(buf) = self.buffers.get(handle.0) {
            let info = vk::BufferDeviceAddressInfo::default().buffer(buf.buffer);
            unsafe { self.get_device().handle.get_buffer_device_address(&info) }
        } else {
            0
        }
    }
}

impl RenderBackendInternal for VulkanBackend {
    fn begin_frame(&mut self) {
        crate::submission::begin_frame(self);
    }

    fn end_frame(&mut self) {
        self.frame_count += 1;
        crate::submission::end_frame(self);
    }

    fn analyze_frame(
        &mut self,
        passes: &[i3_gfx::graph::types::FlatPass],
    ) -> i3_gfx::graph::sync::SyncPlan {
        // Seed using virtual IDs (SymbolId) matching what FlatPass.image_reads/writes use.
        // The external_to_physical map is populated by resolve_resources_recursive before this call.
        let mut image_states = std::collections::HashMap::new();
        let mut buffer_states = std::collections::HashMap::new();

        for (&virtual_id, &physical_id) in &self.external_to_physical {
            if let Some(img) = self.images.get(physical_id) {
                image_states.insert(
                    virtual_id,
                    i3_gfx::graph::sync::ResourceState {
                        layout: crate::sync_planner::translate_layout_to_abstract(img.last_layout),
                        access: crate::sync_planner::translate_access_to_abstract(img.last_access),
                        stage: crate::sync_planner::translate_stages_to_abstract(img.last_stage),
                        queue_family: img.last_queue_family,
                    },
                );
            }
        }
        for (&virtual_id, &physical_id) in &self.external_buffer_to_physical {
            if let Some(buf) = self.buffers.get(physical_id) {
                buffer_states.insert(
                    virtual_id,
                    i3_gfx::graph::sync::ResourceState {
                        layout: i3_gfx::graph::sync::ImageLayout::Undefined,
                        access: crate::sync_planner::translate_access_to_abstract(buf.last_access),
                        stage: crate::sync_planner::translate_stages_to_abstract(buf.last_stage),
                        queue_family: buf.last_queue_family,
                    },
                );
            }
        }

        let mut planner = i3_gfx::graph::sync_planner::SyncPlanner::new();
        let abstract_plan = planner.analyze(passes, &image_states, &buffer_states);

        // Translate abstract plan to Vulkan barriers
        let vk_plan = crate::sync_planner::translate_plan(self, &abstract_plan, passes);
        self.current_plan = Some(vk_plan);

        // Commit final states back to physical resources.
        // begin_frame() already waited for the previous frame's GPU completion,
        // so these planned-end states are safe to commit now for next-frame seeding.
        for (&virtual_id, &final_state) in &abstract_plan.final_states {
            if let Some(&physical_id) = self.external_to_physical.get(&virtual_id) {
                if let Some(img) = self.images.get_mut(physical_id) {
                    img.last_layout =
                        crate::sync_planner::translate_layout_from_abstract(final_state.layout);
                    img.last_access =
                        crate::sync_planner::translate_access_from_abstract(final_state.access);
                    img.last_stage =
                        crate::sync_planner::translate_stages_from_abstract(final_state.stage);
                    img.last_queue_family = final_state.queue_family;
                }
            }
            if let Some(&physical_id) = self.external_buffer_to_physical.get(&virtual_id) {
                if let Some(buf) = self.buffers.get_mut(physical_id) {
                    buf.last_access =
                        crate::sync_planner::translate_access_from_abstract(final_state.access);
                    buf.last_stage =
                        crate::sync_planner::translate_stages_from_abstract(final_state.stage);
                    buf.last_queue_family = final_state.queue_family;
                }
            }
        }

        abstract_plan
    }

    fn acquire_swapchain_image(
        &mut self,
        window: WindowHandle,
    ) -> Result<Option<(BackendImage, u64, u32)>, String> {
        crate::submission::acquire_swapchain_image(self, window)
    }

    fn submit(&mut self, batch: CommandBatch) -> Result<u64, String> {
        crate::submission::submit(self, batch)
    }

    type PreparedPass = VulkanPreparedPass;

    fn prepare_pass(&mut self, pass_index: usize, desc: PassDescriptor) -> Self::PreparedPass {
        crate::commands::prepare_pass(self, pass_index, desc)
    }

    fn get_prepared_pass_queue(&self, prepared: &Self::PreparedPass) -> QueueType {
        prepared.queue
    }

    fn record_barriers(&mut self, passes: &[&Self::PreparedPass]) -> Option<BackendCommandBuffer> {
        crate::commands::record_barriers(self, passes)
    }

    #[cfg(debug_assertions)]
    fn begin_debug_label(&self, command_buffer: BackendCommandBuffer, name: &str, color: [f32; 4]) {
        let c_name = std::ffi::CString::new(name).unwrap();
        let label = vk::DebugUtilsLabelEXT::default()
            .label_name(&c_name)
            .color(color);
        unsafe {
            let cb = vk::CommandBuffer::from_raw(command_buffer.0);
            self.get_device()
                .debug_utils
                .cmd_begin_debug_utils_label(cb, &label);
        }
    }

    #[cfg(debug_assertions)]
    fn end_debug_label(&self, command_buffer: BackendCommandBuffer) {
        unsafe {
            let cb = vk::CommandBuffer::from_raw(command_buffer.0);
            self.get_device().debug_utils.cmd_end_debug_utils_label(cb);
        }
    }

    fn record_pass(
        &self,
        prepared: &Self::PreparedPass,
        pass: &dyn RenderPass,
    ) -> (
        Option<u64>,
        Option<BackendCommandBuffer>,
        Option<ImageHandle>,
    ) {
        let device = self.get_device().clone();

        // Use Rayon's thread index for command buffer allocation, fallback to 0 if not in a Rayon task.
        let thread_idx = rayon::current_thread_index().unwrap_or(0);

        // Pick queue context based on assigned queue
        let q_ctx = match prepared.queue {
            QueueType::Graphics => self
                .graphics
                .as_ref()
                .expect("CRITICAL: Graphics queue context not initialized during command recording! This should have been caught during backend startup."),
            QueueType::AsyncCompute => self
                .compute
                .as_ref()
                .or(self.graphics.as_ref())
                .expect("Compute queue context not initialized"),
            QueueType::Transfer => self
                .transfer
                .as_ref()
                .or(self.graphics.as_ref())
                .expect("Transfer queue context not initialized"),
        };

        let frame_ctx = &q_ctx.frame_contexts[self.global_frame_index];
        let mut tp = frame_ctx.per_thread_pools[thread_idx % frame_ctx.per_thread_pools.len()]
            .lock()
            .unwrap();

        // Allocate Command Buffer from Thread Pool
        let cmd = if tp.cursor < tp.allocated.len() {
            let cmd = tp.allocated[tp.cursor];
            tp.cursor += 1;
            cmd
        } else {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(tp.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe { device.handle.allocate_command_buffers(&alloc_info).unwrap()[0] };
            tp.allocated.push(cmd);
            tp.cursor += 1;
            cmd
        };

        // Begin Recording
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .handle
                .begin_command_buffer(cmd, &begin_info)
                .expect("Failed to begin Vulkan command buffer")
        };

        #[cfg(debug_assertions)]
        self.begin_debug_label(
            BackendCommandBuffer(cmd.as_raw()),
            &prepared.name,
            [1.0, 1.0, 1.0, 1.0],
        );

        let mut ctx = VulkanPassContext {
            cmd,
            device: self.get_device().clone(),
            present_request: None,
            backend: self as *const Self as *mut Self,
            pipeline: None,
            descriptor_pool: tp.descriptor_pool,
            current_pipeline_layout: vk::PipelineLayout::null(),
            current_bind_point: vk::PipelineBindPoint::GRAPHICS,
            pending_descriptor_sets: prepared.descriptor_sets.clone(),
        };

        // If pipeline is set, determine bind point and bind it
        if let Some(pipe_handle) = prepared.pipeline {
            ctx.bind_pipeline(pipe_handle);
        }

        // (Barriers were already emitted globally via submit_barriers before the pass recording started)

        let is_compute = matches!(prepared.domain, PreparedDomain::Compute);

        if !is_compute {
            if let PreparedDomain::Graphics {
                color_attachments,
                color_count,
                depth_attachment,
            } = &prepared.domain
            {
                if *color_count > 0 || depth_attachment.is_some() {
                    let viewport_extent = prepared.viewport_extent;
                    let viewport = vk::Viewport::default()
                        .x(0.0)
                        .y(viewport_extent.height as f32)
                        .width(viewport_extent.width as f32)
                        .height(-(viewport_extent.height as f32))
                        .min_depth(0.0)
                        .max_depth(1.0);
                    let scissor = vk::Rect2D::default().extent(viewport_extent);

                    let rendering_info = vk::RenderingInfo::default()
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: viewport_extent,
                        })
                        .layer_count(1)
                        .color_attachments(&color_attachments[..*color_count]);

                    let rendering_info = if let Some(depth) = depth_attachment {
                        rendering_info.depth_attachment(depth)
                    } else {
                        rendering_info
                    };

                    unsafe {
                        device.handle.cmd_begin_rendering(cmd, &rendering_info);
                        device.handle.cmd_set_viewport(cmd, 0, &[viewport]);
                        device.handle.cmd_set_scissor(cmd, 0, &[scissor]);
                    }
                }
            }
        }

        pass.execute(&mut ctx);

        if !is_compute {
            if let PreparedDomain::Graphics {
                color_attachments: _,
                color_count,
                depth_attachment,
            } = &prepared.domain
            {
                if *color_count > 0 || depth_attachment.is_some() {
                    unsafe {
                        device.handle.cmd_end_rendering(cmd);
                    }
                }
            }
        }

        // Handle explicit transition for Present if requested
        if let Some(handle) = ctx.present_request {
            let pid = self.resolve_image(handle).0;
            if let Some(img) = self.images.get(pid) {
                let aspect_mask = if img.format == vk::Format::D32_SFLOAT {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                };

                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(img.last_stage)
                    .src_access_mask(img.last_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                    .dst_access_mask(vk::AccessFlags2::empty())
                    .old_layout(img.last_layout)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .image(img.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                let sanitized_barrier = crate::sync::sanitize_image_barrier(
                    barrier,
                    self.graphics_family,
                    self.compute_family,
                    self.transfer_family,
                    prepared.queue,
                );
                let barriers = [sanitized_barrier];
                let dependency_info =
                    vk::DependencyInfo::default().image_memory_barriers(&barriers);
                unsafe {
                    device.handle.cmd_pipeline_barrier2(cmd, &dependency_info);
                }
            }
        }

        #[cfg(debug_assertions)]
        self.end_debug_label(BackendCommandBuffer(cmd.as_raw()));

        unsafe {
            device.handle.end_command_buffer(cmd).unwrap();
        }

        (
            Some(q_ctx.cpu_timeline),
            Some(BackendCommandBuffer(unsafe {
                std::mem::transmute::<vk::CommandBuffer, u64>(cmd)
            })),
            ctx.present_request,
        )
    }

    fn mark_image_as_presented(&mut self, handle: ImageHandle) {
        crate::commands::mark_image_as_presented(self, handle)
    }

    fn allocate_descriptor_set(
        &mut self,
        pipeline: PipelineHandle,
        set_index: u32,
    ) -> Result<DescriptorSetHandle, String> {
        crate::descriptors::allocate_descriptor_set(self, pipeline, set_index)
    }

    fn update_descriptor_set(&mut self, set: DescriptorSetHandle, writes: &[DescriptorWrite]) {
        crate::descriptors::update_descriptor_set(self, set, writes);
    }
}
impl Drop for VulkanBackend {
    fn drop(&mut self) {
        info!("Shutting down Vulkan Backend...");
        if let Some(device) = self.device.clone() {
            unsafe {
                debug!("Waiting for GPU idle...");
                device.handle.device_wait_idle().ok();
                debug!("GPU idle. Starting resource cleanup...");

                // 1. Clean up Window Resources
                // Note: Semaphores are destroyed via self.semaphores loop below.
                // Doing it here would cause double-destruction validation errors.

                debug!("Destroying windows and swapchains...");
                self.windows.clear();

                debug!("Destroying frame contexts...");
                let mut contexts_to_clean = Vec::new();
                if let Some(ctx) = self.graphics.take() {
                    contexts_to_clean.push(ctx);
                }
                if let Some(ctx) = self.compute.take() {
                    contexts_to_clean.push(ctx);
                }
                if let Some(ctx) = self.transfer.take() {
                    contexts_to_clean.push(ctx);
                }

                for q_ctx in contexts_to_clean {
                    device.handle.destroy_semaphore(q_ctx.timeline_sem, None);
                    for ctx in q_ctx.frame_contexts {
                        device.handle.destroy_command_pool(ctx.command_pool, None);
                        device
                            .handle
                            .destroy_descriptor_pool(ctx.descriptor_pool, None);
                        for tp_mutex in ctx.per_thread_pools {
                            let tp = tp_mutex.into_inner().unwrap_or_else(|e| e.into_inner());
                            device.handle.destroy_command_pool(tp.pool, None);
                            device
                                .handle
                                .destroy_descriptor_pool(tp.descriptor_pool, None);
                        }
                    }
                }

                debug!("Destroying pipelines and shaders...");
                for id in self.pipeline_resources.ids() {
                    if let Some(p) = self.pipeline_resources.get(id) {
                        device.handle.destroy_pipeline(p.handle, None);
                        device.handle.destroy_pipeline_layout(p.layout, None);
                    }
                }
                for &s in &self.shader_modules {
                    device.handle.destroy_shader_module(s, None);
                }
                for &dsl in &self.descriptor_set_layouts {
                    device.handle.destroy_descriptor_set_layout(dsl, None);
                }

                debug!("Destroying active buffers and allocations...");
                {
                    let allocator = device.allocator.lock().unwrap();
                    for (_id, physical) in self.buffers.iter_mut() {
                        if let Some(alloc) = physical.allocation.as_mut() {
                            allocator.destroy_buffer(physical.buffer, alloc);
                        }
                    }
                }

                debug!("Destroying active images and views...");
                {
                    let allocator = device.allocator.lock().unwrap();
                    for (_id, physical) in self.images.iter_mut() {
                        if let Some(alloc) = physical.allocation.as_mut() {
                            device.handle.destroy_image_view(physical.view, None);
                            allocator.destroy_image(physical.image, alloc);
                        } else {
                            // This is likely a swapchain image, DO NOT destroy its view/image as it's owned elsewhere.
                            debug!("Skipping destruction of external image {:?}", _id);
                        }
                    }
                }

                debug!("Destroying dead resources...");
                {
                    let allocator = device.allocator.lock().unwrap();
                    for (_, buffer, mut alloc) in self.dead_buffers.drain(..) {
                        allocator.destroy_buffer(buffer, &mut alloc);
                    }
                    for (_, image, view, mut alloc) in self.dead_images.drain(..) {
                        device.handle.destroy_image_view(view, None);
                        allocator.destroy_image(image, &mut alloc);
                    }
                    if let Some(as_loader) = &self.accel_struct {
                        for (_, handle, buffer, mut alloc) in self.dead_accel_structs.drain(..) {
                            as_loader.destroy_acceleration_structure(handle, None);
                            allocator.destroy_buffer(buffer, &mut alloc);
                        }
                    }
                    for (_, sampler) in self.dead_samplers.drain(..) {
                        device.handle.destroy_sampler(sampler, None);
                    }
                    for (_, _, sem_handle) in self.dead_semaphores.drain(..) {
                        device.handle.destroy_semaphore(sem_handle, None);
                    }
                }

                debug!("Destroying acceleration structures...");
                if let Some(as_loader) = &self.accel_struct {
                    let allocator = device.allocator.lock().unwrap();
                    for (_id, pas) in self.accel_structs.iter_mut() {
                        as_loader.destroy_acceleration_structure(pas.handle, None);
                        allocator.destroy_buffer(pas.buffer, &mut pas.allocation);
                    }
                }

                debug!("Destroying semaphores...");
                // Iterate over all remaining semaphores in the arena and destroy them
                let semaphore_ids = self.semaphores.ids();
                for id in semaphore_ids {
                    self.destroy_semaphore_internal(id);
                }
                for &sem in &self.recycled_semaphores {
                    device.handle.destroy_semaphore(sem, None);
                }
                debug!("Destroying samplers...");
                for (_, sampler) in self.samplers.iter() {
                    device.handle.destroy_sampler(*sampler, None);
                }

                debug!("Destroying descriptor pool...");
                if self.static_descriptor_pool != vk::DescriptorPool::null() {
                    device
                        .handle
                        .destroy_descriptor_pool(self.static_descriptor_pool, None);
                }
                if self.bindless_set_layout != vk::DescriptorSetLayout::null() {
                    device
                        .handle
                        .destroy_descriptor_set_layout(self.bindless_set_layout, None);
                }
                info!("Vulkan Backend shutdown complete.");
            }
        }
    }
}
use crate::commands::VulkanPassContext;

impl VulkanBackend {
    pub fn create_semaphore_internal(&mut self) -> Result<(vk::Semaphore, u64), String> {
        let device = self.get_device();
        let semaphore = unsafe {
            device
                .handle
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(|e| e.to_string())?
        };

        let handle = self.semaphores.insert(semaphore);
        Ok((semaphore, handle))
    }

    pub fn destroy_semaphore_internal(&mut self, handle: u64) {
        if let Some(sem) = self.semaphores.remove(handle) {
            self.recycled_semaphores.push(sem);
        }
    }

    pub(crate) fn unregister_swapchain_images(&mut self, images: &[vk::Image]) {
        for &image in images {
            let vk_handle = image.as_raw();
            if let Some(arena_id) = self.external_to_physical.remove(&vk_handle) {
                self.images.remove(arena_id);
            }
        }
    }
}
