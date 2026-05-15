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

use crate::resource_arena::{
    PhysicalAccelerationStructure, PhysicalBuffer, PhysicalImage, PhysicalPipeline, ResourceArena,
};
use crate::window_context::WindowContext;

pub(crate) use crate::commands::{
    PreparedDomain, ThreadCommandPool, VulkanFrameContext, VulkanPreparedPass,
};

use crate::sync;

/// A GPU resource queued for deferred destruction.
///
/// Resources are pushed here when logically destroyed and freed only once
/// `vkGetSemaphoreCounterValue` confirms the GPU has completed past `threshold`.
/// This replaces the old frame_count-4 heuristic with an exact timeline query.
pub enum PendingDelete {
    Buffer {
        threshold: u64,
        buffer: vk::Buffer,
        alloc: vk_mem::Allocation,
    },
    Image {
        threshold: u64,
        image: vk::Image,
        views: Vec<vk::ImageView>,
        alloc: vk_mem::Allocation,
    },
    AccelStruct {
        threshold: u64,
        handle: vk::AccelerationStructureKHR,
        buffer: vk::Buffer,
        alloc: vk_mem::Allocation,
    },
    Sampler {
        threshold: u64,
        sampler: vk::Sampler,
    },
}

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
    pub external_to_physical: HashMap<u64, u64>,        // Virtual ImageID -> Physical ID
    pub external_buffer_to_physical: HashMap<u64, u64>, // Virtual BufferID -> Physical ID
    pub external_as_to_physical: HashMap<u64, u64>,     // Virtual AccelStructID -> Physical ID

    pub frame_count: u64,
    pub pending_deletes: Vec<PendingDelete>,
    pub recycled_semaphores: Vec<vk::Semaphore>,

    pub accel_structs: ResourceArena<PhysicalAccelerationStructure>,

    // Transient Pools
    pub(crate) transient_image_pool: HashMap<ImageDesc, Vec<u64>>,
    pub(crate) transient_buffer_pool: HashMap<BufferDesc, Vec<u64>>,

    // Resources
    pub samplers: ResourceArena<vk::Sampler>,
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

    /// Persistent sync planner — reused every frame so its internal HashMaps are not reallocated.
    pub(crate) sync_planner: i3_gfx::graph::sync_planner::SyncPlanner,
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
            external_as_to_physical: HashMap::new(),
            semaphores: ResourceArena::new(),
            samplers: ResourceArena::new(),
            next_semaphore_id: 1,
            frame_count: 0,
            pending_deletes: Vec::new(),
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

            sync_planner: i3_gfx::graph::sync_planner::SyncPlanner::new(),
            current_plan: None,

            frame_started: false,
            global_frame_index: 0,
            next_resource_id: 1,
            bindless_set_layout: vk::DescriptorSetLayout::null(),
            bindless_set_handle: 0,
            accel_structs: ResourceArena::new(),
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

        let mut pool_sizes = vec![
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

        if device.rt_supported {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 64,
            });
        }

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
                    .descriptor_count(64) // Increased from 1
                    .stage_flags(vk::ShaderStageFlags::ALL),
            ];

            let binding_flags = [
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
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

    fn swapchain_image_count(&self, window: WindowHandle) -> usize {
        self.windows
            .get(&window.0)
            .and_then(|w| w.swapchain.as_ref().map(|s| s.images.len()))
            .unwrap_or(3)
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
        self.external_buffer_to_physical.insert(handle.0.0, physical.0);
    }

    fn register_external_accel_struct(
        &mut self,
        handle: AccelerationStructureHandle,
        physical: BackendAccelerationStructure,
    ) {
        self.external_as_to_physical.insert(handle.0.0, physical.0);
    }

    fn resolve_accel_struct(&self, handle: AccelerationStructureHandle) -> BackendAccelerationStructure {
        if let Some(&physical) = self.external_as_to_physical.get(&handle.0.0) {
            BackendAccelerationStructure(physical)
        } else {
            tracing::warn!("resolve_accel_struct: handle {:?} not found", handle);
            BackendAccelerationStructure(u64::MAX)
        }
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
        let Some(gfx) = self.graphics.as_ref() else { return; };
        // Query the GPU's actual completed timeline value — no fixed-frame heuristic.
        let gpu_done = unsafe {
            self.get_device()
                .handle
                .get_semaphore_counter_value(gfx.timeline_sem)
                .unwrap_or(0)
        };

        // Debug: collect all VkAS handles still alive in the arena so we can assert
        // that gc() never destroys a handle whose arena slot is still occupied.
        #[cfg(debug_assertions)]
        let live_vk_as_handles: std::collections::HashSet<ash::vk::AccelerationStructureKHR> =
            self.accel_structs.iter().map(|(_, pas)| pas.handle).collect();

        let device  = self.get_device().clone();
        let as_ext  = self.accel_struct.clone();
        let allocator = device.allocator.lock().unwrap();

        self.pending_deletes.retain_mut(|item| {
            let threshold = match item {
                PendingDelete::Buffer      { threshold, .. }
                | PendingDelete::Image     { threshold, .. }
                | PendingDelete::AccelStruct { threshold, .. }
                | PendingDelete::Sampler   { threshold, .. } => *threshold,
            };
            if threshold > gpu_done {
                return true; // GPU hasn't reached this point yet — keep
            }
            match item {
                PendingDelete::Buffer { buffer, alloc, .. } => unsafe {
                    allocator.destroy_buffer(*buffer, alloc);
                },
                PendingDelete::Image { image, views, alloc, .. } => unsafe {
                    for &v in views.iter() {
                        device.handle.destroy_image_view(v, None);
                    }
                    allocator.destroy_image(*image, alloc);
                },
                PendingDelete::AccelStruct { handle, buffer, alloc, .. } => unsafe {
                    if let Some(ext) = &as_ext {
                        // INVARIANT: by the time gc() frees a VkAS, destroy_blas() must have
                        // already removed it from the arena (generation bumped). If this fires,
                        // something called vkDestroyAccelerationStructure without going through
                        // destroy_blas — the root cause of the VUID-12281 validation error.
                        #[cfg(debug_assertions)]
                        if live_vk_as_handles.contains(handle) {
                            tracing::error!(
                                "blas_dbg gc BUG: freeing VkAS {handle:?} but its arena slot is still occupied! \
                                 This VkAS will remain referenced in the TLAS — VUID-12281 incoming."
                            );
                        }
                        ext.destroy_acceleration_structure(*handle, None);
                    }
                    allocator.destroy_buffer(*buffer, alloc);
                },
                PendingDelete::Sampler { sampler, .. } => unsafe {
                    device.handle.destroy_sampler(*sampler, None);
                },
            }
            false // consumed — remove from vec
        });
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

    fn update_bindless_sampler(&mut self, sampler: SamplerHandle, index: u32, set: u64, binding: u32) {
        crate::descriptors::update_bindless_sampler(self, sampler, index, set, binding);
    }

    #[cfg(any(debug_assertions, feature = "profiling"))]
    fn set_image_name(&mut self, image: BackendImage, name: &str) {
        crate::debug::set_image_name(self, image, name);
    }

    #[cfg(any(debug_assertions, feature = "profiling"))]
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

    fn reset_frame_resources(&mut self) {
        // SymbolIds are minted from a global atomic counter — each frame produces unique IDs.
        // Without this clear, the maps grow unboundedly (N_frames × N_resources entries),
        // making analyze_frame's iteration O(all-time resources) instead of O(current frame).
        self.external_to_physical.clear();
        self.external_buffer_to_physical.clear();
        self.external_as_to_physical.clear();
    }

    fn end_frame(&mut self) {
        crate::submission::end_frame(self);
    }

    fn analyze_frame(
        &mut self,
        passes: &[i3_gfx::graph::types::FlatPass],
    ) -> i3_gfx::graph::sync::SyncPlan {
        // Populate seed maps directly on the planner — no per-frame HashMap allocation.
        // The external_to_physical maps are cleared in begin_frame and repopulated by
        // process_externals_recursive before this call, so they contain exactly the
        // resources active this frame.
        self.sync_planner.image_seed.clear();
        self.sync_planner.buffer_seed.clear();
        for (&virtual_id, &physical_id) in &self.external_to_physical {
            if let Some(img) = self.images.get(physical_id) {
                self.sync_planner.image_seed.insert(
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
                self.sync_planner.buffer_seed.insert(
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

        let abstract_plan = self.sync_planner.analyze(passes);

        // Translate abstract plan to Vulkan barriers
        let vk_plan = crate::sync_planner::translate_plan(self, &abstract_plan);
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

    #[cfg(any(debug_assertions, feature = "profiling"))]
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

    #[cfg(any(debug_assertions, feature = "profiling"))]
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
        frame_data: &i3_gfx::graph::compiler::FrameBlackboard,
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

        #[cfg(any(debug_assertions, feature = "profiling"))]
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
            is_structural: false,
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

        pass.execute(&mut ctx, frame_data);
        
        // Skip purely structural nodes if they have no work to do (no barriers, no present).
        let has_sync = !prepared.sync.pre_barriers.is_empty() || !prepared.sync.post_barriers.is_empty();
        if ctx.is_structural && ctx.present_request.is_none() && !has_sync {
            tracing::debug!("record_pass: skipping structural/empty pass '{}'", pass.name());
            return (None, None, None);
        }

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

        // Emit post-barriers (e.g. present transition: final layout → PresentSrc).
        // These run after execute() so they see the image in its post-write state.
        if !prepared.sync.post_barriers.is_empty() {
            let mut img_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();
            for b in &prepared.sync.post_barriers {
                if let crate::sync::Barrier::Image(b) = b { img_barriers.push(b.clone()); }
            }
            if !img_barriers.is_empty() {
                let dep = vk::DependencyInfo::default().image_memory_barriers(&img_barriers);
                unsafe { device.handle.cmd_pipeline_barrier2(cmd, &dep); }
            }
        }

        #[cfg(any(debug_assertions, feature = "profiling"))]
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

                {
                    let allocator = device.allocator.lock().unwrap();

                    debug!("Destroying active acceleration structures...");
                    for (_id, pas) in self.accel_structs.iter_mut() {
                        if let Some(as_ext) = &self.accel_struct {
                            if pas.handle != vk::AccelerationStructureKHR::null() {
                                as_ext.destroy_acceleration_structure(pas.handle, None);
                                pas.handle = vk::AccelerationStructureKHR::null();
                            }
                        }
                        if let Some(mut alloc) = pas.allocation.take() {
                            allocator.destroy_buffer(pas.buffer, &mut alloc);
                        }
                    }

                    debug!("Destroying active buffers...");
                    for (_id, physical) in self.buffers.iter_mut() {
                        if let Some(mut alloc) = physical.allocation.take() {
                            allocator.destroy_buffer(physical.buffer, &mut alloc);
                        }
                    }

                    debug!("Destroying active images...");
                    for (_id, physical) in self.images.iter_mut() {
                        if let Some(mut alloc) = physical.allocation.take() {
                            // Destroy main view
                            if physical.view != vk::ImageView::null() {
                                device.handle.destroy_image_view(physical.view, None);
                                physical.view = vk::ImageView::null();
                            }
                            
                            // Destroy all subresource views (mips/layers)
                            let mut views = physical.subresource_views.lock().unwrap();
                            for (_, view) in views.drain() {
                                device.handle.destroy_image_view(view, None);
                            }
                            
                            allocator.destroy_image(physical.image, &mut alloc);
                        }
                    }

                    debug!("Destroying pending-delete resources...");
                    let as_ext = self.accel_struct.clone();
                    for item in self.pending_deletes.drain(..) {
                        match item {
                            PendingDelete::Buffer { buffer, mut alloc, .. } => {
                                allocator.destroy_buffer(buffer, &mut alloc);
                            },
                            PendingDelete::Image { image, views, mut alloc, .. } => {
                                for view in views {
                                    device.handle.destroy_image_view(view, None);
                                }
                                allocator.destroy_image(image, &mut alloc);
                            },
                            PendingDelete::AccelStruct { handle, buffer, mut alloc, .. } => {
                                if let Some(ext) = &as_ext {
                                    ext.destroy_acceleration_structure(handle, None);
                                }
                                allocator.destroy_buffer(buffer, &mut alloc);
                            },
                            PendingDelete::Sampler { sampler, .. } => {
                                device.handle.destroy_sampler(sampler, None);
                            },
                        }
                    }
                }

                debug!("Destroying samplers...");
                for (_, &sampler) in self.samplers.iter() {
                    device.handle.destroy_sampler(sampler, None);
                }

                debug!("Destroying semaphores...");
                for id in self.semaphores.ids() {
                    self.destroy_semaphore_internal(id);
                }
                for &sem in &self.recycled_semaphores {
                    device.handle.destroy_semaphore(sem, None);
                }

                debug!("Destroying descriptor pools...");
                if self.static_descriptor_pool != vk::DescriptorPool::null() {
                    device.handle.destroy_descriptor_pool(self.static_descriptor_pool, None);
                }
                if self.bindless_set_layout != vk::DescriptorSetLayout::null() {
                    device.handle.destroy_descriptor_set_layout(self.bindless_set_layout, None);
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
