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

pub(crate) use crate::commands::{ThreadCommandPool, VulkanFrameContext, PreparedDomain, VulkanPreparedPass};

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
    pub dead_accel_structs: Vec<(u64, vk::AccelerationStructureKHR, vk::Buffer, vk_mem::Allocation)>,

    // Transient Pools
    pub(crate) transient_image_pool: HashMap<ImageDesc, Vec<u64>>,
    pub(crate) transient_buffer_pool: HashMap<BufferDesc, Vec<u64>>,

    // Resources
    pub samplers: ResourceArena<vk::Sampler>,
    pub dead_samplers: Vec<(u64, vk::Sampler)>, // Frame, Handle
    pub semaphores: ResourceArena<vk::Semaphore>,
    pub timeline_sem: vk::Semaphore, // Global timeline for graphics queue
    pub cpu_timeline: u64,           // Current CPU submission value
    pub next_semaphore_id: u64,
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

    // Scratch Buffers for hot path
    pub(crate) target_id_scratch: Vec<u64>,
    pub(crate) image_barrier_scratch: Vec<vk::ImageMemoryBarrier2<'static>>,
    pub(crate) buffer_barrier_scratch: Vec<vk::BufferMemoryBarrier2<'static>>,

    // Global Frame Contexts
    pub(crate) frame_contexts: Vec<VulkanFrameContext>,
    pub(crate) frame_started: bool,
    pub(crate) global_frame_index: usize,
    pub next_resource_id: u64,
    pub bindless_set_layout: vk::DescriptorSetLayout,
    pub bindless_set_handle: u64,
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
            transient_image_pool: HashMap::new(),
            transient_buffer_pool: HashMap::new(),
            static_descriptor_pool: vk::DescriptorPool::null(),
            descriptor_set_layouts: Vec::new(),
            descriptor_sets: std::sync::Mutex::new(ResourceArena::new()),
            descriptor_pool_max_sets: 1000,
            timeline_sem: vk::Semaphore::null(), // Will be initialized below
            cpu_timeline: 0,
            frame_contexts: Vec::new(),
            swapchain_loader: None,
            dynamic_rendering: None,
            sync2: None,
            accel_struct: None,
            rt_pipeline: None,
            rt_supported: false,
            target_id_scratch: Vec::with_capacity(32),
            image_barrier_scratch: Vec::with_capacity(32),
            buffer_barrier_scratch: Vec::with_capacity(32),
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
    ) -> Option<vk::ImageMemoryBarrier2<'static>> {
        crate::sync::get_image_barrier(self, physical_id, new_layout, dst_access, dst_stage)
    }

    pub fn get_buffer_barrier(
        &mut self,
        physical_id: u64,
        dst_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
    ) -> Option<vk::BufferMemoryBarrier2<'static>> {
        crate::sync::get_buffer_barrier(self, physical_id, dst_access, dst_stage)
    }

    pub fn get_image_state(
        &self,
        usage: ResourceUsage,
        is_write: bool,
        bind_point: vk::PipelineBindPoint,
    ) -> (vk::ImageLayout, vk::AccessFlags2, vk::PipelineStageFlags2) {
        crate::sync::get_image_state(usage, is_write, bind_point)
    }

    pub fn get_buffer_state(
        &self,
        usage: ResourceUsage,
        bind_point: vk::PipelineBindPoint,
    ) -> (vk::AccessFlags2, vk::PipelineStageFlags2) {
        crate::sync::get_buffer_state(usage, bind_point)
    }

    #[allow(dead_code)]
    pub fn create_semaphore(&mut self) -> u64 {
        crate::sync::create_semaphore(self)
    }

    pub fn next_id(&mut self) -> u64 {
        let id = self.next_resource_id;
        self.next_resource_id += 1;
        id
    }

    pub fn window_size(&self, window: WindowHandle) -> Option<(u32, u32)> {
        crate::window_context::window_size(self, window)
    }

    fn init_frame_contexts(&mut self) -> Result<(), String> {
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
                            .queue_family_index(device.graphics_family)
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
                                .queue_family_index(device.graphics_family)
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
        self.frame_contexts = frame_contexts;
        Ok(())
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
        DeviceCapabilities {
            ray_tracing: self.get_device().rt_supported,
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

        let device = crate::device::VulkanDevice::new_with_physical(
            self.instance.clone(),
            physical_device,
        )?;
        self.device = Some(Arc::new(device));
        self.event_pump = Some(self.sdl.event_pump()?);

        // Create Timeline Semaphore
        let mut type_info =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        self.timeline_sem = unsafe {
            self.get_device()
                .handle
                .create_semaphore(&create_info, None)
                .map_err(|e| format!("Failed to create timeline semaphore: {}", e))?
        };

        // Create Descriptor Pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1000,
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
        self.static_descriptor_pool = unsafe {
            self.get_device()
                .handle
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create static descriptor pool: {}", e))?
        };

        // Create Global Frame Contexts
        self.init_frame_contexts()?;

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
        crate::pipeline_cache::create_graphics_pipeline_from_baked(self, baked, reflection, bytecode)
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
        let semaphores = [self.timeline_sem];
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
        crate::submission::end_frame(self);
    }

    fn acquire_swapchain_image(
        &mut self,
        window: WindowHandle,
    ) -> Result<Option<(BackendImage, u64, u32)>, String> {
        crate::submission::acquire_swapchain_image(self, window)
    }

    fn submit(
        &mut self,
        batch: CommandBatch,
        wait_sems: &[u64],
        signal_sems: &[u64],
    ) -> Result<u64, String> {
        crate::submission::submit(self, batch, wait_sems, signal_sems)
    }

    type PreparedPass = VulkanPreparedPass;

    fn prepare_pass(&mut self, desc: PassDescriptor<'_>) -> Self::PreparedPass {
        // Clear scratch vectors for this pass
        self.image_barrier_scratch.clear();
        self.buffer_barrier_scratch.clear();

        // Resolve target physical IDs from writes (Using scratch)
        self.target_id_scratch.clear();
        for (handle, _) in desc.image_writes {
            let pid = if let Some(&p) = self.external_to_physical.get(&handle.0.0) {
                p
            } else {
                handle.0.0
            };
            self.target_id_scratch.push(pid);
        }

        // Identify Target Window & Extent (for Viewport/Pool)
        let mut viewport_extent = vk::Extent2D {
            width: 800,
            height: 600,
        }; // Fallback

        if let Some(&first_pid) = self.target_id_scratch.first() {
            if let Some(img) = self.images.get(first_pid) {
                viewport_extent = vk::Extent2D {
                    width: img.desc.width,
                    height: img.desc.height,
                };
            }
            // Fast window lookup (Match Arena ID)
            for ctx_win in self.windows.values() {
                if let (Some(sc), Some(idx)) = (&ctx_win.swapchain, ctx_win.current_image_index) {
                    let sc_handle = sc.images[idx as usize].as_raw();
                    if let Some(&sc_arena_id) = self.external_to_physical.get(&sc_handle) {
                        if sc_arena_id == first_pid {
                            viewport_extent = sc.extent;
                            break;
                        }
                    }
                }
            }
        }

        // Infer domain from pipeline bind point (no user-declared domain)
        let is_compute = if let Some(h) = desc.pipeline {
            self.pipeline_resources
                .get(h.0.0)
                .map(|p| p.bind_point == vk::PipelineBindPoint::COMPUTE)
                .unwrap_or(false)
        } else {
            false
        };

        let current_bind_point = if is_compute {
            vk::PipelineBindPoint::COMPUTE
        } else {
            vk::PipelineBindPoint::GRAPHICS
        };

        // Prepare attachments
        // --- Unified Resource Synchronization & Attachment Discovery ---
        let mut color_attachments = [vk::RenderingAttachmentInfo::default(); 8];
        let mut color_count = 0;
        let mut depth_attachment_info = None;

        // Dedup and merge usages for all images while preserving order
        let mut pass_images_order = Vec::new();
        let mut pass_images_map: HashMap<ImageHandle, (ResourceUsage, bool)> = HashMap::new();

        for (handle, usage) in desc.image_writes {
            if !pass_images_map.contains_key(handle) {
                pass_images_order.push(*handle);
            }
            pass_images_map.insert(*handle, (*usage, true));
        }
        for (handle, usage) in desc.image_reads {
            if !pass_images_map.contains_key(handle) {
                pass_images_order.push(*handle);
                pass_images_map.insert(*handle, (*usage, false));
            } else {
                let entry = pass_images_map.get_mut(handle).unwrap();
                entry.0 |= *usage;
            }
        }

        // Synchronize and collect attachments in deterministic order
        for handle in pass_images_order {
            let (usage, is_write) = pass_images_map[&handle];
            let pid = self.resolve_image(handle).0;
            let (target_layout, target_access, target_stage) =
                self.get_image_state(usage, is_write, current_bind_point);

            if let Some(barrier) =
                self.get_image_barrier(pid, target_layout, target_access, target_stage)
            {
                self.image_barrier_scratch.push(barrier);
            }

            if usage.intersects(ResourceUsage::COLOR_ATTACHMENT | ResourceUsage::DEPTH_STENCIL) {
                let img_info = if let Some(img) = self.images.get(pid) {
                    (img.format, img.view)
                } else {
                    continue;
                };

                let load_op = if is_write {
                    if let Some(img) = self.images.get_mut(pid) {
                        if img.last_write_frame < self.frame_count {
                            img.last_write_frame = self.frame_count;
                            vk::AttachmentLoadOp::CLEAR
                        } else {
                            vk::AttachmentLoadOp::LOAD
                        }
                    } else {
                        vk::AttachmentLoadOp::LOAD
                    }
                } else {
                    vk::AttachmentLoadOp::LOAD
                };

                let clear_value = if usage.intersects(ResourceUsage::DEPTH_STENCIL) {
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    }
                } else {
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                };

                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(img_info.1)
                    .image_layout(target_layout)
                    .load_op(load_op)
                    .store_op(if is_write {
                        vk::AttachmentStoreOp::STORE
                    } else {
                        vk::AttachmentStoreOp::NONE
                    })
                    .clear_value(clear_value);

                if usage.intersects(ResourceUsage::DEPTH_STENCIL) {
                    depth_attachment_info = Some(attachment);
                } else if color_count < 8 {
                    color_attachments[color_count] = attachment;
                    color_count += 1;
                }
            }
        }

        // Deduplicate and synchronize buffers while preserving order
        let mut pass_buffers_order = Vec::new();
        let mut pass_buffers_map: HashMap<BufferHandle, ResourceUsage> = HashMap::new();
        for (handle, usage) in desc.buffer_writes {
            if !pass_buffers_map.contains_key(handle) {
                pass_buffers_order.push(*handle);
            }
            pass_buffers_map.insert(*handle, *usage);
        }
        for (handle, usage) in desc.buffer_reads {
            if !pass_buffers_map.contains_key(handle) {
                pass_buffers_order.push(*handle);
                pass_buffers_map.insert(*handle, *usage);
            } else {
                let entry = pass_buffers_map.get_mut(handle).unwrap();
                *entry |= *usage;
            }
        }

        for handle in pass_buffers_order {
            let usage = pass_buffers_map[&handle];
            let pid = self.resolve_buffer(handle).0;
            let (target_access, target_stage) = self.get_buffer_state(usage, current_bind_point);
            if let Some(barrier) = self.get_buffer_barrier(pid, target_access, target_stage) {
                self.buffer_barrier_scratch.push(barrier);
            }
        }

        let domain = if is_compute {
            PreparedDomain::Compute
        } else {
            PreparedDomain::Graphics {
                color_attachments,
                color_count,
                depth_attachment: depth_attachment_info,
            }
        };

        VulkanPreparedPass {
            name: desc.name.to_string(),
            domain,
            pipeline: desc.pipeline,
            viewport_extent,
            image_barriers: self.image_barrier_scratch.clone(),
            buffer_barriers: self.buffer_barrier_scratch.clone(),
            descriptor_sets: desc.descriptor_sets.to_vec(),
        }
    }

    fn record_barriers(&self, passes: &[&Self::PreparedPass]) -> Option<BackendCommandBuffer> {
        let mut total_image_barriers = 0;
        let mut total_buffer_barriers = 0;
        for p in passes {
            total_image_barriers += p.image_barriers.len();
            total_buffer_barriers += p.buffer_barriers.len();
        }

        if total_image_barriers == 0 && total_buffer_barriers == 0 {
            return None;
        }

        let device = self.get_device().clone();
        let thread_idx = rayon::current_thread_index().unwrap_or(0);
        let frame_ctx = &self.frame_contexts[self.global_frame_index];
        let mut tp = frame_ctx.per_thread_pools[thread_idx % frame_ctx.per_thread_pools.len()].lock().unwrap();

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
                .unwrap();
        }

        let mut all_image_barriers = Vec::with_capacity(total_image_barriers);
        let mut all_buffer_barriers = Vec::with_capacity(total_buffer_barriers);
        for p in passes {
            all_image_barriers.extend_from_slice(&p.image_barriers);
            all_buffer_barriers.extend_from_slice(&p.buffer_barriers);
        }

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(&all_image_barriers)
            .buffer_memory_barriers(&all_buffer_barriers);

        unsafe {
            device.handle.cmd_pipeline_barrier2(cmd, &dependency_info);
            device.handle.end_command_buffer(cmd).unwrap();
        }

        Some(BackendCommandBuffer(unsafe {
            std::mem::transmute::<vk::CommandBuffer, u64>(cmd)
        }))
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

        let thread_idx = rayon::current_thread_index().unwrap_or(0);
        let frame_ctx = &self.frame_contexts[self.global_frame_index];
        let mut tp = frame_ctx.per_thread_pools[thread_idx % frame_ctx.per_thread_pools.len()].lock().unwrap();

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
                .unwrap()
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

                let barriers = [barrier];
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
            Some(self.cpu_timeline),
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
                for ctx in &self.frame_contexts {
                    device.handle.destroy_command_pool(ctx.command_pool, None);
                    device
                        .handle
                        .destroy_descriptor_pool(ctx.descriptor_pool, None);

                    for tp_mutex in &ctx.per_thread_pools {
                        let tp = tp_mutex.lock().unwrap();
                        device.handle.destroy_command_pool(tp.pool, None);
                        device
                            .handle
                            .destroy_descriptor_pool(tp.descriptor_pool, None);
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
                if self.timeline_sem != vk::Semaphore::null() {
                    device.handle.destroy_semaphore(self.timeline_sem, None);
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
            unsafe {
                self.get_device().handle.destroy_semaphore(sem, None);
            }
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
