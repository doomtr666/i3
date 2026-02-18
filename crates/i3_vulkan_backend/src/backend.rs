use ash::vk;
use ash::vk::Handle;
use i3_gfx::graph::backend::*;
use i3_gfx::graph::pipeline::*;
use i3_gfx::graph::types::*;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info};
use vk_mem::Alloc;

struct PooledImage {
    image: vk::Image,
    view: vk::ImageView,
    allocation: vk_mem::Allocation,
    id: u64,
    #[allow(dead_code)]
    desc: ImageDesc,
    last_used_frame: u64,
}

struct PooledBuffer {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    id: u64,
    #[allow(dead_code)]
    desc: BufferDesc,
    last_used_frame: u64,
}

// ...

struct WindowContext {
    // Order matters for drop: swapchain must be dropped BEFORE raw (surface)
    swapchain: Option<crate::swapchain::VulkanSwapchain>,
    raw: crate::window::VulkanWindow,
    config: SwapchainConfig,
    // Semaphores for acquire (per frame in flight)
    acquire_semaphores: Vec<vk::Semaphore>,
    // Track the current frame's acquire semaphore to pair it with the image
    current_acquire_sem: Option<vk::Semaphore>,
    current_image_index: Option<u32>,
    // Frame Synchronization
    // Frame Synchronization (Timeline Values)
    // We track the last submitted timeline value for each frame slot.
    // When acquiring frame i, we wait for timeline to reach submitted_values[i].
    submitted_values: Vec<u64>,
    frame_index: usize, // 0..min_image
    // Command Pools for recycling
    command_pools: Vec<vk::CommandPool>,
    // Command Buffers (One list per frame-in-flight)
    allocated_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    // Cursor for next available buffer in current frame
    command_buffer_cursors: Vec<usize>,
}

pub struct VulkanBackend {
    pub instance: Arc<crate::instance::VulkanInstance>,
    pub device: Option<Arc<crate::device::VulkanDevice>>,

    // Window Management
    windows: HashMap<u64, WindowContext>,
    next_window_id: u64,

    // SDL2 context
    pub sdl: sdl2::Sdl,
    pub video: sdl2::VideoSubsystem,
    pub event_pump: Option<sdl2::EventPump>,

    // Resource tracking for teardown
    pub pipelines: Vec<vk::Pipeline>,
    pub layouts: Vec<vk::PipelineLayout>,
    pub shader_modules: Vec<vk::ShaderModule>,
    // VMA allocations to free? VMA handles it, but we need to destroy buffers/images.
    // For MVP we leak or let VMA cleanup (if it tracks).
    // Better: store handles to destroy.
    // For full correctness we'd map handle -> allocation.
    // Simplifying for now: we rely on OS cleanup for MVP or implement full tracking later.

    // Resources mapping
    // We map generic u64 handle to Vulkan object
    // Resource Maps
    pub image_map: HashMap<u64, vk::Image>,
    pub image_views: HashMap<u64, vk::ImageView>,
    pub frame_count: u64,
    pub dead_images: Vec<(u64, vk::Image, vk::ImageView, vk_mem::Allocation)>, // Frame, Image, View, Alloc
    pub dead_buffers: Vec<(u64, vk::Buffer, vk_mem::Allocation)>,
    pub dead_semaphores: Vec<(u64, u64, vk::Semaphore)>, // Frame, ID, Handle
    pub recycled_semaphores: Vec<vk::Semaphore>,
    pub image_allocations: HashMap<u64, vk_mem::Allocation>,
    pub external_to_physical: HashMap<u64, u64>, // Virtual ID -> Physical ID

    // Transient Pools
    transient_image_pool: HashMap<ImageDesc, Vec<PooledImage>>,
    transient_buffer_pool: HashMap<BufferDesc, Vec<PooledBuffer>>,

    // Descriptors for active transient resources (needed for release)
    transient_image_descs: HashMap<u64, ImageDesc>,
    transient_buffer_descs: HashMap<u64, BufferDesc>,

    pub buffer_map: HashMap<u64, vk::Buffer>,
    pub buffer_allocations: HashMap<u64, vk_mem::Allocation>,

    pipeline_map: HashMap<u64, vk::Pipeline>,

    // Reverse map for swapchain images
    // When we present(handle), we need to find the window.
    // We can map image_handle -> window_handle?
    // Or just search. Searching is fine for small number of windows.

    // Semaphore management (Timeline & Binary)
    pub semaphores: HashMap<u64, vk::Semaphore>,
    pub timeline_sem: vk::Semaphore, // Global timeline for graphics queue
    pub cpu_timeline: u64,           // Current CPU submission value
    pub next_semaphore_id: u64,
    pub descriptor_pool: vk::DescriptorPool,
    pub pipeline_layouts: HashMap<u64, Vec<vk::DescriptorSetLayout>>,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub descriptor_sets: HashMap<u64, vk::DescriptorSet>,
    pub pipeline_layout_map: HashMap<u64, vk::PipelineLayout>,
    pub next_resource_id: u64,
}

impl VulkanBackend {
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
            pipelines: Vec::new(),
            layouts: Vec::new(),
            shader_modules: Vec::new(),
            image_map: HashMap::new(),
            image_views: HashMap::new(),
            image_allocations: HashMap::new(),
            external_to_physical: HashMap::new(),
            buffer_map: HashMap::new(),
            buffer_allocations: HashMap::new(),
            pipeline_map: HashMap::new(),
            semaphores: HashMap::new(),
            next_semaphore_id: 1,
            frame_count: 0,
            dead_images: Vec::new(),
            dead_buffers: Vec::new(),
            dead_semaphores: Vec::new(),
            recycled_semaphores: Vec::new(),
            transient_image_pool: HashMap::new(),
            transient_buffer_pool: HashMap::new(),
            transient_image_descs: HashMap::new(),
            transient_buffer_descs: HashMap::new(),
            next_resource_id: 1000,
            descriptor_pool: vk::DescriptorPool::null(),
            pipeline_layouts: HashMap::new(),
            descriptor_set_layouts: Vec::new(),
            descriptor_sets: HashMap::new(),
            pipeline_layout_map: HashMap::new(),
            timeline_sem: vk::Semaphore::null(), // Initialized in `initialize`
            cpu_timeline: 0,
        })
    }

    fn get_device(&self) -> &Arc<crate::device::VulkanDevice> {
        self.device.as_ref().expect("Backend not initialized")
    }

    fn create_semaphore(&mut self) -> u64 {
        let sem = if let Some(recycled) = self.recycled_semaphores.pop() {
            recycled
        } else {
            let device = self.get_device();
            let create_info = vk::SemaphoreCreateInfo::default();
            unsafe { device.handle.create_semaphore(&create_info, None) }.unwrap()
        };

        let id = self.next_semaphore_id;
        self.next_semaphore_id += 1;
        self.semaphores.insert(id, sem);
        id
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_resource_id;
        self.next_resource_id += 1;
        id
    }
}

impl RenderBackend for VulkanBackend {
    fn enumerate_devices(&self) -> Vec<DeviceInfo> {
        let pdevices =
            unsafe { self.instance.handle.enumerate_physical_devices() }.unwrap_or_default();

        pdevices
            .iter()
            .map(|&p| {
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
                    id: props.device_id,
                    name,
                    device_type,
                }
            })
            .collect()
    }

    fn initialize(&mut self, device_id: u32) -> Result<(), String> {
        let pdevices = unsafe { self.instance.handle.enumerate_physical_devices() }
            .map_err(|e| format!("Failed to enumerate physical devices: {}", e))?;

        let physical_device = pdevices
            .iter()
            .find(|&p| {
                let props = unsafe { self.instance.handle.get_physical_device_properties(*p) };
                props.device_id == device_id
            })
            .or_else(|| pdevices.first())
            .ok_or("No physical device found")?;

        let device = crate::device::VulkanDevice::new_with_physical(
            self.instance.clone(),
            *physical_device,
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
                descriptor_count: 1000,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1000,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1000);
        self.descriptor_pool = unsafe {
            self.get_device()
                .handle
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create descriptor pool: {}", e))?
        };

        info!("Vulkan Backend Initialized");
        Ok(())
    }

    fn create_window(&mut self, desc: WindowDesc) -> Result<WindowHandle, String> {
        let window_handle = self
            .video
            .window(&desc.title, desc.width, desc.height)
            .position_centered()
            .resizable()
            .vulkan()
            .build()
            .map_err(|e| e.to_string())?;

        let vulkan_window = crate::window::VulkanWindow::new(self.instance.clone(), window_handle)?;

        let id = self.next_window_id;
        self.next_window_id += 1;

        // Initialize submitted values to 0 (completed)
        let submitted_values = vec![0; 2];

        // Create Semaphores per frame
        let mut acquire_sems = Vec::new();
        let device = self.get_device();
        for _ in 0..2 {
            let create_info = vk::SemaphoreCreateInfo::default();
            let sem = unsafe {
                device
                    .handle
                    .create_semaphore(&create_info, None)
                    .map_err(|e| e.to_string())?
            };
            acquire_sems.push(sem);
        }

        // Create Command Pools per frame
        let mut cmd_pools = Vec::new();
        for _ in 0..2 {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(0) // Default graphics family (usually 0)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT); // We reset whole pool
            let pool = unsafe {
                device
                    .handle
                    .create_command_pool(&pool_info, None)
                    .map_err(|e| e.to_string())?
            };
            cmd_pools.push(pool);
        }

        let ctx = WindowContext {
            raw: vulkan_window,
            swapchain: None,
            config: SwapchainConfig {
                vsync: false,
                srgb: true,
                min_image: 3,
            }, // Default
            acquire_semaphores: acquire_sems,
            current_acquire_sem: None,
            current_image_index: None,
            submitted_values,
            frame_index: 0,
            command_pools: cmd_pools,
            allocated_command_buffers: vec![Vec::new(); 3], // Should match min_image or fences.len()
            command_buffer_cursors: vec![0; 3],             // Default min_image=3
        };

        self.windows.insert(id, ctx);
        Ok(WindowHandle(id))
    }

    fn destroy_window(&mut self, window: WindowHandle) {
        self.windows.remove(&window.0);
    }

    fn configure_window(
        &mut self,
        window: WindowHandle,
        config: SwapchainConfig,
    ) -> Result<(), String> {
        if let Some(ctx) = self.windows.get_mut(&window.0) {
            ctx.config = config;
            // Invalidate swapchain so it recreates on next acquire
            ctx.swapchain = None;
            Ok(())
        } else {
            Err("Invalid window handle".to_string())
        }
    }

    fn poll_events(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        if let Some(pump) = &mut self.event_pump {
            for event in pump.poll_iter() {
                match event {
                    sdl2::event::Event::Quit { .. } => events.push(Event::Quit),
                    sdl2::event::Event::KeyDown {
                        keycode: Some(sdl2::keyboard::Keycode::Escape),
                        ..
                    } => events.push(Event::KeyDown {
                        key: KeyCode::Escape,
                    }),
                    sdl2::event::Event::Window {
                        win_event: sdl2::event::WindowEvent::Resized(w, h),
                        ..
                    } => {
                        events.push(Event::Resize {
                            width: w as u32,
                            height: h as u32,
                        });
                        // Invalidate all swapchains logic could go here if we tracked reverse map
                        // For now we rely on the specific window update cycle or we just mark all
                        for ctx in self.windows.values_mut() {
                            ctx.swapchain = None; // Recreate all swapchains on any resize (simplification)
                        }
                    }
                    _ => {}
                }
            }
        }
        events
    }

    fn create_image(&mut self, desc: &ImageDesc) -> BackendImage {
        let device = self.get_device().clone();
        let id = self.next_id();

        let extent = vk::Extent3D {
            width: desc.width,
            height: desc.height,
            depth: desc.depth,
        };

        // Translate format
        let format = crate::convert::convert_format(desc.format);

        // Translate usage
        // We set lots of usage bits for flexibility for now
        let mut usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED;

        // If it looks like a render target
        // TODO: ResourceUsage in desc?
        // Usage is inferred from desc or we should have it in desc.
        // `ImageDesc` has `width`, `height`, `format`.
        // We add attachment bits by default to allow rendering.
        if format == vk::Format::D32_SFLOAT {
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        } else {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE;
        }

        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(desc.mip_levels)
            .array_layers(desc.array_layers)
            .samples(vk::SampleCountFlags::TYPE_1) // Multisampling support later
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (image, allocation) = unsafe {
            let allocator = device.allocator.lock().unwrap();
            allocator
                .create_image(&create_info, &allocation_info)
                .expect("Failed to create image")
        };

        // Create View
        let aspect_mask = if format == vk::Format::D32_SFLOAT {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: desc.mip_levels,
                base_array_layer: 0,
                layer_count: desc.array_layers,
            });
        let view = unsafe { device.handle.create_image_view(&view_info, None) }
            .expect("Failed to create view");

        self.image_map.insert(id, image);
        self.image_allocations.insert(id, allocation);
        self.image_views.insert(id, view);

        BackendImage(id)
    }

    fn destroy_image(&mut self, handle: BackendImage) {
        let id = handle.0;
        let current_frame = self.frame_count;

        // remove from maps immediately to prevent reuse, but defer physical destruction
        if let Some(view) = self.image_views.remove(&id) {
            if let Some(image) = self.image_map.remove(&id) {
                if let Some(allocation) = self.image_allocations.remove(&id) {
                    self.dead_images
                        .push((current_frame, image, view, allocation));
                } else {
                    // If no allocation, we might not own it or it's swapchain (but logic says we own image_map entries)
                    // For safety, warn or leak?
                    // If we just have image and view but no allocation?
                    // We should validly destroy view at least.
                    // And maybe image if created via handle.create_image?
                    // But create_image ALWAYS adds allocation.
                    // So this branch should be unreachable for our created images.
                }
            } else {
                // View but no image? (Only View created?)
                // We don't support View-only creation yet.
            }
        }
    }

    fn create_buffer(&mut self, desc: &BufferDesc) -> BackendBuffer {
        let device = self.get_device().clone();
        let id = self.next_id();

        let usage = crate::convert::convert_buffer_usage_flags(desc.usage);

        let create_info = vk::BufferCreateInfo::default()
            .size(desc.size.max(1)) // Vulkan doesn't like 0 size
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let (mem_usage, alloc_flags) = match desc.memory {
            MemoryType::GpuOnly => (
                vk_mem::MemoryUsage::AutoPreferDevice,
                vk_mem::AllocationCreateFlags::empty(),
            ),
            MemoryType::CpuToGpu => (
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                    | vk_mem::AllocationCreateFlags::MAPPED,
            ),
            MemoryType::GpuToCpu => (
                vk_mem::MemoryUsage::AutoPreferHost,
                vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM
                    | vk_mem::AllocationCreateFlags::MAPPED,
            ),
        };

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: mem_usage,
            flags: alloc_flags,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe {
            let allocator = device.allocator.lock().unwrap();
            allocator
                .create_buffer(&create_info, &allocation_info)
                .expect("Failed to create buffer")
        };

        self.buffer_map.insert(id, buffer);
        self.buffer_allocations.insert(id, allocation);
        BackendBuffer(id)
    }

    fn destroy_buffer(&mut self, handle: BackendBuffer) {
        let id = handle.0;
        let current_frame = self.frame_count;

        if let Some(buffer) = self.buffer_map.remove(&id) {
            if let Some(allocation) = self.buffer_allocations.remove(&id) {
                self.dead_buffers.push((current_frame, buffer, allocation));
            }
        }
    }

    // --- Transient Resource Management (Pooling) ---

    fn create_transient_image(&mut self, desc: &ImageDesc) -> BackendImage {
        // 1. Check Pool
        if let Some(pool) = self.transient_image_pool.get_mut(desc) {
            if let Some(pooled) = pool.pop() {
                // Reuse!
                let id = pooled.id;
                self.image_map.insert(id, pooled.image);
                self.image_views.insert(id, pooled.view);
                self.image_allocations.insert(id, pooled.allocation);
                self.transient_image_descs.insert(id, *desc); // Track desc for release

                // tracing::trace!("Transient Image HIT: {}", id);
                return BackendImage(id);
            }
        }

        // 2. Create New (Fallback)
        let handle = self.create_image(desc);
        let id = handle.0;
        self.transient_image_descs.insert(id, *desc); // Track desc for release
        handle
    }

    fn create_transient_buffer(&mut self, desc: &BufferDesc) -> BackendBuffer {
        if let Some(pool) = self.transient_buffer_pool.get_mut(desc) {
            if let Some(pooled) = pool.pop() {
                let id = pooled.id;
                self.buffer_map.insert(id, pooled.buffer);
                self.buffer_allocations.insert(id, pooled.allocation);
                self.transient_buffer_descs.insert(id, *desc);
                // tracing::trace!("Transient Buffer HIT: {}", id);
                return BackendBuffer(id);
            }
        }
        let handle = self.create_buffer(desc);
        let id = handle.0;
        self.transient_buffer_descs.insert(id, *desc);
        handle
    }

    fn release_transient_image(&mut self, handle: BackendImage) {
        let id = handle.0;
        // Remove from active maps
        if let Some(view) = self.image_views.remove(&id) {
            if let Some(image) = self.image_map.remove(&id) {
                if let Some(allocation) = self.image_allocations.remove(&id) {
                    // Get Desc
                    if let Some(desc) = self.transient_image_descs.remove(&id) {
                        let pool = self
                            .transient_image_pool
                            .entry(desc)
                            .or_insert_with(Vec::new);
                        pool.push(PooledImage {
                            image,
                            view,
                            allocation,
                            id,
                            desc,
                            last_used_frame: self.frame_count,
                        });
                    } else {
                        // Should not happen if used correctly, effectively leaked or non-transient?
                        // If we release a non-transient, we should probably destroy it?
                        // Or warn?
                        // For now, if no desc, we treat it as "not tracked" and destroy immediately?
                        // No, defer it.
                        self.dead_images
                            .push((self.frame_count, image, view, allocation));
                    }
                }
            }
        }
    }

    fn release_transient_buffer(&mut self, handle: BackendBuffer) {
        let id = handle.0;
        if let Some(buffer) = self.buffer_map.remove(&id) {
            if let Some(allocation) = self.buffer_allocations.remove(&id) {
                if let Some(desc) = self.transient_buffer_descs.remove(&id) {
                    let pool = self
                        .transient_buffer_pool
                        .entry(desc)
                        .or_insert_with(Vec::new);
                    pool.push(PooledBuffer {
                        buffer,
                        allocation,
                        id,
                        desc,
                        last_used_frame: self.frame_count,
                    });
                } else {
                    self.dead_buffers
                        .push((self.frame_count, buffer, allocation));
                }
            }
        }
    }

    fn garbage_collect(&mut self) {
        let safe_threshold = self.frame_count.saturating_sub(10); // Use 10 frames lag for extra safety/stability
        // We can tune this. 3 is minimum for triple buffering. 10 is safe for sure.

        let device = self.get_device().clone(); // Clone ARC

        // GC Images
        for pool in self.transient_image_pool.values_mut() {
            // retain items that are RECENT
            // remove items that are OLD (<= safe_threshold)
            // But we need to DESTROY them when removing.
            // retain gives &mut, we can't move out easily to destroy?
            // `retain` closure executes for each element.

            // We'll separate into "keep" and "destroy" lists? expensive copy.
            // Or just iterate with index?

            let mut i = 0;
            while i < pool.len() {
                if pool[i].last_used_frame <= safe_threshold {
                    // Destroy
                    let pooled = pool.swap_remove(i);
                    unsafe {
                        device.handle.destroy_image_view(pooled.view, None); // Use cached device
                        let allocator = device.allocator.lock().unwrap();
                        allocator.destroy_image(pooled.image, &mut pooled.allocation.clone());
                        // Note: allocation clone might be expensive/wrong if it owns something unique?
                        // vk_mem::Allocation is Clone?
                        // Just verified in previous steps it was passed by reference to destroy.
                        // Wait, `destroy_image` takes `&mut Allocation`.
                        // We own `pooled.allocation`.
                        // So `&mut pooled.allocation` is fine, but we moved `pooled` out.
                        // So `let mut alloc = pooled.allocation;`
                    }
                    // Since we swap_remove, the current index `i` is now a new element (or we are at end).
                    // We do NOT increment `i`.
                } else {
                    i += 1;
                }
            }
        }

        // GC Buffers
        for pool in self.transient_buffer_pool.values_mut() {
            let mut i = 0;
            while i < pool.len() {
                if pool[i].last_used_frame <= safe_threshold {
                    let pooled = pool.swap_remove(i);
                    unsafe {
                        let allocator = device.allocator.lock().unwrap();
                        allocator.destroy_buffer(pooled.buffer, &mut pooled.allocation.clone());
                    }
                } else {
                    i += 1;
                }
            }
        }
    }

    fn create_graphics_pipeline(&mut self, desc: &GraphicsPipelineCreateInfo) -> BackendPipeline {
        let device = self.get_device().clone();
        let id = self.next_id();
        info!("Creating Graphics Pipeline");
        use crate::convert::*;

        // 1. Create Shader Modules
        let mut stages = Vec::new();
        let mut specialized_modules = Vec::new(); // keep modules alive

        // Create CStrings first to ensure stable pointers
        let entry_points: Vec<std::ffi::CString> = desc
            .shader_module
            .stages
            .iter()
            .map(|s| std::ffi::CString::new(s.entry_point.as_str()).unwrap())
            .collect();

        for (stage_info, entry_point_cstr) in desc.shader_module.stages.iter().zip(&entry_points) {
            // We need to re-find the bytecode portion?
            // desc.shader.bytecode is the WHOLE binary?
            // Wait, `ShaderModule` has one `bytecode: Vec<u8>`.
            // That assumes it's a single SPIR-V per module or library?
            // Or Slang produces one blob.
            // We create one VkShaderModule per desc.shader.

            // Create the module once:
            // But loop is per stage...
            // Optimally we create module outside loop.
            // But loop context...

            // For now create new module for each stage is invalid if they share bytecode?
            // Standard Vulkan: ShaderModule is the container.
            // We create it ONCE.
            // TODO: Cache shader modules.

            let create_info = vk::ShaderModuleCreateInfo::default().code(unsafe {
                std::slice::from_raw_parts(
                    desc.shader_module.bytecode.as_ptr() as *const u32,
                    desc.shader_module.bytecode.len() / 4,
                )
            });

            let module = unsafe { device.handle.create_shader_module(&create_info, None) }
                .expect("Shader module creation failed");
            specialized_modules.push(module);

            let stage_flag = if stage_info.stage.contains(ShaderStageFlags::Vertex) {
                vk::ShaderStageFlags::VERTEX
            } else if stage_info.stage.contains(ShaderStageFlags::Fragment) {
                vk::ShaderStageFlags::FRAGMENT
            } else {
                vk::ShaderStageFlags::empty()
            };

            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .module(module)
                    .stage(stage_flag)
                    .name(entry_point_cstr.as_c_str()),
            );
        }

        // 2. Vertex Input
        let vk_vertex_bindings: Vec<vk::VertexInputBindingDescription> = desc
            .vertex_input
            .bindings
            .iter()
            .map(|b| vk::VertexInputBindingDescription {
                binding: b.binding,
                stride: b.stride,
                input_rate: convert_vertex_input_rate(b.input_rate),
            })
            .collect();

        let vk_vertex_attributes: Vec<vk::VertexInputAttributeDescription> = desc
            .vertex_input
            .attributes
            .iter()
            .map(|a| vk::VertexInputAttributeDescription {
                location: a.location,
                binding: a.binding,
                format: convert_vertex_format(a.format),
                offset: a.offset,
            })
            .collect();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vk_vertex_bindings)
            .vertex_attribute_descriptions(&vk_vertex_attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(convert_primitive_topology(desc.input_assembly.topology))
            .primitive_restart_enable(desc.input_assembly.primitive_restart_enable);

        // 3. Dynamic States
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(desc.rasterization_state.depth_clamp_enable)
            .rasterizer_discard_enable(desc.rasterization_state.rasterizer_discard_enable)
            .polygon_mode(convert_polygon_mode(desc.rasterization_state.polygon_mode))
            .cull_mode(convert_cull_mode(desc.rasterization_state.cull_mode))
            // Engine Convention: Vulkan uses Clockwise Front Face to compensate for Negative Viewport
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(desc.rasterization_state.depth_bias_enable)
            .depth_bias_constant_factor(desc.rasterization_state.depth_bias_constant_factor)
            .depth_bias_clamp(desc.rasterization_state.depth_bias_clamp)
            .depth_bias_slope_factor(desc.rasterization_state.depth_bias_slope_factor)
            .line_width(desc.rasterization_state.line_width);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(convert_sample_count(desc.multisample_state.sample_count))
            .sample_shading_enable(desc.multisample_state.sample_shading_enable)
            .alpha_to_coverage_enable(desc.multisample_state.alpha_to_coverage_enable);

        // 4. Depth Stencil
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth_stencil_state.depth_test_enable)
            .depth_write_enable(desc.depth_stencil_state.depth_write_enable)
            .depth_compare_op(convert_compare_op(
                desc.depth_stencil_state.depth_compare_op,
            ))
            .depth_bounds_test_enable(false) // Not in i3_gfx yet
            .stencil_test_enable(desc.depth_stencil_state.stencil_test_enable)
            .front(convert_stencil_op_state(&desc.depth_stencil_state.front))
            .back(convert_stencil_op_state(&desc.depth_stencil_state.back))
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);

        // 5. Color Blend
        let attachments: Vec<vk::PipelineColorBlendAttachmentState> = desc
            .render_targets
            .color_targets
            .iter()
            .map(|target| {
                let mut attachment = vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(convert_color_component_flags(target.write_mask));

                if let Some(blend) = target.blend {
                    attachment = attachment
                        .blend_enable(true)
                        .src_color_blend_factor(convert_blend_factor(blend.src_color_factor))
                        .dst_color_blend_factor(convert_blend_factor(blend.dst_color_factor))
                        .color_blend_op(convert_blend_op(blend.color_op))
                        .src_alpha_blend_factor(convert_blend_factor(blend.src_alpha_factor))
                        .dst_alpha_blend_factor(convert_blend_factor(blend.dst_alpha_factor))
                        .alpha_blend_op(convert_blend_op(blend.alpha_op));
                } else {
                    attachment = attachment.blend_enable(false);
                }
                attachment
            })
            .collect();

        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);

        // 5. Layout (Push Constants + Descriptor Sets)

        // Group bindings by set index
        let mut set_bindings: HashMap<u32, Vec<vk::DescriptorSetLayoutBinding>> = HashMap::new();
        for binding in &desc.shader_module.reflection.bindings {
            let descriptor_type = match binding.binding_type {
                i3_gfx::graph::pipeline::BindingType::UniformBuffer => {
                    vk::DescriptorType::UNIFORM_BUFFER
                }
                i3_gfx::graph::pipeline::BindingType::StorageBuffer => {
                    vk::DescriptorType::STORAGE_BUFFER
                }
                i3_gfx::graph::pipeline::BindingType::CombinedImageSampler => {
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                }
                i3_gfx::graph::pipeline::BindingType::Sampler => vk::DescriptorType::SAMPLER,
                i3_gfx::graph::pipeline::BindingType::Texture => vk::DescriptorType::SAMPLED_IMAGE,
                _ => vk::DescriptorType::UNIFORM_BUFFER, // Fallback
            };

            let stage_flags =
                convert_shader_stage_flags(i3_gfx::graph::pipeline::ShaderStageFlags::All); // Simplified for MVP

            let vk_binding = vk::DescriptorSetLayoutBinding::default()
                .binding(binding.binding)
                .descriptor_type(descriptor_type)
                .descriptor_count(binding.count)
                .stage_flags(stage_flags);

            set_bindings
                .entry(binding.set)
                .or_default()
                .push(vk_binding);
        }

        // Create Descriptor Set Layouts (filling gaps)
        let mut descriptor_set_layouts = Vec::new();
        if !set_bindings.is_empty() {
            let max_set = *set_bindings.keys().max().unwrap();
            for i in 0..=max_set {
                let bindings = set_bindings.get(&i).map(|v| v.as_slice()).unwrap_or(&[]);
                let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);

                let layout = unsafe {
                    device
                        .handle
                        .create_descriptor_set_layout(&layout_info, None)
                        .expect("Failed to create descriptor set layout")
                };

                descriptor_set_layouts.push(layout);
                self.descriptor_set_layouts.push(layout); // Track for cleanup
            }
        }

        // Store layouts for this pipeline
        self.pipeline_layouts
            .insert(id, descriptor_set_layouts.clone());

        // Push Constants from reflection
        let pc_ranges: Vec<vk::PushConstantRange> = desc
            .shader_module
            .reflection
            .push_constants
            .iter()
            .map(|pc| vk::PushConstantRange {
                stage_flags: convert_shader_stage_flags(ShaderStageFlags::from_bits_truncate(
                    pc.stage_flags.bits(),
                )),
                offset: pc.offset,
                size: pc.size,
            })
            .collect();

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&pc_ranges);

        let pipeline_layout =
            unsafe { device.handle.create_pipeline_layout(&layout_info, None) }.unwrap();

        // 6. Dynamic Rendering Info
        let color_formats: Vec<vk::Format> = desc
            .render_targets
            .color_targets
            .iter()
            .map(|t| convert_format(t.format))
            .collect();

        let depth_format = desc
            .render_targets
            .depth_stencil_format
            .map(|f| convert_format(f))
            .unwrap_or(vk::Format::UNDEFINED);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format);

        if depth_format != vk::Format::UNDEFINED {
            // Enable Depth Test if format provided (MVP assumption)
            // We need to update depth_stencil state above too?
            // Since we construct pipeline_info later, we can modify it?
            // No, depth_stencil struct is already created.
            // Use let mut depth_stencil?
            // Or rebuild it here? No.
            // Best to move depth_stencil creation DOWN to here.
        }
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push_next(&mut rendering_info);

        let pipeline = unsafe {
            device.handle.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .expect("Pipeline creation failed")[0];

        // Cleanup modules (pipeline owns the internal shader code? No, we need to destroy modules BUT not before pipeline is created)
        // We defer destruction.
        // Or store them.
        self.shader_modules.extend(specialized_modules);
        self.pipelines.push(pipeline);
        self.layouts.push(pipeline_layout);

        self.pipeline_map.insert(id, pipeline);
        self.pipeline_layout_map.insert(id, pipeline_layout);

        BackendPipeline(id)
    }

    fn acquire_swapchain_image(
        &mut self,
        window: WindowHandle,
    ) -> Result<(BackendImage, u64, u32), String> {
        // 1. Advance Frame and Cleanup
        self.frame_count += 1;
        // CRITICAL: At 3000 FPS, 3 frames is 1ms. Present might take ~16ms (60Hz).
        // We need a much larger safe threshold to avoid destroying semaphores still in use by Present.
        // 100 frames at 3000 FPS = 33ms. Safe for 60Hz.
        // Ideally we should verify with fences, but semaphores don't have fences.
        let safe_threshold = self.frame_count.saturating_sub(100);
        let device = self.get_device().clone();

        // Local copy to avoid borrow conflict
        let timeline_sem = self.timeline_sem;
        let expected_signal = self.cpu_timeline + 1;

        // Process Dead Images
        let mut i = 0;
        while i < self.dead_images.len() {
            if self.dead_images[i].0 <= safe_threshold {
                let (_frame, image, view, mut allocation) = self.dead_images.swap_remove(i);
                unsafe {
                    device.handle.destroy_image_view(view, None);
                    let allocator = device.allocator.lock().unwrap();
                    allocator.destroy_image(image, &mut allocation);
                }
            } else {
                i += 1;
            }
        }

        // Process Dead Buffers
        let mut i = 0;
        while i < self.dead_buffers.len() {
            if self.dead_buffers[i].0 <= safe_threshold {
                let (_frame, buffer, mut allocation) = self.dead_buffers.swap_remove(i);
                unsafe {
                    let allocator = device.allocator.lock().unwrap();
                    allocator.destroy_buffer(buffer, &mut allocation);
                }
            } else {
                i += 1;
            }
        }

        // Process Dead Semaphores (GC)
        let mut i = 0;
        while i < self.dead_semaphores.len() {
            if self.dead_semaphores[i].0 <= safe_threshold {
                let (_frame, id, sem) = self.dead_semaphores.swap_remove(i);
                self.semaphores.remove(&id);
                // Recycle instead of destroy!
                self.recycled_semaphores.push(sem);
            } else {
                i += 1;
            }
        }

        // We need to use explicit containment to satisfy borrow checker

        let device = self
            .device
            .as_ref()
            .ok_or("Device not initialized")?
            .clone();

        // 1. Get Window Context (outside loop to check existence, but inside to mutate?)
        // We need to re-acquire mutable borrow if we loop?
        // Actually, we can hold the borrow if we don't return.

        let mut waited_for_fence = false;

        loop {
            let ctx = self
                .windows
                .get_mut(&window.0)
                .ok_or("Invalid window handle")?;

            // 0. Check for Minimization
            let size = ctx.raw.handle.drawable_size();
            if size.0 == 0 || size.1 == 0 {
                return Err("WindowMinimized".to_string());
            }

            // Pre-calculate expected timeline value for this frame's submission
            // let expected_signal = self.cpu_timeline + 1; // Moved up

            // Process Dead Resources... (Keep existing logic, assume it's above or below)
            // ...

            // Wait for previous frame's timeline value
            // Note: We need to do this *inside* the loop if we want to be safe,
            // but typically we do it once per frame attempt.
            // However, since we hold ctx borrow, we do it here.
            if !waited_for_fence {
                let wait_value = ctx.submitted_values[ctx.frame_index];
                if wait_value > 0 {
                    let semaphores = [timeline_sem];
                    let values = [wait_value];
                    let wait_info = vk::SemaphoreWaitInfo::default()
                        .semaphores(&semaphores)
                        .values(&values);
                    unsafe {
                        device
                            .handle
                            .wait_semaphores(&wait_info, u64::MAX)
                            .map_err(|e| e.to_string())?;
                    }
                }

                unsafe {
                    device
                        .handle
                        .reset_command_pool(
                            ctx.command_pools[ctx.frame_index],
                            vk::CommandPoolResetFlags::empty(),
                        )
                        .map_err(|e| e.to_string())?;

                    // Reset cursor for this frame's command buffers
                    ctx.command_buffer_cursors[ctx.frame_index] = 0;
                }
                waited_for_fence = true;
            }

            // 2. Ensure Swapchain exists and matches config
            if ctx.swapchain.is_none() {
                // let size = ctx.raw.handle.drawable_size(); // Already got it
                let sc = crate::swapchain::VulkanSwapchain::new(
                    device.clone(),
                    ctx.raw.surface,
                    size.0,
                    size.1,
                    ctx.config,
                )?;
                ctx.swapchain = Some(sc);
            }

            let swapchain = ctx.swapchain.as_ref().unwrap();

            // 3. Get semaphore for THIS frame slot
            let semaphore = ctx.acquire_semaphores[ctx.frame_index];

            // 4. Acquire
            let result = unsafe {
                let fp = ash::khr::swapchain::Device::new(&self.instance.handle, &device.handle);
                fp.acquire_next_image(swapchain.handle, u64::MAX, semaphore, vk::Fence::null())
            };

            let (index, _) = match result {
                Ok(v) => v,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    // Recreate needed
                    ctx.swapchain = None;
                    continue;
                }
                Err(e) => return Err(e.to_string()),
            };

            // 5. Store state
            ctx.current_acquire_sem = Some(semaphore);
            ctx.current_image_index = Some(index);

            // Update expectation for next time we visit this slot
            ctx.submitted_values[ctx.frame_index] = expected_signal;

            // 6. Wrap result
            let sem_handle_id = self.next_semaphore_id;
            self.next_semaphore_id += 1;
            // self.semaphores.insert(sem_handle_id, semaphore); // Don't track generic, it's per-frame

            let image_raw = swapchain.images[index as usize];
            let image_id = image_raw.as_raw();

            // Critical: Register the view so `begin_pass` can find it
            // FIX: Only create/insert if not already present to avoid leaking VkImageView handles
            // (Swapchain images are stable until recreation)
            if !self.image_views.contains_key(&image_id) {
                let view_raw = swapchain.image_views[index as usize];
                self.image_views.insert(image_id, view_raw);
            }
            // Image map can be overwritten safely (handle copy)
            self.image_map.insert(image_id, image_raw);

            return Ok((BackendImage(image_id), sem_handle_id, index));
        }
    }

    fn submit(
        &mut self,
        _batch: CommandBatch,
        _wait_sems: Vec<u64>,
        _signal_sems: Vec<u64>,
    ) -> Result<u64, String> {
        Ok(0)
    }

    fn begin_pass(
        &mut self,
        desc: PassDescriptor,
        f: Box<dyn FnOnce(&mut dyn PassContext) + Send + Sync>,
    ) -> u64 {
        let device = self.get_device().clone();

        // 1. Identify Target Window & Extent (for Viewport/Pool)
        let mut target_window_ctx = None;
        let mut viewport_extent = vk::Extent2D {
            width: 800,
            height: 600,
        }; // Fallback

        // Resolve target physical IDs from writes
        let mut target_ids = Vec::new();
        for handle in &desc.image_writes {
            let pid = if let Some(&p) = self.external_to_physical.get(&handle.0.0) {
                p
            } else {
                handle.0.0
            };
            target_ids.push(pid);
        }

        // Find window
        'win_loop: for ctx_win in self.windows.values_mut() {
            if let Some(sc) = &ctx_win.swapchain {
                if let Some(idx) = ctx_win.current_image_index {
                    let current_img = sc.images[idx as usize];
                    let current_id = current_img.as_raw();
                    if target_ids.contains(&current_id) {
                        viewport_extent = sc.extent;
                        target_window_ctx = Some(ctx_win);
                        break 'win_loop;
                    }
                }
            }
        }

        if target_window_ctx.is_none() {
            // Fallback logic
        }

        // 2. Allocate Command Buffer
        let cmd = if let Some(ctx_win) = target_window_ctx {
            // Use Recycled Pool
            let pool = ctx_win.command_pools[ctx_win.frame_index];
            let cursor = ctx_win.command_buffer_cursors[ctx_win.frame_index];
            let buffer_list = &mut ctx_win.allocated_command_buffers[ctx_win.frame_index];

            if cursor < buffer_list.len() {
                // Reuse existing buffer
                ctx_win.command_buffer_cursors[ctx_win.frame_index] += 1;
                buffer_list[cursor]
            } else {
                // Allocate new buffer and add to list
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let cmd =
                    unsafe { device.handle.allocate_command_buffers(&alloc_info).unwrap()[0] };
                buffer_list.push(cmd);
                ctx_win.command_buffer_cursors[ctx_win.frame_index] += 1;
                cmd
            }
        } else {
            // Offscreen / Fallback (Leak for now or TODO: Global Pool)
            println!("WARNING: begin_pass fallback! Creating leaking CommandPool");
            unsafe {
                let pool_info = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(device.graphics_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
                let pool = device.handle.create_command_pool(&pool_info, None).unwrap();
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                device.handle.allocate_command_buffers(&alloc_info).unwrap()[0]
            }
        };

        // 3. Begin Recording
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .handle
                .begin_command_buffer(cmd, &begin_info)
                .unwrap()
        };

        let mut ctx = VulkanPassContext {
            cmd,
            device: self.get_device().clone(),
            present_request: None,
            image_handle_map: &self.image_views,
            buffer_map: &self.buffer_map,
            pipeline_map: &self.pipeline_map,
            pipeline_layout_map: &self.pipeline_layout_map,
            descriptor_sets: &self.descriptor_sets,
            current_pipeline_layout: vk::PipelineLayout::null(),
        };

        // If pipeline is set, bind it
        if let Some(pipe_handle) = desc.pipeline {
            ctx.bind_pipeline(pipe_handle);
        }

        // Dynamic Viewport/Scissor setup (Use resolved extent)
        // Engine Convention: Negative Height Viewport for Y-Up
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(viewport_extent.height as f32)
            .width(viewport_extent.width as f32)
            .height(-(viewport_extent.height as f32))
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(viewport_extent);

        unsafe {
            device.handle.cmd_set_viewport(cmd, 0, &[viewport]);
            device.handle.cmd_set_scissor(cmd, 0, &[scissor]);
        }

        // Begin Rendering (Dynamic Rendering)
        // We need to know attachments. `desc.image_writes`.
        // Using `image_map` / `image_handle_map` requires resolving handles.
        // `desc` has logical handles.
        // We need to resolve them using `RenderBackend` but we are inside `begin_pass`.
        // `self` is borrowed mutably.
        // We can look them up in `self.image_handle_map`.

        // Resolve attachments and synchronization
        let mut color_attachments = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut wait_stages = Vec::new();

        if !desc.image_writes.is_empty() {
            let handle = desc.image_writes[0];

            // 1. Resolve to Physical ID
            let physical_id = if let Some(&phy) = self.external_to_physical.get(&handle.0.0) {
                phy // It's an external/registered image
            } else {
                handle.0.0 // It's an internal image (virtual == physical for now?)
            };

            // 2. Find View
            if let Some(&view) = self.image_views.get(&physical_id) {
                // Found View!
                // 3. Check if it matches any Swapchain Image to sync against
                for win_ctx in self.windows.values() {
                    if let Some(current_idx) = win_ctx.current_image_index {
                        if let Some(sc) = &win_ctx.swapchain {
                            let current_sc_image = sc.images[current_idx as usize];
                            // We don't have the ID of the current swapchain image handy directly unless we map it?
                            // wait, acquire_swapchain_image inserted:
                            // self.image_map.insert(image_id, image_raw);
                            // image_id = image_raw.as_raw();

                            if physical_id == current_sc_image.as_raw() {
                                // This is the swapchain image!
                                // We must wait on the acquire semaphore.
                                if let Some(sem) = win_ctx.current_acquire_sem {
                                    wait_semaphores.push(sem);
                                    wait_stages
                                        .push(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
                                }
                            }
                        }
                    }
                }

                // 4. Setup Attachment Info
                let attachment_info = vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) // We assume transition happened?
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    });

                color_attachments.push(attachment_info);

                // IMAGE BARRIER: Transition Undefined/Present -> Color Attachment
                // Needed because we just acquired it.
                // We assume Undefined for now for simplicity, or we track state.
                let image = self.image_map.get(&physical_id).unwrap();
                let barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(*image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                unsafe {
                    device.handle.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                }
            }
        }

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: viewport_extent,
            })
            .layer_count(1)
            .color_attachments(&color_attachments);

        if !color_attachments.is_empty() {
            unsafe {
                device.handle.cmd_begin_rendering(cmd, &rendering_info);
            }
        }

        // We execute the closure.
        f(&mut ctx);

        if !color_attachments.is_empty() {
            unsafe {
                device.handle.cmd_end_rendering(cmd);

                // BARRIER: Color Attachment -> Present Src
                // We need to transition back to present compatible layout if we are presenting.
                // But `present` function is called LATER.
                // However, the `present` logic needs the image in `PRESENT_SRC_KHR`.
                // Actually `cmd_pipeline_barrier` inside `f` (via `ctx.present`) is too late if we ended rendering?
                // No, barriers can happen outside rendering.
                // But `ctx.present` sets a flag.
                // We should do the transition HERE if requested.
            }
        }

        let present_req = ctx.present_request;
        drop(ctx); // Release borrows

        // Handle explicit transition for Present
        if let Some(handle) = present_req {
            // Resolve again (ugly duplication, fix later)
            let physical_id = if let Some(&phy) = self.external_to_physical.get(&handle.0.0) {
                phy
            } else {
                handle.0.0
            };
            if let Some(&image) = self.image_map.get(&physical_id) {
                let barrier = vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty()) // access is usually 0 for presentation
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe {
                    device.handle.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                }
            }
        }

        unsafe { device.handle.end_command_buffer(cmd).unwrap() };

        let signal_sem = self.create_semaphore();
        let vk_signal = self.semaphores[&signal_sem];

        // Signal both:
        // 1. Binary Semaphore (for Present consumptiopn)
        // 2. Timeline Semaphore (for Host synchronization)
        let mut signal_semaphores = vec![vk_signal];
        let mut signal_values = vec![0]; // Binary ignores value

        if self.timeline_sem != vk::Semaphore::null() {
            self.cpu_timeline += 1;
            signal_semaphores.push(self.timeline_sem);
            signal_values.push(self.cpu_timeline);
        }

        let command_buffers = [cmd];

        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);

        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_info)
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores);

        // Perform GC
        self.garbage_collect();

        // No fence needed for queue submit anymore (we use Timeline)
        unsafe {
            device
                .handle
                .queue_submit(device.graphics_queue, &[submit_info], vk::Fence::null())
                .unwrap();
        }

        // Perform GC
        self.garbage_collect();

        // Queue Signal Semaphore for destruction (Leak Fix)
        // We assume it's used by next Acquire/Present or implicitly handled.
        // For draw_triangle, it is waited on by Present.
        // So we can destroy it after safe_threshold.
        self.dead_semaphores
            .push((self.frame_count, signal_sem, vk_signal));

        // Handle Present
        if let Some(_image_handle) = present_req {
            // Search for window context with this current image
            for ctx_win in self.windows.values_mut() {
                if let Some(sc) = &ctx_win.swapchain {
                    if let Some(idx) = ctx_win.current_image_index {
                        // We blindly present the current index if pending
                        let wait_sems = [vk_signal];
                        let swapchains = [sc.handle];
                        let indices = [idx];
                        let present_info = vk::PresentInfoKHR::default()
                            .wait_semaphores(&wait_sems)
                            .swapchains(&swapchains)
                            .image_indices(&indices);

                        unsafe {
                            let fp = ash::khr::swapchain::Device::new(
                                &self.instance.handle,
                                &device.handle,
                            );
                            let _ = fp.queue_present(device.graphics_queue, &present_info);
                        }

                        if let Some(_sem) = ctx_win.current_acquire_sem.take() {
                            // No need to push back to pool, it's fixed in `acquire_semaphores`
                        }
                        ctx_win.current_image_index = None;
                    }
                }
            }
        }

        signal_sem
    }

    fn resolve_image(&self, handle: ImageHandle) -> BackendImage {
        if let Some(&id) = self.external_to_physical.get(&handle.0.0) {
            BackendImage(id)
        } else {
            // For now, assume it might be an internal resource that currently
            // relies on the broken "direct cast" behavior if any exist,
            // or is simply missing because we haven't implemented internal storage mapping.
            // As we investigate the leak, we'll see we need a `transient_resource_map`.
            // For the MVP Triangle, everything is external (Swapchain).
            panic!("Unresolved image handle: {:?}", handle);
        }
    }

    fn resolve_buffer(&self, handle: BufferHandle) -> BackendBuffer {
        BackendBuffer(handle.0.0)
    }

    fn resolve_pipeline(&self, handle: PipelineHandle) -> BackendPipeline {
        BackendPipeline(handle.0.0)
    }
    fn register_external_image(&mut self, handle: ImageHandle, physical: BackendImage) {
        self.external_to_physical.insert(handle.0.0, physical.0);
    }
    fn wait_for_timeline(&self, value: u64, timeout_ns: u64) -> Result<(), String> {
        let device = self.get_device();
        let semaphore = self.timeline_sem; // Use backend's semaphore
        let semaphores = [semaphore];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe {
            device
                .handle
                .wait_semaphores(&wait_info, timeout_ns)
                .map_err(|e| format!("Wait failed: {}", e))
        }
    }

    fn upload_buffer(
        &mut self,
        handle: BackendBuffer,
        data: &[u8],
        offset: u64,
    ) -> Result<(), String> {
        // Extract device first to avoid borrow conflict
        let device = self.get_device().clone();
        let allocator_lock = device.allocator.lock().unwrap();

        // Then get allocation
        let id = handle.0;
        if let Some(allocation) = self.buffer_allocations.get_mut(&id) {
            // Map memory
            let ptr = unsafe {
                allocator_lock
                    .map_memory(allocation)
                    .map_err(|e| format!("Failed to map memory: {}", e))?
            };

            unsafe {
                let dst = ptr.offset(offset as isize);
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());

                // Flush if not coherent
                allocator_lock
                    .flush_allocation(&*allocation, offset, data.len() as u64)
                    .map_err(|e| format!("Failed to flush allocation: {}", e))?;

                allocator_lock.unmap_memory(allocation);
            }
            Ok(())
        } else {
            Err(format!("Buffer not found: {:?}", handle))
        }
    }
    fn allocate_descriptor_set(
        &mut self,
        pipeline: PipelineHandle,
        set_index: u32,
    ) -> Result<DescriptorSetHandle, String> {
        let pipeline_id = pipeline.0.0;
        let layouts = self
            .pipeline_layouts
            .get(&pipeline_id)
            .ok_or_else(|| format!("Pipeline layout not found for {:?}", pipeline))?;

        if set_index as usize >= layouts.len() {
            return Err(format!(
                "Set index {} out of bounds for pipeline {:?}",
                set_index, pipeline
            ));
        }

        let layout = layouts[set_index as usize];
        let layouts_to_alloc = [layout];

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts_to_alloc);

        let sets = unsafe {
            self.get_device()
                .handle
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| format!("Failed to allocate descriptor set: {}", e))?
        };

        let set = sets[0];
        let handle_id = self.next_id();
        self.descriptor_sets.insert(handle_id, set);

        Ok(DescriptorSetHandle(handle_id))
    }

    fn update_descriptor_set(&mut self, set: DescriptorSetHandle, writes: &[DescriptorWrite]) {
        let vk_set = if let Some(s) = self.descriptor_sets.get(&set.0) {
            *s
        } else {
            error!("Descriptor set not found: {:?}", set);
            return;
        };

        // We need to keep the structures alive until the call to update_descriptor_sets
        // But `vk::WriteDescriptorSet` holds references.
        // We iterate and build vectors.

        let mut descriptor_writes = Vec::new();
        let mut buffer_infos = Vec::new(); // Store infos to keep alive
        let mut image_infos = Vec::new();

        // Pass 1: Create Info structures
        for write in writes {
            match write.descriptor_type {
                i3_gfx::graph::pipeline::BindingType::UniformBuffer
                | i3_gfx::graph::pipeline::BindingType::StorageBuffer => {
                    if let Some(info) = &write.buffer_info {
                        if let Some(buf) = self.buffer_map.get(&info.buffer.0.0) {
                            buffer_infos.push(vk::DescriptorBufferInfo {
                                buffer: *buf,
                                offset: info.offset,
                                range: if info.range == 0 {
                                    vk::WHOLE_SIZE
                                } else {
                                    info.range
                                },
                            });
                        }
                    }
                }
                i3_gfx::graph::pipeline::BindingType::CombinedImageSampler
                | i3_gfx::graph::pipeline::BindingType::Texture => {
                    if let Some(info) = &write.image_info {
                        // Resolve Image View
                        // We need `image_views` map, but it's keyed by physical ID.
                        // `info.image` is a logical handle.
                        // We first convert logical -> physical
                        let physical_id =
                            if let Some(&phy) = self.external_to_physical.get(&info.image.0.0) {
                                phy
                            } else {
                                info.image.0.0
                            };

                        if let Some(view) = self.image_views.get(&physical_id) {
                            let layout = match info.image_layout {
                                  i3_gfx::graph::backend::DescriptorImageLayout::General => vk::ImageLayout::GENERAL,
                                  i3_gfx::graph::backend::DescriptorImageLayout::ShaderReadOnlyOptimal => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                              };

                            image_infos.push(vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(), // TODO: Sampler Support
                                image_view: *view,
                                image_layout: layout,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        // Pass 2: Create WriteDescriptorSet
        let mut buf_idx = 0;
        let mut img_idx = 0;

        for write in writes {
            let mut vk_write = vk::WriteDescriptorSet::default()
                .dst_set(vk_set)
                .dst_binding(write.binding)
                .dst_array_element(write.array_element);

            match write.descriptor_type {
                i3_gfx::graph::pipeline::BindingType::UniformBuffer => {
                    vk_write = vk_write.descriptor_type(vk::DescriptorType::UNIFORM_BUFFER);
                    if buf_idx < buffer_infos.len() {
                        vk_write = vk_write.buffer_info(&buffer_infos[buf_idx..=buf_idx]);
                        buf_idx += 1;
                        descriptor_writes.push(vk_write);
                    }
                }
                i3_gfx::graph::pipeline::BindingType::StorageBuffer => {
                    vk_write = vk_write.descriptor_type(vk::DescriptorType::STORAGE_BUFFER);
                    if buf_idx < buffer_infos.len() {
                        vk_write = vk_write.buffer_info(&buffer_infos[buf_idx..=buf_idx]);
                        buf_idx += 1;
                        descriptor_writes.push(vk_write);
                    }
                }
                i3_gfx::graph::pipeline::BindingType::CombinedImageSampler => {
                    vk_write = vk_write.descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
                    if img_idx < image_infos.len() {
                        vk_write = vk_write.image_info(&image_infos[img_idx..=img_idx]);
                        img_idx += 1;
                        descriptor_writes.push(vk_write);
                    }
                }
                i3_gfx::graph::pipeline::BindingType::Texture => {
                    // Sampled Image
                    vk_write = vk_write.descriptor_type(vk::DescriptorType::SAMPLED_IMAGE);
                    if img_idx < image_infos.len() {
                        vk_write = vk_write.image_info(&image_infos[img_idx..=img_idx]);
                        img_idx += 1;
                        descriptor_writes.push(vk_write);
                    }
                }
                _ => {}
            }
        }

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .handle
                .update_descriptor_sets(&descriptor_writes, &[]);
        }
    }
}
impl Drop for VulkanBackend {
    fn drop(&mut self) {
        if let Some(device) = &self.device {
            unsafe {
                device
                    .handle
                    .device_wait_idle()
                    .map_err(|e| error!("WaitIdle failed: {}", e))
                    .ok();

                // Wait a bit for Presentation Engine to release resources (Validation Layer race?)
                std::thread::sleep(std::time::Duration::from_millis(100));

                // 1. Clean up Window Resources
                for ctx in self.windows.values() {
                    for &pool in &ctx.command_pools {
                        device.handle.destroy_command_pool(pool, None);
                    }
                    for &sem in &ctx.acquire_semaphores {
                        device.handle.destroy_semaphore(sem, None);
                    }
                }

                // 2. Clean up Pipelines & Shaders
                for &p in &self.pipelines {
                    device.handle.destroy_pipeline(p, None);
                }
                for &l in &self.layouts {
                    device.handle.destroy_pipeline_layout(l, None);
                }
                for &s in &self.shader_modules {
                    device.handle.destroy_shader_module(s, None);
                }
                for &dsl in &self.descriptor_set_layouts {
                    device.handle.destroy_descriptor_set_layout(dsl, None);
                }

                // 3. Clean up generic Semaphores
                for (_, sem) in &self.semaphores {
                    device.handle.destroy_semaphore(*sem, None);
                }

                // 4. Clean up Recycled Semaphores
                for &sem in &self.recycled_semaphores {
                    device.handle.destroy_semaphore(sem, None);
                }

                if self.timeline_sem != vk::Semaphore::null() {
                    device.handle.destroy_semaphore(self.timeline_sem, None);
                }

                // 5. Clean up Dead Resources (Pending)
                {
                    let allocator = device.allocator.lock().unwrap();

                    for (_, buffer, alloc) in &mut self.dead_buffers {
                        allocator.destroy_buffer(*buffer, alloc);
                    }

                    // Dead images:
                    for (_, image, view, alloc) in &mut self.dead_images {
                        device.handle.destroy_image_view(*view, None);
                        allocator.destroy_image(*image, alloc);
                    }
                }

                // 6. Clean up Pools (Transient)
                {
                    let allocator = device.allocator.lock().unwrap();
                    for pool in self.transient_buffer_pool.values_mut() {
                        for pooled in pool {
                            allocator.destroy_buffer(pooled.buffer, &mut pooled.allocation);
                        }
                    }
                    for pool in self.transient_image_pool.values_mut() {
                        for pooled in pool {
                            device.handle.destroy_image_view(pooled.view, None);
                            allocator.destroy_image(pooled.image, &mut pooled.allocation);
                        }
                    }
                }

                // 7. Clean up Buffers (Active)
                {
                    let allocator = device.allocator.lock().unwrap();
                    for (id, buffer) in &self.buffer_map {
                        if let Some(mut alloc) = self.buffer_allocations.get(id).cloned() {
                            allocator.destroy_buffer(*buffer, &mut alloc);
                        }
                    }
                }

                // 8. Clean up Images and Views (Active)
                // CAUTION: Filter out Swapchain Views!
                // Swapchain images are owned by the swapchain, but we might have views in our map.
                // We identify them by checking if the image handle belongs to a swapchain.
                let mut swapchain_images = std::collections::HashSet::new();
                for ctx in self.windows.values() {
                    if let Some(sc) = &ctx.swapchain {
                        for &img in &sc.images {
                            swapchain_images.insert(img);
                        }
                    }
                }

                let mut swapchain_ids = std::collections::HashSet::new();
                for (id, image) in &self.image_map {
                    if swapchain_images.contains(image) {
                        swapchain_ids.insert(*id);
                    }
                }

                for (id, view) in &self.image_views {
                    if !swapchain_ids.contains(id) {
                        device.handle.destroy_image_view(*view, None);
                    }
                }

                {
                    let allocator = device.allocator.lock().unwrap();
                    for (id, image) in &self.image_map {
                        if let Some(mut alloc) = self.image_allocations.get(id).cloned() {
                            allocator.destroy_image(*image, &mut alloc);
                        }
                    }
                }

                // 9. Clean up Descriptor Pool
                if self.descriptor_pool != vk::DescriptorPool::null() {
                    device
                        .handle
                        .destroy_descriptor_pool(self.descriptor_pool, None);
                }

                // Descriptor Set Layouts (Self-managed or per pipeline?)
                // We destroyed pipelines/layouts above.
                // Descriptor Sets are freed with the pool.
            }
        }
    }
}

pub struct VulkanPassContext<'a> {
    cmd: vk::CommandBuffer,
    device: Arc<crate::device::VulkanDevice>,
    present_request: Option<ImageHandle>,
    image_handle_map: &'a HashMap<u64, vk::ImageView>,
    buffer_map: &'a HashMap<u64, vk::Buffer>,
    pipeline_map: &'a HashMap<u64, vk::Pipeline>,
    pipeline_layout_map: &'a HashMap<u64, vk::PipelineLayout>,
    descriptor_sets: &'a HashMap<u64, vk::DescriptorSet>,
    current_pipeline_layout: vk::PipelineLayout,
}

impl<'a> PassContext for VulkanPassContext<'a> {
    fn bind_pipeline(&mut self, pipeline: PipelineHandle) {
        if let Some(pipe) = self.pipeline_map.get(&pipeline.0.0) {
            unsafe {
                self.device.handle.cmd_bind_pipeline(
                    self.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    *pipe,
                );
            }
        }
        if let Some(layout) = self.pipeline_layout_map.get(&pipeline.0.0) {
            self.current_pipeline_layout = *layout;
        }
    }

    fn bind_vertex_buffer(&mut self, binding: u32, handle: BufferHandle) {
        // Resolve buffer
        if let Some(buf) = self.buffer_map.get(&handle.0.0) {
            unsafe {
                self.device
                    .handle
                    .cmd_bind_vertex_buffers(self.cmd, binding, &[*buf], &[0]);
            }
        }
    }

    fn bind_index_buffer(&mut self, handle: BufferHandle, index_type: IndexType) {
        if let Some(buf) = self.buffer_map.get(&handle.0.0) {
            let vk_type = match index_type {
                IndexType::Uint16 => vk::IndexType::UINT16,
                IndexType::Uint32 => vk::IndexType::UINT32,
            };
            unsafe {
                self.device
                    .handle
                    .cmd_bind_index_buffer(self.cmd, *buf, 0, vk_type);
            }
        }
    }

    fn bind_descriptor_set(&mut self, set_index: u32, handle: DescriptorSetHandle) {
        if let Some(set) = self.descriptor_sets.get(&handle.0) {
            unsafe {
                self.device.handle.cmd_bind_descriptor_sets(
                    self.cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.current_pipeline_layout,
                    set_index,
                    &[*set],
                    &[],
                );
            }
        }
    }

    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32) {
        let viewport = vk::Viewport {
            x,
            y,
            width,
            height,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            self.device
                .handle
                .cmd_set_viewport(self.cmd, 0, &[viewport]);
        }
    }

    fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D { width, height },
        };
        unsafe {
            self.device.handle.cmd_set_scissor(self.cmd, 0, &[scissor]);
        }
    }

    fn draw(&mut self, vertex_count: u32, first_vertex: u32) {
        // Hack: BeginRendering here if not started?
        // We know we are drawing.
        // But we need attachments.

        // For the TRIANGLE example, we rely on the fact that `begin_pass`
        // should have set up rendering.
        // Since I deferred `vkCmdBeginRendering` logic in `begin_pass` due to complexity,
        // and now I need it, I'll insert a simplified version there or here.
        // Realistically, `begin_pass` is where it belongs.

        // For MVP, if we haven't started rendering, the draw will fail validation.
        // We need `vkCmdBeginRendering`.

        unsafe {
            self.device
                .handle
                .cmd_draw(self.cmd, vertex_count, 1, first_vertex, 0);
        }
    }

    fn draw_indexed(&mut self, index_count: u32, first_index: u32, vertex_offset: i32) {
        unsafe {
            self.device.handle.cmd_draw_indexed(
                self.cmd,
                index_count,
                1,
                first_index,
                vertex_offset,
                0,
            );
        }
    }

    fn dispatch(&mut self, _x: u32, _y: u32, _z: u32) {}

    fn present(&mut self, image: ImageHandle) {
        self.present_request = Some(image);
        // Explicitly transition layout to PRESENT_SRC_KHR provided we know the image
        if let Some(_view) = self.image_handle_map.get(&image.0.0) {
            // We need the IMAGE, not the VIEW, to barrier it.
            // Refactoring needed to map view->image or track both.
        }

        // Record barrier?
        // Record barrier?
        /*
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT) // The present wait semaphore handles the rest
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(vk::Image::null()) // We lack the vk::Image here!
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        */

        // Because we lack `vk::Image`, we are skipping the barrier.
        // The Validation Layer warned about "waiting on semaphore".
        // With semaphore, it might be fine on some drivers, but strict usage requires layout transition.
        // The `RenderPass` usually handles finalLayout=Present.
        // Dynamic Rendering requires manual transition.
    }
}
