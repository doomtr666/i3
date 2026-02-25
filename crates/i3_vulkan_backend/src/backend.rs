use ash::vk;
use ash::vk::Handle;
use i3_gfx::graph::backend::RenderBackendInternal;
use i3_gfx::graph::backend::*;
use i3_gfx::graph::pass::RenderPass;
use i3_gfx::graph::pipeline::*;
use i3_gfx::graph::types::*;

use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, error, info};
use vk_mem::Alloc;

pub struct PhysicalImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Option<vk_mem::Allocation>,
    pub desc: ImageDesc,
    pub format: vk::Format,

    pub last_layout: vk::ImageLayout,
    pub last_access: vk::AccessFlags2,
    pub last_stage: vk::PipelineStageFlags2,
    pub last_write_frame: u64,
}

pub struct PhysicalBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<vk_mem::Allocation>,
    pub desc: BufferDesc,

    // Synchronization state (Sync2)
    pub last_access: vk::AccessFlags2,
    pub last_stage: vk::PipelineStageFlags2,
}

enum Slot<T> {
    Occupied {
        data: T,
        generation: u32,
    },
    Free {
        next_free: Option<u32>,
        generation: u32,
    },
}

pub struct ResourceArena<T> {
    slots: Vec<Slot<T>>,
    free_head: Option<u32>,
}

impl<T> ResourceArena<T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::with_capacity(256),
            free_head: None,
        }
    }

    pub fn insert(&mut self, data: T) -> u64 {
        if let Some(index) = self.free_head {
            let slot = &mut self.slots[index as usize];
            if let Slot::Free {
                next_free,
                generation,
            } = *slot
            {
                let generation_val = generation;
                *slot = Slot::Occupied {
                    data,
                    generation: generation_val,
                };
                self.free_head = next_free;
                return ((generation_val as u64) << 32) | (index as u64);
            }
        }

        let index = self.slots.len() as u32;
        let generation = 1u32;
        self.slots.push(Slot::Occupied { data, generation });
        ((generation as u64) << 32) | (index as u64)
    }

    pub fn get(&self, id: u64) -> Option<&T> {
        let index = (id & 0xFFFFFFFF) as usize;
        let generation_val = (id >> 32) as u32;
        if let Some(Slot::Occupied { data, generation }) = self.slots.get(index) {
            if *generation == generation_val {
                return Some(data);
            }
        }
        None
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut T> {
        let index = (id & 0xFFFFFFFF) as usize;
        let generation_val = (id >> 32) as u32;
        if let Some(Slot::Occupied { data, generation }) = self.slots.get_mut(index) {
            if *generation == generation_val {
                return Some(data);
            }
        }
        None
    }

    pub fn remove(&mut self, id: u64) -> Option<T> {
        let index = (id & 0xFFFFFFFF) as usize;
        let generation_val = (id >> 32) as u32;
        if index >= self.slots.len() {
            return None;
        }

        match self.slots[index] {
            Slot::Occupied { generation, .. } if generation == generation_val => {
                let old_slot = std::mem::replace(
                    &mut self.slots[index],
                    Slot::Free {
                        next_free: self.free_head,
                        generation: generation + 1,
                    },
                );
                self.free_head = Some(index as u32);
                if let Slot::Occupied { data, .. } = old_slot {
                    Some(data)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u64, &mut T)> {
        self.slots.iter_mut().enumerate().filter_map(|(i, slot)| {
            if let Slot::Occupied { data, generation } = slot {
                let id = ((*generation as u64) << 32) | (i as u64);
                Some((id, data))
            } else {
                None
            }
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (u64, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| {
            if let Slot::Occupied { data, generation } = slot {
                let id = ((*generation as u64) << 32) | (i as u64);
                Some((id, data))
            } else {
                None
            }
        })
    }

    pub fn ids(&self) -> Vec<u64> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                if let Slot::Occupied { generation, .. } = slot {
                    Some(((*generation as u64) << 32) | (i as u64))
                } else {
                    None
                }
            })
            .collect()
    }
}

// ...

struct VulkanFrameContext {
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    allocated_command_buffers: Vec<vk::CommandBuffer>,
    cursor: usize,
    submitted_cursor: usize,
    last_completion_value: u64,
}

struct WindowContext {
    // Order matters for drop: swapchain must be dropped BEFORE raw (surface)
    swapchain: Option<crate::swapchain::VulkanSwapchain>,
    raw: crate::window::VulkanWindow,
    config: SwapchainConfig,
    // Semaphores for acquire (per frame in flight)
    acquire_semaphores: Vec<vk::Semaphore>,
    acquire_semaphore_ids: Vec<u64>,
    // Semaphores for present (per frame in flight)
    present_semaphores: Vec<vk::Semaphore>,
    #[allow(dead_code)]
    present_semaphore_ids: Vec<u64>,
    // Track the current frame's acquire semaphore to pair it with the image
    current_acquire_sem_id: Option<u64>,
    current_image_index: Option<u32>,
}

#[derive(Clone)]
pub struct PhysicalPipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub bind_point: vk::PipelineBindPoint,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub pushable_sets_mask: u32,
}

pub struct VulkanBackend {
    // Window Management
    windows: HashMap<u64, WindowContext>,
    next_window_id: u64,

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

    pub frame_count: u64,
    pub dead_images: Vec<(u64, vk::Image, vk::ImageView, vk_mem::Allocation)>, // Frame, Image, View, Alloc
    pub dead_buffers: Vec<(u64, vk::Buffer, vk_mem::Allocation)>,
    pub dead_semaphores: Vec<(u64, u64, vk::Semaphore)>, // Frame, ID, Handle
    pub recycled_semaphores: Vec<vk::Semaphore>,

    // Transient Pools
    transient_image_pool: HashMap<ImageDesc, Vec<u64>>,
    transient_buffer_pool: HashMap<BufferDesc, Vec<u64>>,

    // Resources
    pub samplers: ResourceArena<vk::Sampler>,
    pub dead_samplers: Vec<(u64, vk::Sampler)>, // Frame, Handle
    pub semaphores: ResourceArena<vk::Semaphore>,
    pub timeline_sem: vk::Semaphore, // Global timeline for graphics queue
    pub cpu_timeline: u64,           // Current CPU submission value
    pub next_semaphore_id: u64,
    pub static_descriptor_pool: vk::DescriptorPool,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pub descriptor_sets: ResourceArena<vk::DescriptorSet>,
    pub descriptor_pool_max_sets: u32,

    // Loaders
    swapchain_loader: Option<ash::khr::swapchain::Device>,

    // Scratch Buffers for hot path
    target_id_scratch: Vec<u64>,
    image_barrier_scratch: Vec<vk::ImageMemoryBarrier2<'static>>,
    buffer_barrier_scratch: Vec<vk::BufferMemoryBarrier2<'static>>,

    // Global Frame Contexts
    frame_contexts: Vec<VulkanFrameContext>,
    frame_started: bool,
    global_frame_index: usize,
    next_resource_id: u64,

    // Dependencies (Dropped Last)
    pub device: Option<Arc<crate::device::VulkanDevice>>,
    pub instance: Arc<crate::instance::VulkanInstance>,
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
            pipeline_resources: ResourceArena::new(),
            shader_modules: Vec::new(),
            images: ResourceArena::new(),
            buffers: ResourceArena::new(),
            external_to_physical: HashMap::new(),
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
            descriptor_sets: ResourceArena::new(),
            descriptor_pool_max_sets: 1000,
            timeline_sem: vk::Semaphore::null(), // Will be initialized below
            cpu_timeline: 0,
            frame_contexts: Vec::new(),
            swapchain_loader: None,
            target_id_scratch: Vec::with_capacity(32),
            image_barrier_scratch: Vec::with_capacity(32),
            buffer_barrier_scratch: Vec::with_capacity(32),
            frame_started: false,
            global_frame_index: 0,
            next_resource_id: 1,
        })
    }

    fn get_device(&self) -> &Arc<crate::device::VulkanDevice> {
        self.device.as_ref().expect("Backend not initialized")
    }

    fn get_image_barrier(
        &mut self,
        physical_id: u64,
        new_layout: vk::ImageLayout,
        dst_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
    ) -> Option<vk::ImageMemoryBarrier2<'static>> {
        if let Some(img) = self.images.get_mut(physical_id) {
            // Optimization: Skip only for Read-After-Read (RAR) where layout and stage already match
            let is_write = |access: vk::AccessFlags2| {
                access.intersects(
                    vk::AccessFlags2::SHADER_WRITE
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE
                        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                        | vk::AccessFlags2::TRANSFER_WRITE
                        | vk::AccessFlags2::MEMORY_WRITE,
                )
            };

            let needs_barrier =
                img.last_layout != new_layout || is_write(img.last_access) || is_write(dst_access);

            if !needs_barrier && (img.last_stage & dst_stage) == dst_stage {
                return None;
            }

            debug!(
                "Transition Image {:?}: {:?} -> {:?}",
                physical_id, img.last_layout, new_layout
            );

            let aspect_mask = if img.format == vk::Format::D32_SFLOAT {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };

            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(img.last_stage)
                .src_access_mask(img.last_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
                .old_layout(img.last_layout)
                .new_layout(new_layout)
                .image(img.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            img.last_layout = new_layout;
            img.last_access = dst_access;
            img.last_stage = dst_stage;

            Some(barrier)
        } else {
            None
        }
    }

    fn get_buffer_barrier(
        &mut self,
        physical_id: u64,
        dst_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
    ) -> Option<vk::BufferMemoryBarrier2<'static>> {
        if let Some(buf) = self.buffers.get_mut(physical_id) {
            // Optimization: Skip only for Read-After-Read (RAR) where state already matches
            let is_write = |access: vk::AccessFlags2| {
                access.intersects(
                    vk::AccessFlags2::SHADER_WRITE
                        | vk::AccessFlags2::SHADER_STORAGE_WRITE
                        | vk::AccessFlags2::TRANSFER_WRITE
                        | vk::AccessFlags2::MEMORY_WRITE,
                )
            };

            let needs_barrier = is_write(buf.last_access) || is_write(dst_access);

            if !needs_barrier && (buf.last_stage & dst_stage) == dst_stage {
                return None;
            }

            debug!(
                "Transition Buffer {:?}: {:?} -> {:?} / {:?} -> {:?}",
                physical_id, buf.last_stage, dst_stage, buf.last_access, dst_access
            );

            let barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(buf.last_stage)
                .src_access_mask(buf.last_access)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)
                .buffer(buf.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE);

            buf.last_access = dst_access;
            buf.last_stage = dst_stage;

            Some(barrier)
        } else {
            None
        }
    }

    fn get_image_state(
        &self,
        usage: ResourceUsage,
        is_write: bool,
        bind_point: vk::PipelineBindPoint,
    ) -> (vk::ImageLayout, vk::AccessFlags2, vk::PipelineStageFlags2) {
        let mut layout = vk::ImageLayout::GENERAL;
        let mut access = vk::AccessFlags2::empty();
        let mut stage = vk::PipelineStageFlags2::NONE;

        if usage.intersects(ResourceUsage::COLOR_ATTACHMENT) {
            layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            access = vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
            if !is_write {
                access |= vk::AccessFlags2::COLOR_ATTACHMENT_READ;
            }
            stage = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
        } else if usage.intersects(ResourceUsage::DEPTH_STENCIL) {
            if is_write || usage.intersects(ResourceUsage::WRITE) {
                layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                access = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
            } else {
                layout = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                access = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
            }
            stage = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        } else if usage.intersects(ResourceUsage::SHADER_WRITE) {
            layout = vk::ImageLayout::GENERAL;
            access = vk::AccessFlags2::SHADER_STORAGE_WRITE
                | vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_WRITE;
            stage = if bind_point == vk::PipelineBindPoint::COMPUTE {
                vk::PipelineStageFlags2::COMPUTE_SHADER
            } else {
                vk::PipelineStageFlags2::FRAGMENT_SHADER
            };
        } else if usage.intersects(ResourceUsage::SHADER_READ) {
            layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            access = vk::AccessFlags2::SHADER_READ;
            stage = if bind_point == vk::PipelineBindPoint::COMPUTE {
                vk::PipelineStageFlags2::COMPUTE_SHADER
            } else {
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::VERTEX_SHADER
            };
        } else if usage.intersects(ResourceUsage::TRANSFER_WRITE) {
            layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            access = vk::AccessFlags2::TRANSFER_WRITE;
            stage = vk::PipelineStageFlags2::TRANSFER;
        } else if usage.intersects(ResourceUsage::TRANSFER_READ) {
            layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            access = vk::AccessFlags2::TRANSFER_READ;
            stage = vk::PipelineStageFlags2::TRANSFER;
        }

        (layout, access, stage)
    }

    fn get_buffer_state(
        &self,
        usage: ResourceUsage,
        bind_point: vk::PipelineBindPoint,
    ) -> (vk::AccessFlags2, vk::PipelineStageFlags2) {
        let mut access = vk::AccessFlags2::empty();
        let mut stage = vk::PipelineStageFlags2::NONE;

        if usage.intersects(ResourceUsage::SHADER_WRITE) {
            access = vk::AccessFlags2::SHADER_STORAGE_WRITE
                | vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_WRITE;
            stage = if bind_point == vk::PipelineBindPoint::COMPUTE {
                vk::PipelineStageFlags2::COMPUTE_SHADER
            } else {
                vk::PipelineStageFlags2::FRAGMENT_SHADER
            };
        } else if usage.intersects(ResourceUsage::SHADER_READ) {
            access = vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::UNIFORM_READ;
            stage = if bind_point == vk::PipelineBindPoint::COMPUTE {
                vk::PipelineStageFlags2::COMPUTE_SHADER
            } else {
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::VERTEX_SHADER
            };
        } else if usage.intersects(ResourceUsage::TRANSFER_WRITE) {
            access = vk::AccessFlags2::TRANSFER_WRITE;
            stage = vk::PipelineStageFlags2::TRANSFER;
        } else if usage.intersects(ResourceUsage::TRANSFER_READ) {
            access = vk::AccessFlags2::TRANSFER_READ;
            stage = vk::PipelineStageFlags2::TRANSFER;
        }

        (access, stage)
    }

    #[allow(dead_code)]
    pub fn create_semaphore(&mut self) -> u64 {
        let sem = self.create_semaphore_raw();
        self.semaphores.insert(sem)
    }

    pub fn next_id(&mut self) -> u64 {
        let id = self.next_resource_id;
        self.next_resource_id += 1;
        id
    }

    fn create_semaphore_raw(&mut self) -> vk::Semaphore {
        if let Some(recycled) = self.recycled_semaphores.pop() {
            recycled
        } else {
            let device = self.get_device();
            let create_info = vk::SemaphoreCreateInfo::default();
            unsafe { device.handle.create_semaphore(&create_info, None) }.unwrap()
        }
    }

    pub fn window_size(&self, window: WindowHandle) -> Option<(u32, u32)> {
        self.windows
            .get(&window.0)
            .map(|ctx| ctx.raw.handle.drawable_size())
    }

    fn sdl_to_keycode(sdl: sdl2::keyboard::Keycode) -> Option<KeyCode> {
        match sdl {
            sdl2::keyboard::Keycode::Escape => Some(KeyCode::Escape),
            sdl2::keyboard::Keycode::Tab => Some(KeyCode::Tab),
            sdl2::keyboard::Keycode::Space => Some(KeyCode::Space),
            sdl2::keyboard::Keycode::W => Some(KeyCode::W),
            sdl2::keyboard::Keycode::A => Some(KeyCode::A),
            sdl2::keyboard::Keycode::S => Some(KeyCode::S),
            sdl2::keyboard::Keycode::D => Some(KeyCode::D),
            sdl2::keyboard::Keycode::Z => Some(KeyCode::Z),
            sdl2::keyboard::Keycode::Q => Some(KeyCode::Q),
            sdl2::keyboard::Keycode::LShift => Some(KeyCode::LShift),
            _ => None,
        }
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
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1000,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 1000,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1000,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1000);
        self.static_descriptor_pool = unsafe {
            self.get_device()
                .handle
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| format!("Failed to create static descriptor pool: {}", e))?
        };

        // Create Global Frame Contexts
        let mut frame_contexts = Vec::new();
        let device = self.get_device().clone();

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
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1000,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 1000,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1000,
            },
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1000);

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
                    .unwrap();

                let d_pool = device
                    .handle
                    .create_descriptor_pool(&pool_info, None)
                    .unwrap();

                frame_contexts.push(VulkanFrameContext {
                    command_pool: pool,
                    descriptor_pool: d_pool,
                    allocated_command_buffers: Vec::new(),
                    cursor: 0,
                    submitted_cursor: 0,
                    last_completion_value: 0,
                });
            }
        }
        self.frame_contexts = frame_contexts;

        // Initialize Loaders
        self.swapchain_loader = Some(ash::khr::swapchain::Device::new(
            &self.instance.handle,
            &self.get_device().handle,
        ));

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

        let _id = self.next_window_id;
        self.next_window_id += 1;

        // Create Semaphores per frame for this window (typically 3 for triple buffering)
        let win_id = self.next_window_id;
        self.next_window_id += 1;

        let mut acquire_sems = Vec::new();
        let mut acquire_sem_ids = Vec::new();
        let mut present_sems = Vec::new();
        let mut present_sem_ids = Vec::new();
        let _device_handle = self.get_device().handle.clone();

        for _ in 0..3 {
            let a_id = self.create_semaphore();
            let p_id = self.create_semaphore();

            let a_sem = self.semaphores.get(a_id).cloned().unwrap();
            let p_sem = self.semaphores.get(p_id).cloned().unwrap();

            acquire_sems.push(a_sem);
            acquire_sem_ids.push(a_id);

            present_sems.push(p_sem);
            present_sem_ids.push(p_id);
        }

        let context = WindowContext {
            raw: vulkan_window,
            swapchain: None, // This will be created later
            config: SwapchainConfig {
                vsync: false,
                srgb: true,
                min_image: 3,
            }, // Default
            acquire_semaphores: acquire_sems,
            acquire_semaphore_ids: acquire_sem_ids,
            present_semaphores: present_sems,
            present_semaphore_ids: present_sem_ids,
            current_acquire_sem_id: None,
            current_image_index: None,
        };

        self.windows.insert(win_id, context);
        Ok(WindowHandle(win_id))
    }

    fn destroy_window(&mut self, window: WindowHandle) {
        if let Some(mut ctx) = self.windows.remove(&window.0) {
            if let Some(sc) = ctx.swapchain.take() {
                let device = self.get_device();
                unsafe {
                    device.handle.device_wait_idle().ok();
                }
                self.unregister_swapchain_images(&sc.images);
            }
        }
    }

    fn configure_window(
        &mut self,
        window: WindowHandle,
        config: SwapchainConfig,
    ) -> Result<(), String> {
        let sc_opt = if let Some(ctx) = self.windows.get_mut(&window.0) {
            ctx.config = config;
            // Invalidate swapchain so it recreates on next acquire
            ctx.swapchain.take()
        } else {
            return Err("Invalid window handle".to_string());
        };

        if let Some(sc) = sc_opt {
            let device = self.get_device();
            unsafe {
                device.handle.device_wait_idle().ok();
            }
            self.unregister_swapchain_images(&sc.images);
        }
        Ok(())
    }

    fn poll_events(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        let mut resize_happened = false;
        if let Some(pump) = &mut self.event_pump {
            for event in pump.poll_iter() {
                match event {
                    sdl2::event::Event::Quit { .. } => events.push(Event::Quit),
                    sdl2::event::Event::KeyDown {
                        keycode: Some(kd), ..
                    } => {
                        if let Some(key) = Self::sdl_to_keycode(kd) {
                            events.push(Event::KeyDown { key });
                        }
                    }
                    sdl2::event::Event::KeyUp {
                        keycode: Some(kd), ..
                    } => {
                        if let Some(key) = Self::sdl_to_keycode(kd) {
                            events.push(Event::KeyUp { key });
                        }
                    }
                    sdl2::event::Event::Window {
                        win_event: sdl2::event::WindowEvent::Resized(w, h),
                        ..
                    } => {
                        events.push(Event::Resize {
                            width: w as u32,
                            height: h as u32,
                        });
                        resize_happened = true;
                    }
                    sdl2::event::Event::MouseButtonDown {
                        mouse_btn, x, y, ..
                    } => {
                        events.push(Event::MouseDown {
                            button: match mouse_btn {
                                sdl2::mouse::MouseButton::Left => 1,
                                sdl2::mouse::MouseButton::Right => 2,
                                sdl2::mouse::MouseButton::Middle => 3,
                                _ => 0,
                            },
                            x,
                            y,
                        });
                    }
                    sdl2::event::Event::MouseButtonUp {
                        mouse_btn, x, y, ..
                    } => {
                        events.push(Event::MouseUp {
                            button: match mouse_btn {
                                sdl2::mouse::MouseButton::Left => 1,
                                sdl2::mouse::MouseButton::Right => 2,
                                sdl2::mouse::MouseButton::Middle => 3,
                                _ => 0,
                            },
                            x,
                            y,
                        });
                    }
                    sdl2::event::Event::MouseMotion { x, y, .. } => {
                        events.push(Event::MouseMove { x, y });
                    }
                    sdl2::event::Event::MouseWheel { y, .. } => {
                        events.push(Event::MouseWheel { x: 0, y });
                    }
                    _ => {}
                }
            }
        }

        if resize_happened {
            let mut to_unregister = if self.windows.len() > 0 {
                Vec::with_capacity(self.windows.len())
            } else {
                Vec::new()
            };
            for ctx in self.windows.values_mut() {
                if let Some(sc) = ctx.swapchain.take() {
                    to_unregister.push(sc);
                }
            }
            if !to_unregister.is_empty() {
                let device = self.get_device();
                unsafe {
                    device.handle.device_wait_idle().ok();
                }
                for sc in to_unregister {
                    self.unregister_swapchain_images(&sc.images);
                }
            }
        }
        events
    }

    fn create_image(&mut self, desc: &ImageDesc) -> BackendImage {
        let device = self.get_device().clone();
        debug!("Creating Image: {:?}", desc);

        let extent = vk::Extent3D {
            width: desc.width,
            height: desc.height,
            depth: desc.depth,
        };

        // Translate format
        let format = crate::convert::convert_format(desc.format);

        // Use provided usage flags, but add common bits for flexibility
        let mut usage = crate::convert::convert_image_usage_flags(desc.usage);
        usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;

        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(desc.mip_levels.max(1))
            .array_layers(desc.array_layers.max(1))
            .samples(vk::SampleCountFlags::TYPE_1)
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
            .view_type(crate::convert::convert_image_view_type(desc.view_type))
            .format(format)
            .components(crate::convert::convert_component_mapping(desc.swizzle))
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: desc.mip_levels.max(1),
                base_array_layer: 0,
                layer_count: desc.array_layers.max(1),
            });
        let view = unsafe { device.handle.create_image_view(&view_info, None) }
            .expect("Failed to create view");

        let id = self.images.insert(PhysicalImage {
            image,
            view,
            allocation: Some(allocation),
            desc: *desc,
            format,
            last_layout: vk::ImageLayout::UNDEFINED,
            last_access: vk::AccessFlags2::empty(),
            last_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
            last_write_frame: 0,
        });

        BackendImage(id)
    }

    fn destroy_image(&mut self, handle: BackendImage) {
        if let Some(physical) = self.images.remove(handle.0) {
            if let (Some(image), Some(allocation)) = (Some(physical.image), physical.allocation) {
                self.dead_images
                    .push((self.frame_count, image, physical.view, allocation));
            }
        }
    }

    fn create_buffer(&mut self, desc: &BufferDesc) -> BackendBuffer {
        let device = self.get_device().clone();

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

        let id = self.buffers.insert(PhysicalBuffer {
            buffer,
            allocation: Some(allocation),
            desc: *desc,
            last_access: vk::AccessFlags2::empty(),
            last_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
        });
        BackendBuffer(id)
    }

    fn destroy_buffer(&mut self, handle: BackendBuffer) {
        if let Some(physical) = self.buffers.remove(handle.0) {
            if let Some(allocation) = physical.allocation {
                self.dead_buffers
                    .push((self.frame_count, physical.buffer, allocation));
            }
        }
    }

    fn create_sampler(&mut self, desc: &SamplerDesc) -> SamplerHandle {
        let mag_filter = match desc.mag_filter {
            Filter::Nearest => vk::Filter::NEAREST,
            Filter::Linear => vk::Filter::LINEAR,
        };
        let min_filter = match desc.min_filter {
            Filter::Nearest => vk::Filter::NEAREST,
            Filter::Linear => vk::Filter::LINEAR,
        };
        let mipmap_mode = match desc.mipmap_mode {
            i3_gfx::graph::types::MipmapMode::Nearest => vk::SamplerMipmapMode::NEAREST,
            i3_gfx::graph::types::MipmapMode::Linear => vk::SamplerMipmapMode::LINEAR,
        };

        let convert_address = |mode: AddressMode| match mode {
            AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            AddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
            AddressMode::MirrorClampToEdge => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
        };

        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(mag_filter)
            .min_filter(min_filter)
            .mipmap_mode(mipmap_mode)
            .address_mode_u(convert_address(desc.address_mode_u))
            .address_mode_v(convert_address(desc.address_mode_v))
            .address_mode_w(convert_address(desc.address_mode_w))
            .max_anisotropy(1.0)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE);

        let sampler = unsafe {
            self.get_device()
                .handle
                .create_sampler(&create_info, None)
                .expect("Failed to create sampler")
        };

        let handle = self.samplers.insert(sampler);
        SamplerHandle(handle)
    }

    fn destroy_sampler(&mut self, handle: SamplerHandle) {
        if let Some(sampler) = self.samplers.remove(handle.0) {
            self.dead_samplers.push((self.frame_count, sampler));
        }
    }

    fn create_graphics_pipeline(&mut self, desc: &GraphicsPipelineCreateInfo) -> BackendPipeline {
        let device = self.get_device().clone();
        let id = self.next_id();
        debug!("Creating Graphics Pipeline");
        use crate::convert::*;

        // 1. Create Shader Module (once per pipeline setup)
        let create_info = vk::ShaderModuleCreateInfo::default().code(unsafe {
            std::slice::from_raw_parts(
                desc.shader_module.bytecode.as_ptr() as *const u32,
                desc.shader_module.bytecode.len() / 4,
            )
        });

        let module = unsafe { device.handle.create_shader_module(&create_info, None) }
            .expect("Shader module creation failed");
        self.shader_modules.push(module);

        let mut stages = Vec::new();

        // Create CStrings first to ensure stable pointers
        let entry_points: Vec<std::ffi::CString> = desc
            .shader_module
            .stages
            .iter()
            .map(|s| std::ffi::CString::new(s.entry_point.as_str()).unwrap())
            .collect();

        for (stage_info, entry_point_cstr) in desc.shader_module.stages.iter().zip(&entry_points) {
            let stage_flag = if stage_info.stage.contains(ShaderStageFlags::Vertex) {
                vk::ShaderStageFlags::VERTEX
            } else if stage_info.stage.contains(ShaderStageFlags::Fragment) {
                vk::ShaderStageFlags::FRAGMENT
            } else if stage_info.stage.contains(ShaderStageFlags::Compute) {
                vk::ShaderStageFlags::COMPUTE
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

        let tessellation = vk::PipelineTessellationStateCreateInfo::default()
            .patch_control_points(desc.tessellation_state.patch_control_points);

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

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&attachments)
            .logic_op_enable(desc.render_targets.logic_op.is_some())
            .logic_op(convert_logic_op(
                desc.render_targets.logic_op.unwrap_or(LogicOp::NoOp),
            ));

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
        let mut pushable_sets_mask = 0;
        if !set_bindings.is_empty() {
            let max_set = *set_bindings.keys().max().unwrap();
            for i in 0..=max_set {
                let bindings = set_bindings.get(&i).map(|v| v.as_slice()).unwrap_or(&[]);
                let mut layout_info =
                    vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);

                // Enable Push Descriptors for Set 0 (implied by backend requirement)
                if i == 0 {
                    layout_info =
                        layout_info.flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
                    pushable_sets_mask |= 1 << 0;
                }

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
        // (pipeline_layouts field was removed, descriptor layouts are stored in PhysicalPipeline)

        // Push Constants from reflection
        let pc_ranges: Vec<vk::PushConstantRange> = desc
            .shader_module
            .reflection
            .push_constants
            .iter()
            .map(|pc| vk::PushConstantRange {
                stage_flags: crate::convert::convert_shader_stage_flags(
                    ShaderStageFlags::from_bits_truncate(pc.stage_flags.bits()),
                ),
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
        let mut color_formats = Vec::new();
        for rt in &desc.render_targets.color_targets {
            color_formats.push(convert_format(rt.format));
        }

        let depth_format = desc
            .render_targets
            .depth_stencil_format
            .map(|f| convert_format(f))
            .unwrap_or(vk::Format::UNDEFINED);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format);

        debug!(
            "Pipeline {:?} formats: color={:?}, depth={:?}",
            id, color_formats, depth_format
        );

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(desc.depth_stencil_state.depth_test_enable)
            .depth_write_enable(desc.depth_stencil_state.depth_write_enable)
            .depth_compare_op(convert_compare_op(
                desc.depth_stencil_state.depth_compare_op,
            ))
            .depth_bounds_test_enable(desc.depth_stencil_state.depth_bounds_test_enable)
            .stencil_test_enable(desc.depth_stencil_state.stencil_test_enable)
            .front(convert_stencil_op_state(&desc.depth_stencil_state.front))
            .back(convert_stencil_op_state(&desc.depth_stencil_state.back))
            .min_depth_bounds(desc.depth_stencil_state.min_depth_bounds)
            .max_depth_bounds(desc.depth_stencil_state.max_depth_bounds);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .tessellation_state(&tessellation)
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

        // Create PhysicalPipeline struct
        let physical = PhysicalPipeline {
            handle: pipeline,
            layout: pipeline_layout,
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            set_layouts: descriptor_set_layouts,
            pushable_sets_mask,
        };

        let physical_handle = self.pipeline_resources.insert(physical);
        BackendPipeline(physical_handle)
    }

    fn create_compute_pipeline(&mut self, desc: &ComputePipelineCreateInfo) -> BackendPipeline {
        let device = self.get_device().clone();
        let _id = self.next_id();
        debug!("Creating Compute Pipeline");
        use crate::convert::*;

        // 1. Create Shader Module
        let create_info = vk::ShaderModuleCreateInfo::default().code(unsafe {
            std::slice::from_raw_parts(
                desc.shader_module.bytecode.as_ptr() as *const u32,
                desc.shader_module.bytecode.len() / 4,
            )
        });

        let module = unsafe { device.handle.create_shader_module(&create_info, None) }
            .expect("Shader module creation failed");
        self.shader_modules.push(module);

        // Compute has exactly one stage
        let entry_point =
            std::ffi::CString::new(desc.shader_module.stages[0].entry_point.as_str()).unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry_point);

        // 2. Layout
        let mut set_bindings: HashMap<u32, Vec<vk::DescriptorSetLayoutBinding>> = HashMap::new();
        for binding in &desc.shader_module.reflection.bindings {
            let descriptor_type = convert_binding_type_to_descriptor(binding.binding_type.clone());
            let stage_flags = vk::ShaderStageFlags::COMPUTE;

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

        let mut descriptor_set_layouts = Vec::new();
        let mut pushable_sets_mask = 0;
        if !set_bindings.is_empty() {
            let max_set = *set_bindings.keys().max().unwrap();
            for i in 0..=max_set {
                let bindings = set_bindings.get(&i).map(|v| v.as_slice()).unwrap_or(&[]);
                let mut layout_info =
                    vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);

                // Enable Push Descriptors for Set 0 (implied by backend requirement)
                if i == 0 {
                    layout_info =
                        layout_info.flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
                    pushable_sets_mask |= 1 << 0;
                }

                let layout = unsafe {
                    device
                        .handle
                        .create_descriptor_set_layout(&layout_info, None)
                        .expect("Failed to create descriptor set layout")
                };

                descriptor_set_layouts.push(layout);
                self.descriptor_set_layouts.push(layout);
            }
        }

        // (pipeline_layouts field was removed, descriptor layouts are stored in PhysicalPipeline)

        let pc_ranges: Vec<vk::PushConstantRange> = desc
            .shader_module
            .reflection
            .push_constants
            .iter()
            .map(|pc| vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: pc.offset,
                size: pc.size,
            })
            .collect();

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&pc_ranges);

        let pipeline_layout =
            unsafe { device.handle.create_pipeline_layout(&layout_info, None) }.unwrap();

        // 3. Pipeline
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            device.handle.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .expect("Compute pipeline creation failed")[0];

        // Create PhysicalPipeline struct
        let physical = PhysicalPipeline {
            handle: pipeline,
            layout: pipeline_layout,
            bind_point: vk::PipelineBindPoint::COMPUTE,
            set_layouts: descriptor_set_layouts,
            pushable_sets_mask,
        };

        let handle = self.pipeline_resources.insert(physical);
        BackendPipeline(handle)
    }

    fn upload_buffer(
        &mut self,
        handle: BackendBuffer,
        data: &[u8],
        offset: u64,
    ) -> Result<(), String> {
        let device = self.get_device().clone();
        if let Some(buf) = self.buffers.get_mut(handle.0) {
            if let Some(alloc) = &mut buf.allocation {
                unsafe {
                    let allocator = device.allocator.lock().unwrap();
                    let ptr = allocator.map_memory(alloc).map_err(|e| e.to_string())?;

                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        ptr.add(offset as usize),
                        data.len(),
                    );

                    allocator.unmap_memory(alloc);
                }
                Ok(())
            } else {
                Err("Buffer has no allocation (external?)".to_string())
            }
        } else {
            Err(format!("Buffer not found: {:?}", handle))
        }
    }
}

impl RenderBackendInternal for VulkanBackend {
    fn begin_frame(&mut self) {
        if self.frame_started {
            return;
        }

        let device = self.get_device().clone();
        self.global_frame_index = (self.global_frame_index + 1) % self.frame_contexts.len();
        self.frame_count += 1;
        self.cpu_timeline += 1;

        let ctx = &mut self.frame_contexts[self.global_frame_index];

        // Wait for this frame slot to be ready
        if ctx.last_completion_value > 0 {
            let semaphores = [self.timeline_sem];
            let values = [ctx.last_completion_value];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            unsafe {
                device
                    .handle
                    .wait_semaphores(&wait_info, u64::MAX)
                    .expect("Failed to wait for frame timeline");
            }
        }

        // Reset the pools for this frame
        unsafe {
            device
                .handle
                .reset_command_pool(ctx.command_pool, vk::CommandPoolResetFlags::empty())
                .expect("Failed to reset command pool");
            device
                .handle
                .reset_descriptor_pool(ctx.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .expect("Failed to reset descriptor pool");
        }

        ctx.cursor = 0;
        ctx.submitted_cursor = 0;
        self.frame_started = true;
    }

    fn end_frame(&mut self) {
        self.garbage_collect();
        self.frame_started = false;
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
        if let Some(physical) = self.images.get_mut(handle.0) {
            let desc = physical.desc;
            self.transient_image_pool
                .entry(desc)
                .or_default()
                .push(handle.0);
        }
    }

    fn release_transient_buffer(&mut self, handle: BackendBuffer) {
        if let Some(physical) = self.buffers.get_mut(handle.0) {
            let desc = physical.desc;
            self.transient_buffer_pool
                .entry(desc)
                .or_default()
                .push(handle.0);
        }
    }

    fn garbage_collect(&mut self) {
        let safe_threshold = self.frame_count.saturating_sub(10);

        let mut i = 0;
        while i < self.dead_semaphores.len() {
            if self.dead_semaphores[i].0 <= safe_threshold {
                let (_, _, sem_id) = self.dead_semaphores.swap_remove(i);
                self.recycled_semaphores.push(sem_id);
            } else {
                i += 1;
            }
        }
    }

    fn acquire_swapchain_image(
        &mut self,
        window: WindowHandle,
    ) -> Result<Option<(BackendImage, u64, u32)>, String> {
        let device = self.get_device().clone();
        let frame_slot = self.global_frame_index;

        loop {
            let (sc_handle, acquire_sem_id, semaphore) = {
                let ctx = self
                    .windows
                    .get_mut(&window.0)
                    .ok_or("Invalid window handle")?;
                let size = ctx.raw.handle.drawable_size();
                if size.0 == 0 || size.1 == 0 {
                    return Ok(None);
                }

                if ctx.swapchain.is_none() {
                    let sc_res = crate::swapchain::VulkanSwapchain::new(
                        device.clone(),
                        ctx.raw.surface,
                        size.0,
                        size.1,
                        ctx.config,
                    );

                    match sc_res {
                        Ok(sc) => ctx.swapchain = Some(sc),
                        Err(e) if e == "ZeroExtent" => return Ok(None),
                        Err(e) => return Err(e),
                    }
                }

                let swapchain = ctx.swapchain.as_ref().unwrap();
                let sem_id =
                    ctx.acquire_semaphore_ids[frame_slot % ctx.acquire_semaphore_ids.len()];
                let sem = ctx.acquire_semaphores[frame_slot % ctx.acquire_semaphores.len()];

                (swapchain.handle, sem_id, sem)
            };

            let fp = self.swapchain_loader.as_ref().unwrap();
            let res =
                unsafe { fp.acquire_next_image(sc_handle, u64::MAX, semaphore, vk::Fence::null()) };

            match res {
                Ok((index, suboptimal)) => {
                    if suboptimal {
                        debug!("Swapchain is suboptimal, invalidating for recreation");
                        let images_to_remove = {
                            let ctx = self.windows.get_mut(&window.0).unwrap();
                            let sc = ctx.swapchain.take().unwrap();
                            let imgs = sc.images.clone();
                            ctx.swapchain = Some(sc); // Put it back if we still want to use it
                            imgs
                        };
                        unsafe {
                            self.get_device().handle.device_wait_idle().ok();
                        }
                        self.unregister_swapchain_images(&images_to_remove);
                        let ctx = self.windows.get_mut(&window.0).unwrap();
                        ctx.swapchain = None;
                    }

                    let ctx = self.windows.get_mut(&window.0).unwrap();
                    let swapchain = ctx.swapchain.as_ref().unwrap();
                    ctx.current_acquire_sem_id = Some(acquire_sem_id);
                    ctx.current_image_index = Some(index);

                    let image_raw = swapchain.images[index as usize];
                    let image_id = image_raw.as_raw();
                    let arena_id = if let Some(&id) = self.external_to_physical.get(&image_id) {
                        if let Some(img) = self.images.get_mut(id) {
                            img.last_layout = vk::ImageLayout::UNDEFINED;
                            img.last_access = vk::AccessFlags2::empty();
                            img.last_stage = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
                        }
                        id
                    } else {
                        let view_raw = swapchain.image_views[index as usize];
                        let new_id = self.images.insert(PhysicalImage {
                            image: image_raw,
                            view: view_raw,
                            allocation: None,
                            desc: ImageDesc::new(
                                swapchain.extent.width,
                                swapchain.extent.height,
                                crate::convert::convert_vk_format(swapchain.format),
                            ),
                            format: swapchain.format,
                            last_layout: vk::ImageLayout::UNDEFINED,
                            last_access: vk::AccessFlags2::empty(),
                            last_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                            last_write_frame: 0,
                        });
                        self.external_to_physical.insert(image_id, new_id);
                        new_id
                    };

                    return Ok(Some((BackendImage(arena_id), acquire_sem_id, index)));
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    debug!("Swapchain out of date during acquire, invalidating...");
                    let images_to_remove = {
                        let ctx = self.windows.get_mut(&window.0).unwrap();
                        if let Some(sc) = ctx.swapchain.take() {
                            sc.images.clone()
                        } else {
                            Vec::new()
                        }
                    };
                    unsafe {
                        self.get_device().handle.device_wait_idle().ok();
                    }
                    self.unregister_swapchain_images(&images_to_remove);
                    continue; // Loop and recreate
                }
                Err(e) => {
                    return Err(format!("Failed to acquire swapchain image: {}", e));
                }
            }
        }
    }

    fn submit(
        &mut self,
        _batch: CommandBatch,
        _wait_sems: &[u64],
        _signal_sems: &[u64],
    ) -> Result<u64, String> {
        // Timeline advancement
        self.cpu_timeline += 1;
        let signal_value = self.cpu_timeline;

        // Collect all binary semaphores from windows that acquired images
        // 1. Collect Active Window Contexts (Borrow scope)
        let mut active_windows = Vec::with_capacity(2);
        let frame_slot = self.global_frame_index;
        for ctx in self.windows.values_mut() {
            if let (Some(a_id), Some(i)) = (
                ctx.current_acquire_sem_id.take(),
                ctx.current_image_index.take(),
            ) {
                let release_sem = ctx.present_semaphores[frame_slot % ctx.present_semaphores.len()];
                let acquire_sem = self.semaphores.get(a_id).cloned().unwrap();
                active_windows.push((
                    ctx.swapchain.as_ref().unwrap().handle,
                    i,
                    acquire_sem,
                    release_sem,
                ));
            }
        }

        // 2. Process Binary Semaphores (Outside borrow scope)
        let mut wait_binary: Vec<vk::Semaphore> = Vec::with_capacity(active_windows.len());
        let mut signal_binary: Vec<vk::Semaphore> = Vec::with_capacity(active_windows.len());
        let mut present_info = Vec::with_capacity(active_windows.len());

        for (sc_handle, image_index, acquire_sem, release_sem) in active_windows {
            wait_binary.push(acquire_sem);
            signal_binary.push(release_sem);
            present_info.push((sc_handle, image_index, release_sem));
        }

        let device = self.get_device().clone();

        let wait_values = [0u64; 8];
        let mut signal_values = [0u64; 8];
        signal_values[0] = signal_value;

        let num_binary = signal_binary.len();
        let mut all_signals = Vec::with_capacity(num_binary + 1);
        all_signals.push(self.timeline_sem);
        all_signals.extend(&signal_binary);

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT; 8];

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values[..wait_binary.len()])
            .signal_semaphore_values(&signal_values[..all_signals.len()]);

        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_info)
            .wait_semaphores(&wait_binary)
            .wait_dst_stage_mask(&wait_stages[..wait_binary.len()])
            .signal_semaphores(&all_signals);

        // Collect all command buffers from current frame context (Only those not yet submitted)
        let frame_ctx = &mut self.frame_contexts[self.global_frame_index];
        let cmds =
            &frame_ctx.allocated_command_buffers[frame_ctx.submitted_cursor..frame_ctx.cursor];
        let submit_info = submit_info.command_buffers(cmds);

        unsafe {
            device
                .handle
                .queue_submit(device.graphics_queue, &[submit_info], vk::Fence::null())
                .map_err(|e| e.to_string())?;
        }

        // Update submitted_cursor to current cursor
        frame_ctx.submitted_cursor = frame_ctx.cursor;

        // Present all windows
        let fp = self.swapchain_loader.as_ref().unwrap();
        for (swapchain, index, wait_sem) in present_info {
            let swapchains = [swapchain];
            let indices = [index];
            let wait_sems = [wait_sem];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_sems)
                .swapchains(&swapchains)
                .image_indices(&indices);

            unsafe {
                fp.queue_present(device.graphics_queue, &present_info).ok(); // Presentation errors handled on next acquire
            }
        }

        // Advance slot's last completion value
        frame_ctx.last_completion_value = signal_value;

        Ok(signal_value)
    }

    fn begin_pass(&mut self, desc: PassDescriptor<'_>, pass: &dyn RenderPass) -> u64 {
        let device = self.get_device().clone();

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

        // 2. Allocate Command Buffer from Global Pool
        let frame_ctx = &mut self.frame_contexts[self.global_frame_index];
        let cmd = if frame_ctx.cursor < frame_ctx.allocated_command_buffers.len() {
            let cmd = frame_ctx.allocated_command_buffers[frame_ctx.cursor];
            frame_ctx.cursor += 1;
            cmd
        } else {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(frame_ctx.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd = unsafe { device.handle.allocate_command_buffers(&alloc_info).unwrap()[0] };
            frame_ctx.allocated_command_buffers.push(cmd);
            frame_ctx.cursor += 1;
            cmd
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
        let is_compute = desc.domain != i3_gfx::graph::types::PassDomain::Graphics
            || if let Some(h) = desc.pipeline {
                self.pipeline_resources
                    .get(h.0.0)
                    .map(|p| p.bind_point == vk::PipelineBindPoint::COMPUTE)
                    .unwrap_or(false)
            } else {
                false
            };

        let mut ctx = VulkanPassContext {
            cmd,
            device: self.get_device().clone(),
            present_request: None,
            backend: self as *mut VulkanBackend,
            pipeline: None,
            descriptor_pool: self.frame_contexts
                [self.global_frame_index % self.frame_contexts.len()]
            .descriptor_pool,
            current_pipeline_layout: vk::PipelineLayout::null(),
            current_bind_point: vk::PipelineBindPoint::GRAPHICS,
        };

        // If pipeline is set, determine bind point and bind it
        if let Some(pipe_handle) = desc.pipeline {
            if let Some(pipe) = self.pipeline_resources.get(pipe_handle.0.0).cloned() {
                unsafe {
                    device
                        .handle
                        .cmd_bind_pipeline(cmd, pipe.bind_point, pipe.handle);
                }
                ctx.pipeline = Some(pipe.clone());
                ctx.current_pipeline_layout = pipe.layout;
                ctx.current_bind_point = pipe.bind_point;

                // Bind Declarative Descriptor Sets
                for (set_index, writes) in desc.descriptor_sets {
                    let set_index = *set_index;
                    if (pipe.pushable_sets_mask & (1 << set_index)) != 0 {
                        // Push Descriptor Path
                        let mut buffer_infos = Vec::with_capacity(writes.len());
                        let mut image_infos = Vec::with_capacity(writes.len());

                        // Pass 1: Resolve and collect infos
                        for write in writes.iter() {
                            match write.descriptor_type {
                                i3_gfx::graph::pipeline::BindingType::UniformBuffer
                                | i3_gfx::graph::pipeline::BindingType::StorageBuffer
                                | i3_gfx::graph::pipeline::BindingType::RawBuffer
                                | i3_gfx::graph::pipeline::BindingType::MutableRawBuffer => {
                                    if let Some(info) = &write.buffer_info {
                                        let pid = self.resolve_buffer(info.buffer).0;
                                        if let Some(buf) = self.buffers.get(pid) {
                                            buffer_infos.push(vk::DescriptorBufferInfo {
                                                buffer: buf.buffer,
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
                                | i3_gfx::graph::pipeline::BindingType::Texture
                                | i3_gfx::graph::pipeline::BindingType::StorageTexture
                                | i3_gfx::graph::pipeline::BindingType::Sampler => {
                                    if let Some(info) = &write.image_info {
                                        let pid = self.resolve_image(info.image).0;
                                        if let Some(img) = self.images.get(pid) {
                                            let layout = match info.image_layout {
                                                i3_gfx::graph::backend::DescriptorImageLayout::General => vk::ImageLayout::GENERAL,
                                                i3_gfx::graph::backend::DescriptorImageLayout::ShaderReadOnlyOptimal => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                            };
                                            let vk_sampler =
                                                if let Some(sampler_handle) = info.sampler {
                                                    self.samplers
                                                        .get(sampler_handle.0)
                                                        .cloned()
                                                        .unwrap_or(vk::Sampler::null())
                                                } else {
                                                    vk::Sampler::null()
                                                };
                                            image_infos.push(vk::DescriptorImageInfo {
                                                sampler: vk_sampler,
                                                image_view: img.view,
                                                image_layout: layout,
                                            });
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }

                        // Pass 2: Build writes (using collected infos)
                        let mut descriptor_writes = Vec::with_capacity(writes.len());
                        let mut buf_ptr = 0;
                        let mut img_ptr = 0;

                        for write in writes.iter() {
                            let mut vk_write = vk::WriteDescriptorSet::default()
                                .dst_binding(write.binding)
                                .dst_array_element(write.array_element)
                                .descriptor_count(1);

                            match write.descriptor_type {
                                i3_gfx::graph::pipeline::BindingType::UniformBuffer => {
                                    if buf_ptr < buffer_infos.len() {
                                        vk_write = vk_write
                                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                            .buffer_info(std::slice::from_ref(
                                                &buffer_infos[buf_ptr],
                                            ));
                                        buf_ptr += 1;
                                        descriptor_writes.push(vk_write);
                                    }
                                }
                                i3_gfx::graph::pipeline::BindingType::StorageBuffer
                                | i3_gfx::graph::pipeline::BindingType::RawBuffer
                                | i3_gfx::graph::pipeline::BindingType::MutableRawBuffer => {
                                    if buf_ptr < buffer_infos.len() {
                                        vk_write = vk_write
                                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                            .buffer_info(std::slice::from_ref(
                                                &buffer_infos[buf_ptr],
                                            ));
                                        buf_ptr += 1;
                                        descriptor_writes.push(vk_write);
                                    }
                                }
                                i3_gfx::graph::pipeline::BindingType::CombinedImageSampler => {
                                    if img_ptr < image_infos.len() {
                                        vk_write = vk_write
                                            .descriptor_type(
                                                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                            )
                                            .image_info(std::slice::from_ref(
                                                &image_infos[img_ptr],
                                            ));
                                        img_ptr += 1;
                                        descriptor_writes.push(vk_write);
                                    }
                                }
                                i3_gfx::graph::pipeline::BindingType::Texture => {
                                    if img_ptr < image_infos.len() {
                                        vk_write = vk_write
                                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                            .image_info(std::slice::from_ref(
                                                &image_infos[img_ptr],
                                            ));
                                        img_ptr += 1;
                                        descriptor_writes.push(vk_write);
                                    }
                                }
                                i3_gfx::graph::pipeline::BindingType::Sampler => {
                                    if img_ptr < image_infos.len() {
                                        vk_write = vk_write
                                            .descriptor_type(vk::DescriptorType::SAMPLER)
                                            .image_info(std::slice::from_ref(
                                                &image_infos[img_ptr],
                                            ));
                                        img_ptr += 1;
                                        descriptor_writes.push(vk_write);
                                    }
                                }
                                i3_gfx::graph::pipeline::BindingType::StorageTexture => {
                                    if img_ptr < image_infos.len() {
                                        vk_write = vk_write
                                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                            .image_info(std::slice::from_ref(
                                                &image_infos[img_ptr],
                                            ));
                                        img_ptr += 1;
                                        descriptor_writes.push(vk_write);
                                    }
                                }
                                _ => {}
                            }
                        }

                        unsafe {
                            device.push_descriptor.cmd_push_descriptor_set(
                                cmd,
                                pipe.bind_point,
                                pipe.layout,
                                set_index,
                                &descriptor_writes,
                            );
                        }
                    } else {
                        // Pool Path (Static allocation from frame pool)
                        let set_handle = self
                            .allocate_descriptor_set(pipe_handle, set_index)
                            .unwrap();
                        self.update_descriptor_set(set_handle, writes);
                        ctx.bind_descriptor_set(set_index, set_handle);
                    }
                }
            }
        }

        // Dynamic Viewport/Scissor setup (Use resolved extent)
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
                self.get_image_state(usage, is_write, ctx.current_bind_point);

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
            let (target_access, target_stage) =
                self.get_buffer_state(usage, ctx.current_bind_point);
            if let Some(barrier) = self.get_buffer_barrier(pid, target_access, target_stage) {
                self.buffer_barrier_scratch.push(barrier);
            }
        }

        // --- Unified Pipeline Barrier Emission ---
        if !self.image_barrier_scratch.is_empty() || !self.buffer_barrier_scratch.is_empty() {
            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(&self.image_barrier_scratch)
                .buffer_memory_barriers(&self.buffer_barrier_scratch);

            unsafe {
                device.handle.cmd_pipeline_barrier2(cmd, &dependency_info);
            }
        }

        if !is_compute && (color_count > 0 || depth_attachment_info.is_some()) {
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: viewport_extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachments[..color_count]);

            let rendering_info = if let Some(depth) = &depth_attachment_info {
                rendering_info.depth_attachment(depth)
            } else {
                rendering_info
            };

            unsafe {
                device.handle.cmd_begin_rendering(cmd, &rendering_info);
            }
        }

        pass.execute(&mut ctx);

        if !is_compute && (color_count > 0 || depth_attachment_info.is_some()) {
            unsafe {
                device.handle.cmd_end_rendering(cmd);
            }
        }

        // Handle explicit transition for Present if requested
        if let Some(handle) = ctx.present_request {
            let pid = self.resolve_image(handle).0;
            if let Some(barrier) = self.get_image_barrier(
                pid,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            ) {
                let barriers = [barrier];
                let dependency_info =
                    vk::DependencyInfo::default().image_memory_barriers(&barriers);
                unsafe {
                    device.handle.cmd_pipeline_barrier2(cmd, &dependency_info);
                }
            }
        }

        unsafe {
            device.handle.end_command_buffer(cmd).unwrap();
        }

        self.cpu_timeline
    }

    fn resolve_image(&self, handle: ImageHandle) -> BackendImage {
        if let Some(phy) = self.external_to_physical.get(&handle.0.0).cloned() {
            BackendImage(phy)
        } else {
            BackendImage(handle.0.0)
        }
    }

    fn resolve_buffer(&self, handle: BufferHandle) -> BackendBuffer {
        if let Some(phy) = self.external_to_physical.get(&handle.0.0).cloned() {
            BackendBuffer(phy)
        } else {
            BackendBuffer(handle.0.0)
        }
    }

    fn resolve_pipeline(&self, handle: PipelineHandle) -> BackendPipeline {
        BackendPipeline(handle.0.0)
    }

    fn register_external_image(&mut self, handle: ImageHandle, physical: BackendImage) {
        self.external_to_physical.insert(handle.0.0, physical.0);
    }

    fn register_external_buffer(&mut self, handle: BufferHandle, physical: BackendBuffer) {
        self.external_to_physical.insert(handle.0.0, physical.0);
    }

    fn wait_for_timeline(&self, value: u64, timeout_ns: u64) -> Result<(), String> {
        let device = self.get_device();
        let sems = [self.timeline_sem];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&sems)
            .values(&values);

        unsafe {
            device
                .handle
                .wait_semaphores(&wait_info, timeout_ns)
                .map_err(|e| e.to_string())
        }
    }

    fn allocate_descriptor_set(
        &mut self,
        pipeline: PipelineHandle,
        set_index: u32,
    ) -> Result<DescriptorSetHandle, String> {
        let pipeline_id = pipeline.0.0;
        let layout = {
            let p = self
                .pipeline_resources
                .get(pipeline_id)
                .ok_or_else(|| format!("Pipeline layout not found for {:?}", pipeline))?;

            if set_index as usize >= p.set_layouts.len() {
                return Err(format!(
                    "Set index {} out of bounds for pipeline {:?}",
                    set_index, pipeline
                ));
            }
            p.set_layouts[set_index as usize]
        };

        let layouts_to_alloc = [layout];
        let pool = self.static_descriptor_pool;
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts_to_alloc);

        let sets = unsafe {
            self.get_device()
                .handle
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| format!("Failed to allocate descriptor set: {}", e))?
        };

        let set = sets[0];
        let handle_id = self.descriptor_sets.insert(set);

        Ok(DescriptorSetHandle(handle_id))
    }

    fn update_descriptor_set(&mut self, set: DescriptorSetHandle, writes: &[DescriptorWrite]) {
        let vk_set = if let Some(s) = self.descriptor_sets.get(set.0) {
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
                        if let Some(buf) = self.buffers.get(info.buffer.0.0) {
                            buffer_infos.push(vk::DescriptorBufferInfo {
                                buffer: buf.buffer,
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
                | i3_gfx::graph::pipeline::BindingType::Texture
                | i3_gfx::graph::pipeline::BindingType::StorageTexture
                | i3_gfx::graph::pipeline::BindingType::Sampler => {
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

                        if let Some(img) = self.images.get(physical_id) {
                            let layout = match info.image_layout {
                                i3_gfx::graph::backend::DescriptorImageLayout::General => vk::ImageLayout::GENERAL,
                                i3_gfx::graph::backend::DescriptorImageLayout::ShaderReadOnlyOptimal => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            };

                            let vk_sampler = if let Some(sampler_handle) = info.sampler {
                                self.samplers
                                    .get(sampler_handle.0)
                                    .cloned()
                                    .unwrap_or(vk::Sampler::null())
                            } else {
                                vk::Sampler::null()
                            };

                            image_infos.push(vk::DescriptorImageInfo {
                                sampler: vk_sampler,
                                image_view: img.view,
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
                i3_gfx::graph::pipeline::BindingType::StorageTexture => {
                    vk_write = vk_write.descriptor_type(vk::DescriptorType::STORAGE_IMAGE);
                    if img_idx < image_infos.len() {
                        vk_write = vk_write.image_info(&image_infos[img_idx..=img_idx]);
                        img_idx += 1;
                        descriptor_writes.push(vk_write);
                    }
                }
                i3_gfx::graph::pipeline::BindingType::Sampler => {
                    vk_write = vk_write.descriptor_type(vk::DescriptorType::SAMPLER);
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
            self.get_device()
                .handle
                .update_descriptor_sets(&descriptor_writes, &[]);
        }
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
                    for (_, buffer, alloc) in &mut self.dead_buffers {
                        allocator.destroy_buffer(*buffer, alloc);
                    }
                    for (_, image, view, alloc) in &mut self.dead_images {
                        device.handle.destroy_image_view(*view, None);
                        allocator.destroy_image(*image, alloc);
                    }
                    for (_, sampler) in &mut self.dead_samplers {
                        device.handle.destroy_sampler(*sampler, None);
                    }
                    for (_, _, sem_handle) in &self.dead_semaphores {
                        device.handle.destroy_semaphore(*sem_handle, None);
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
                info!("Vulkan Backend shutdown complete.");
            }
        }
    }
}

pub struct VulkanPassContext {
    pub cmd: vk::CommandBuffer,
    pub device: Arc<crate::device::VulkanDevice>,
    pub present_request: Option<ImageHandle>,
    pub backend: *mut VulkanBackend,
    pub pipeline: Option<PhysicalPipeline>,
    pub descriptor_pool: vk::DescriptorPool,
    pub current_pipeline_layout: vk::PipelineLayout,
    pub current_bind_point: vk::PipelineBindPoint,
}

impl VulkanPassContext {
    pub fn backend(&self) -> &VulkanBackend {
        unsafe { &*self.backend }
    }

    pub fn backend_mut(&mut self) -> &mut VulkanBackend {
        unsafe { &mut *self.backend }
    }
}

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

    fn unregister_swapchain_images(&mut self, images: &[vk::Image]) {
        for &image in images {
            let vk_handle = image.as_raw();
            if let Some(arena_id) = self.external_to_physical.remove(&vk_handle) {
                self.images.remove(arena_id);
            }
        }
    }
}

impl PassContext for VulkanPassContext {
    fn bind_pipeline(&mut self, pipeline: PipelineHandle) {
        let p = if let Some(p) = self.backend().pipeline_resources.get(pipeline.0.0) {
            p.clone()
        } else {
            return;
        };

        unsafe {
            self.device
                .handle
                .cmd_bind_pipeline(self.cmd, p.bind_point, p.handle);
        }
        self.pipeline = Some(p.clone());
        self.current_pipeline_layout = p.layout;
        self.current_bind_point = p.bind_point;
    }

    fn bind_vertex_buffer(&mut self, binding: u32, handle: BufferHandle) {
        if let Some(buf) = self.backend().buffers.get(handle.0.0) {
            unsafe {
                self.device
                    .handle
                    .cmd_bind_vertex_buffers(self.cmd, binding, &[buf.buffer], &[0]);
            }
        }
    }

    fn bind_index_buffer(&mut self, handle: BufferHandle, index_type: IndexType) {
        if let Some(buf) = self.backend().buffers.get(handle.0.0) {
            let vk_type = match index_type {
                IndexType::Uint16 => vk::IndexType::UINT16,
                IndexType::Uint32 => vk::IndexType::UINT32,
            };
            unsafe {
                self.device
                    .handle
                    .cmd_bind_index_buffer(self.cmd, buf.buffer, 0, vk_type);
            }
        }
    }

    fn bind_descriptor_set(&mut self, set_index: u32, handle: DescriptorSetHandle) {
        if let Some(set) = self.backend().descriptor_sets.get(handle.0) {
            unsafe {
                self.device.handle.cmd_bind_descriptor_sets(
                    self.cmd,
                    self.current_bind_point,
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

    fn push_bytes(
        &mut self,
        stages: i3_gfx::graph::pipeline::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            self.device.handle.cmd_push_constants(
                self.cmd,
                self.current_pipeline_layout,
                crate::convert::convert_shader_stage_flags(stages),
                offset,
                data,
            );
        }
    }

    fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        unsafe {
            let device = self.backend().get_device();
            device.handle.cmd_dispatch(self.cmd, x, y, z);
        }
    }

    fn clear_buffer(&mut self, buffer: i3_gfx::graph::types::BufferHandle, clear_value: u32) {
        let physical_id = if let Some(&phy) = self.backend().external_to_physical.get(&buffer.0.0) {
            phy
        } else {
            buffer.0.0
        };

        if let Some(buf) = self.backend().buffers.get(physical_id) {
            unsafe {
                let device = self.backend().get_device();
                device.handle.cmd_fill_buffer(
                    self.cmd,
                    buf.buffer,
                    0,
                    ash::vk::WHOLE_SIZE,
                    clear_value,
                );
            }
        }
    }

    fn present(&mut self, image: i3_gfx::graph::types::ImageHandle) {
        self.present_request = Some(image);
    }
}
