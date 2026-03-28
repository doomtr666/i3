use ash::vk;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::{debug, info};
use vk_mem::{Allocator, AllocatorCreateInfo};

use std::mem::ManuallyDrop;

pub struct VulkanDevice {
    pub instance: Arc<crate::instance::VulkanInstance>,
    pub physical_device: vk::PhysicalDevice,
    pub handle: ash::Device,
    pub graphics_queue: vk::Queue,
    pub graphics_family: u32,
    pub compute_queue: Option<vk::Queue>,
    pub compute_family: Option<u32>,
    pub transfer_queue: Option<vk::Queue>,
    pub transfer_family: Option<u32>,
    pub allocator: ManuallyDrop<Mutex<Allocator>>,

    pub dynamic_rendering: ash::khr::dynamic_rendering::Device,
    pub sync2: ash::khr::synchronization2::Device,

    pub accel_struct: Option<ash::khr::acceleration_structure::Device>,
    pub rt_pipeline: Option<ash::khr::ray_tracing_pipeline::Device>,
    pub rt_supported: bool,

    #[cfg(debug_assertions)]
    pub debug_utils: ash::ext::debug_utils::Device,
}

impl VulkanDevice {
    pub fn new(instance: Arc<crate::instance::VulkanInstance>) -> Result<Self, String> {
        let pdevices = unsafe { instance.handle.enumerate_physical_devices() }
            .map_err(|e| format!("Failed to enumerate physical devices: {}", e))?;

        // Simple selection: first discrete GPU, or first GPU
        let physical_device = pdevices
            .iter()
            .find(|&p| {
                let props = unsafe { instance.handle.get_physical_device_properties(*p) };
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .copied()
            .or_else(|| pdevices.first().copied())
            .ok_or_else(|| "No suitable GPU found".to_string())?;

        Self::new_with_physical(instance, physical_device)
    }

    pub fn new_with_physical(
        instance: Arc<crate::instance::VulkanInstance>,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, String> {
        let props = unsafe {
            instance
                .handle
                .get_physical_device_properties(physical_device)
        };
        info!("Selected GPU: {:?}", unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
        });

        // Find graphics queue family
        let queue_families = unsafe {
            instance
                .handle
                .get_physical_device_queue_family_properties(physical_device)
        };
        let graphics_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, prop)| prop.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| "No graphics queue family found".to_string())?;

        let compute_family = queue_families
            .iter()
            .enumerate()
            .find(|(i, prop)| {
                prop.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !prop.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && *i as u32 != graphics_family
            })
            .map(|(i, _)| i as u32);

        let transfer_family = queue_families
            .iter()
            .enumerate()
            .find(|(i, prop)| {
                prop.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !prop.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && !prop.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && *i as u32 != graphics_family
                    && compute_family.map_or(true, |cf| *i as u32 != cf)
            })
            .map(|(i, _)| i as u32);

        // Enable Vulkan 1.3/1.2/1.0 features
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .draw_indirect_count(true)
            .timeline_semaphore(true)
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .scalar_block_layout(true);

        let mut features11 = vk::PhysicalDeviceVulkan11Features::default()
            .shader_draw_parameters(true)
            .storage_buffer16_bit_access(true)
            .uniform_and_storage_buffer16_bit_access(true);

        let features10 = vk::PhysicalDeviceFeatures::default()
            .fill_mode_non_solid(true)
            .sampler_anisotropy(true)
            .shader_int64(true)
            .shader_int16(true);

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(features10);

        // Build the feature-chain manually to ensure nothing is overwritten
        features2.p_next = &mut features13 as *mut _ as *mut _;
        features13.p_next = &mut features12 as *mut _ as *mut _;
        features12.p_next = &mut features11 as *mut _ as *mut _;
        // Keep tracking the end of the chain for RT features later
        let mut last_feature: *mut vk::BaseOutStructure = &mut features11 as *mut _ as *mut _;

        let mut unique_families = std::collections::HashSet::new();
        unique_families.insert(graphics_family);
        if let Some(f) = compute_family {
            unique_families.insert(f);
        }
        if let Some(f) = transfer_family {
            unique_families.insert(f);
        }

        let queue_priorities = [1.0f32];
        let mut queue_create_infos = Vec::new();
        for &family in &unique_families {
            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&queue_priorities),
            );
        }

        let mut device_extensions = vec![
            ash::khr::swapchain::NAME.as_ptr(),
            ash::khr::dynamic_rendering::NAME.as_ptr(),
            ash::khr::synchronization2::NAME.as_ptr(),
        ];

        // Probe for RT extensions
        let available_extensions = unsafe {
            instance
                .handle
                .enumerate_device_extension_properties(physical_device)
        }
        .unwrap_or_default();

        let has_extension = |name: &std::ffi::CStr| {
            available_extensions.iter().any(|ext| {
                let s = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
                s == name
            })
        };

        let has_as = has_extension(ash::khr::acceleration_structure::NAME);
        let has_deferred = has_extension(ash::khr::deferred_host_operations::NAME);
        let has_rt_pipeline = has_extension(ash::khr::ray_tracing_pipeline::NAME);

        let rt_supported = has_as && has_deferred;
        let mut as_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true);
        let mut rt_pipeline_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
            .ray_tracing_pipeline(true);

        if rt_supported {
            info!("Ray Tracing extensions detected and enabled");
            device_extensions.push(ash::khr::acceleration_structure::NAME.as_ptr());
            device_extensions.push(ash::khr::deferred_host_operations::NAME.as_ptr());
            
            unsafe {
                (*last_feature).p_next = &mut as_features as *mut _ as *mut _;
                last_feature = &mut as_features as *mut _ as *mut _;
            }

            if has_rt_pipeline {
                device_extensions.push(ash::khr::ray_tracing_pipeline::NAME.as_ptr());
                unsafe {
                    (*last_feature).p_next = &mut rt_pipeline_features as *mut _ as *mut _;
                }
            }
        }

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features2);

        let handle = unsafe {
            instance
                .handle
                .create_device(physical_device, &device_create_info, None)
        }
        .map_err(|e| format!("Failed to create logical device: {}", e))?;

        let graphics_queue = unsafe { handle.get_device_queue(graphics_family, 0) };
        let compute_queue =
            compute_family.map(|f| unsafe { handle.get_device_queue(f, 0) });
        let transfer_queue =
            transfer_family.map(|f| unsafe { handle.get_device_queue(f, 0) });

        // Load extensions
        let dynamic_rendering = ash::khr::dynamic_rendering::Device::new(&instance.handle, &handle);
        let sync2 = ash::khr::synchronization2::Device::new(&instance.handle, &handle);

        let accel_struct = if rt_supported {
            Some(ash::khr::acceleration_structure::Device::new(&instance.handle, &handle))
        } else {
            None
        };

        let rt_pipeline = if rt_supported && has_rt_pipeline {
            Some(ash::khr::ray_tracing_pipeline::Device::new(&instance.handle, &handle))
        } else {
            None
        };

        #[cfg(debug_assertions)]
        let debug_utils = ash::ext::debug_utils::Device::new(&instance.handle, &handle);

        // Initialize VMA Allocator with BDA support
        let mut allocator_create_info =
            AllocatorCreateInfo::new(&instance.handle, &handle, physical_device);
        allocator_create_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

        let allocator = unsafe { Allocator::new(allocator_create_info) }
            .map_err(|e| format!("Failed to create VMA allocator: {}", e))?;

        Ok(VulkanDevice {
            instance,
            physical_device,
            handle,
            graphics_queue,
            graphics_family,
            compute_queue,
            compute_family,
            transfer_queue,
            transfer_family,
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            dynamic_rendering,
            sync2,
            accel_struct,
            rt_pipeline,
            rt_supported,
            #[cfg(debug_assertions)]
            debug_utils,
        })
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            // Drop allocator explicitly BEFORE destroying device
            ManuallyDrop::drop(&mut self.allocator);

            self.handle.destroy_device(None);
        }
        debug!("Vulkan Device destroyed");
    }
}
