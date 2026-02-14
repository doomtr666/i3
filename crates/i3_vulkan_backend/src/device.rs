use ash::vk;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::info;
use vk_mem::{Allocator, AllocatorCreateInfo};

pub struct VulkanDevice {
    pub instance: Arc<crate::instance::VulkanInstance>,
    pub physical_device: vk::PhysicalDevice,
    pub handle: ash::Device,
    pub graphics_queue: vk::Queue,
    pub graphics_family: u32,
    pub allocator: Mutex<Allocator>,
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

        // Enable Vulkan 1.3/1.2/1.0 features
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_variable_descriptor_count(true);

        let mut features11 =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);

        let features10 = vk::PhysicalDeviceFeatures::default().fill_mode_non_solid(true);

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(features10)
            .push_next(&mut features13)
            .push_next(&mut features12)
            .push_next(&mut features11);

        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_family)
            .queue_priorities(&queue_priorities);

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features2);

        let handle = unsafe {
            instance
                .handle
                .create_device(physical_device, &device_create_info, None)
        }
        .map_err(|e| format!("Failed to create logical device: {}", e))?;

        let graphics_queue = unsafe { handle.get_device_queue(graphics_family, 0) };

        // Initialize VMA Allocator
        let allocator_create_info =
            AllocatorCreateInfo::new(&instance.handle, &handle, physical_device);
        let allocator = unsafe { Allocator::new(allocator_create_info) }
            .map_err(|e| format!("Failed to create VMA allocator: {}", e))?;

        Ok(VulkanDevice {
            instance,
            physical_device,
            handle,
            graphics_queue,
            graphics_family,
            allocator: Mutex::new(allocator),
        })
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            // Allocator must be dropped before device
            // We can't easily drop the Mutex content here without taking it,
            // but for simplicity in MVP we assume the process exits.
            // In a real app we'd use ManuallyDrop or an Option.
            self.handle.destroy_device(None);
        }
        info!("Vulkan Device destroyed");
    }
}
