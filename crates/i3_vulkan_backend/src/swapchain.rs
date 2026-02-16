use ash::vk;
use std::sync::Arc;
use tracing::info;

pub struct VulkanSwapchain {
    pub loader: ash::khr::swapchain::Device,
    pub device: Arc<crate::device::VulkanDevice>,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub image_views: Vec<vk::ImageView>,
}

impl VulkanSwapchain {
    pub fn new(
        device: Arc<crate::device::VulkanDevice>,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
        config: i3_gfx::graph::backend::SwapchainConfig,
    ) -> Result<Self, String> {
        let instance = &device.instance.handle;
        let loader = ash::khr::swapchain::Device::new(instance, &device.handle);

        let format = if config.srgb {
            vk::Format::B8G8R8A8_SRGB
        } else {
            vk::Format::B8G8R8A8_UNORM
        };

        let surface_loader =
            ash::khr::surface::Instance::new(&device.instance.entry, &device.instance.handle);
        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(device.physical_device, surface)
                .map_err(|e| format!("Failed to query surface capabilities: {}", e))?
        };

        let mut extent = vk::Extent2D { width, height };
        if capabilities.current_extent.width != u32::MAX {
            extent = capabilities.current_extent;
        } else {
            // Clamp
            extent.width = extent.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            );
            extent.height = extent.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            );
        }

        let color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(device.physical_device, surface)
                .map_err(|e| format!("Failed to query present modes: {}", e))?
        };

        let present_mode = if config.vsync {
            if present_modes.contains(&vk::PresentModeKHR::FIFO) {
                vk::PresentModeKHR::FIFO
            } else if present_modes.contains(&vk::PresentModeKHR::FIFO_RELAXED) {
                vk::PresentModeKHR::FIFO_RELAXED
            } else {
                present_modes[0] // Fallback (should not happen as FIFO is required)
            }
        } else {
            // Prefer Mailbox > Immediate > FifoRelaxed > Fifo
            if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else if present_modes.contains(&vk::PresentModeKHR::FIFO_RELAXED) {
                vk::PresentModeKHR::FIFO_RELAXED
            } else {
                vk::PresentModeKHR::FIFO
            }
        };

        info!("Selected Present Mode: {:?}", present_mode);

        // Also check composite alpha
        let composite_alpha = if capabilities
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else {
            vk::CompositeAlphaFlagsKHR::INHERIT // Fallback
        };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(config.min_image.max(capabilities.min_image_count)) // Ensure min
            .image_format(format)
            .image_color_space(color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform) // Use current transform
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true);

        let handle = unsafe { loader.create_swapchain(&create_info, None) }
            .map_err(|e| format!("Failed to create swapchain: {}", e))?;

        let images = unsafe { loader.get_swapchain_images(handle) }
            .map_err(|e| format!("Failed to get swapchain images: {}", e))?;

        info!("Vulkan Swapchain created with {} images", images.len());

        let image_views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { device.handle.create_image_view(&create_info, None) }.unwrap()
            })
            .collect();

        Ok(VulkanSwapchain {
            loader,
            device,
            handle,
            images,
            format,
            extent,
            image_views,
        })
    }
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        unsafe {
            for &view in &self.image_views {
                self.device.handle.destroy_image_view(view, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
        info!("Vulkan Swapchain and Views destroyed");
    }
}
