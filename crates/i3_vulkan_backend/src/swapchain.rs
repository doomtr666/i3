use ash::vk;
use tracing::info;

pub struct VulkanSwapchain {
    pub loader: ash::khr::swapchain::Device,
    pub device: ash::Device,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub image_views: Vec<vk::ImageView>,
}

impl VulkanSwapchain {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        _physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let loader = ash::khr::swapchain::Device::new(instance, device);

        let extent = vk::Extent2D { width, height };
        let format = vk::Format::B8G8R8A8_UNORM; // Default for now
        let color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(3)
            .image_format(format)
            .image_color_space(color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
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
                unsafe { device.create_image_view(&create_info, None) }.unwrap()
            })
            .collect();

        Ok(VulkanSwapchain {
            loader,
            device: device.clone(),
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
                self.device.destroy_image_view(view, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
        info!("Vulkan Swapchain and Views destroyed");
    }
}
