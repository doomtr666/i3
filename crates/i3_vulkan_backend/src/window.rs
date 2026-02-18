use ash::vk;
use ash::vk::Handle;
use std::sync::Arc;
use tracing::debug;

pub struct VulkanWindow {
    pub instance: Arc<crate::instance::VulkanInstance>,
    pub handle: sdl2::video::Window,
    pub surface: vk::SurfaceKHR,
}

impl VulkanWindow {
    pub fn new(
        instance: Arc<crate::instance::VulkanInstance>,
        handle: sdl2::video::Window,
    ) -> Result<Self, String> {
        let surface_raw = handle
            .vulkan_create_surface(instance.handle.handle().as_raw() as usize)
            .map_err(|e| e.to_string())?;

        debug!("Vulkan Surface created from existing SDL2 Window");

        let surface = vk::SurfaceKHR::from_raw(surface_raw);

        Ok(VulkanWindow {
            instance,
            handle,
            surface,
        })
    }
}

impl Drop for VulkanWindow {
    fn drop(&mut self) {
        unsafe {
            use ash::khr::surface;
            let surface_loader =
                surface::Instance::new(&self.instance.entry, &self.instance.handle);
            surface_loader.destroy_surface(self.surface, None);
        }
        debug!("Vulkan Surface and Window destroyed");
    }
}
