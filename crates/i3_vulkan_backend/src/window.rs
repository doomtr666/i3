use ash::vk;
use ash::vk::Handle;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::info;

pub struct VulkanWindow {
    pub instance: Arc<crate::instance::VulkanInstance>,
    pub sdl: sdl2::Sdl,
    pub video: sdl2::VideoSubsystem,
    pub handle: sdl2::video::Window,
    pub surface: vk::SurfaceKHR,
    pub event_pump: Mutex<Option<sdl2::EventPump>>,
}

impl VulkanWindow {
    pub fn new(
        instance: Arc<crate::instance::VulkanInstance>,
        title: &str,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let sdl = sdl2::init()?;
        let video = sdl.video()?;
        let event_pump = sdl.event_pump()?;

        let handle = video
            .window(title, width, height)
            .vulkan()
            .position_centered()
            .resizable()
            .build()
            .map_err(|e| e.to_string())?;

        let surface_raw = handle
            .vulkan_create_surface(instance.handle.handle().as_raw() as usize)
            .map_err(|e| e.to_string())?;

        info!("SDL2 Window and Vulkan Surface created");

        let surface = vk::SurfaceKHR::from_raw(surface_raw);

        Ok(VulkanWindow {
            instance,
            sdl,
            video,
            handle,
            surface,
            event_pump: Mutex::new(Some(event_pump)),
        })
    }

    pub fn take_event_pump(&self) -> Option<sdl2::EventPump> {
        self.event_pump.lock().unwrap().take()
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
        info!("Vulkan Surface and Window destroyed");
    }
}
