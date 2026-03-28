use ash::vk;

use i3_gfx::prelude::{Event, KeyCode, SwapchainConfig, WindowDesc, WindowHandle};

use crate::backend::VulkanBackend;
use crate::swapchain::VulkanSwapchain;

pub(crate) struct WindowContext {
    // Order matters for drop: swapchain must be dropped BEFORE raw (surface)
    pub(crate) swapchain: Option<VulkanSwapchain>,
    pub(crate) raw: crate::window::VulkanWindow,
    pub(crate) config: SwapchainConfig,
    // Semaphores for acquire (per frame in flight)
    pub(crate) acquire_semaphores: Vec<vk::Semaphore>,
    pub(crate) acquire_semaphore_ids: Vec<u64>,
    // Semaphores for present (per swapchain image)
    pub(crate) present_semaphores: Vec<vk::Semaphore>,
    pub(crate) present_semaphore_ids: Vec<u64>,
    // Track the current frame's acquire semaphore to pair it with the image
    pub(crate) current_acquire_sem_id: Option<u64>,
    pub(crate) current_image_index: Option<u32>,
}

pub fn create_window(
    backend: &mut VulkanBackend,
    desc: WindowDesc,
) -> Result<WindowHandle, String> {
    let window_handle = backend
        .video
        .window(&desc.title, desc.width, desc.height)
        .position_centered()
        .resizable()
        .vulkan()
        .build()
        .map_err(|e| e.to_string())?;

    let vulkan_window = crate::window::VulkanWindow::new(backend.instance.clone(), window_handle)?;

    let _id = backend.next_window_id;
    backend.next_window_id += 1;

    // Create Semaphores per frame for this window (typically 3 for triple buffering)
    let win_id = backend.next_window_id;
    backend.next_window_id += 1;

    let mut acquire_sems = Vec::new();
    let mut acquire_sem_ids = Vec::new();
    let _device_handle = backend.get_device().handle.clone();

    // Acquire semaphores are per-frame-in-flight (usually 3)
    for _ in 0..3 {
        let a_id = backend.create_semaphore();
        let a_sem = backend.semaphores.get(a_id).cloned().unwrap();

        acquire_sems.push(a_sem);
        acquire_sem_ids.push(a_id);
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
        present_semaphores: Vec::new(),
        present_semaphore_ids: Vec::new(),
        current_acquire_sem_id: None,
        current_image_index: None,
    };

    backend.windows.insert(win_id, context);
    Ok(WindowHandle(win_id))
}

pub fn destroy_window(backend: &mut VulkanBackend, window: WindowHandle) {
    if let Some(mut ctx) = backend.windows.remove(&window.0) {
        if let Some(sc) = ctx.swapchain.take() {
            let device = backend.get_device();
            unsafe {
                device.handle.device_wait_idle().ok();
            }
            backend.unregister_swapchain_images(&sc.images);
        }
    }
}

pub fn configure_window(
    backend: &mut VulkanBackend,
    window: WindowHandle,
    config: SwapchainConfig,
) -> Result<(), String> {
    let sc_opt = if let Some(ctx) = backend.windows.get_mut(&window.0) {
        ctx.config = config;
        // Invalidate swapchain so it recreates on next acquire
        ctx.swapchain.take()
    } else {
        return Err("Invalid window handle".to_string());
    };

    if let Some(sc) = sc_opt {
        let device = backend.get_device();
        unsafe {
            device.handle.device_wait_idle().ok();
        }
        backend.unregister_swapchain_images(&sc.images);
    }
    Ok(())
}

pub fn set_fullscreen(backend: &mut VulkanBackend, window: WindowHandle, fullscreen: bool) {
    if let Some(ctx) = backend.windows.get_mut(&window.0) {
        let mode = if fullscreen {
            sdl2::video::FullscreenType::Desktop
        } else {
            sdl2::video::FullscreenType::Off
        };
        let _ = ctx.raw.handle.set_fullscreen(mode);
    }
}

pub fn poll_events(backend: &mut VulkanBackend) -> Vec<Event> {
    let mut events = Vec::new();
    let mut resize_happened = false;
    if let Some(pump) = &mut backend.event_pump {
        for event in pump.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. } => events.push(Event::Quit),
                sdl2::event::Event::KeyDown {
                    keycode: Some(kd), ..
                } => {
                    if let Some(key) = sdl_to_keycode(kd) {
                        events.push(Event::KeyDown { key });
                    }
                }
                sdl2::event::Event::KeyUp {
                    keycode: Some(kd), ..
                } => {
                    if let Some(key) = sdl_to_keycode(kd) {
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
                    events.push(Event::MouseWheel { x: 0, y: y });
                }
                _ => {}
            }
        }
    }

    if resize_happened {
        let mut to_unregister = if backend.windows.len() > 0 {
            Vec::with_capacity(backend.windows.len())
        } else {
            Vec::new()
        };
        for ctx in backend.windows.values_mut() {
            if let Some(sc) = ctx.swapchain.take() {
                to_unregister.push(sc);
            }
        }
        if !to_unregister.is_empty() {
            let device = backend.get_device();
            unsafe {
                device.handle.device_wait_idle().ok();
            }
            for sc in to_unregister {
                backend.unregister_swapchain_images(&sc.images);
            }
        }
    }
    events
}

pub fn window_size(backend: &VulkanBackend, window: WindowHandle) -> Option<(u32, u32)> {
    backend
        .windows
        .get(&window.0)
        .map(|ctx| ctx.raw.handle.drawable_size())
}

pub fn sdl_to_keycode(sdl: sdl2::keyboard::Keycode) -> Option<KeyCode> {
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
        sdl2::keyboard::Keycode::F11 => Some(KeyCode::F11),
        sdl2::keyboard::Keycode::Return => Some(KeyCode::Return),
        sdl2::keyboard::Keycode::LShift => Some(KeyCode::LShift),
        _ => None,
    }
}
