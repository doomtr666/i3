use ash::vk;
use i3_gfx::graph::backend::{BackendBuffer, BackendImage};

use crate::backend::VulkanBackend;

/// Set a debug name for an image resource.
#[cfg(debug_assertions)]
pub fn set_image_name(backend: &mut VulkanBackend, image: BackendImage, name: &str) {
    if let Some(img) = backend.images.get(image.0) {
        let c_name = std::ffi::CString::new(name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(img.image)
            .object_name(&c_name);
        unsafe {
            backend
                .get_device()
                .debug_utils
                .set_debug_utils_object_name(&name_info)
                .ok();
        }
    }
}

/// Set a debug name for a buffer resource.
#[cfg(debug_assertions)]
pub fn set_buffer_name(backend: &mut VulkanBackend, buffer: BackendBuffer, name: &str) {
    if let Some(buf) = backend.buffers.get(buffer.0) {
        let c_name = std::ffi::CString::new(name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_handle(buf.buffer)
            .object_name(&c_name);
        unsafe {
            backend
                .get_device()
                .debug_utils
                .set_debug_utils_object_name(&name_info)
                .ok();
        }
    }
}
