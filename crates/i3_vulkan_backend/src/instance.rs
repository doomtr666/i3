use ash::vk;
#[cfg(debug_assertions)]
use std::ffi::CStr;
use std::ffi::CString;
use std::sync::Arc;
use tracing::info;
#[cfg(debug_assertions)]
use tracing::{error, warn};

#[cfg(debug_assertions)]
unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message_str = unsafe {
        let message = CStr::from_ptr((*p_callback_data).p_message);
        message.to_string_lossy()
    };

    let type_str = if message_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        "VALIDATION"
    } else if message_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        "PERFORMANCE"
    } else {
        "GENERAL"
    };

    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        error!(target: "vulkan_backend::validation", "[{}] {}", type_str, message_str);
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!(target: "vulkan_backend::validation", "[{}] {}", type_str, message_str);
    } else {
        info!(target: "vulkan_backend::validation", "[{}] {}", type_str, message_str);
    }

    vk::FALSE
}

pub struct VulkanInstance {
    pub entry: ash::Entry,
    pub handle: ash::Instance,

    #[cfg(debug_assertions)]
    debug_utils: ash::ext::debug_utils::Instance,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl VulkanInstance {
    pub fn new() -> Result<Arc<Self>, String> {
        let entry = ash::Entry::linked();

        let app_name = CString::new("i3fx").unwrap();
        let engine_name = CString::new("i3fx").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        let extensions = vec![
            ash::khr::surface::NAME.as_ptr(),
            #[cfg(target_os = "windows")]
            ash::khr::win32_surface::NAME.as_ptr(),
            #[cfg(debug_assertions)]
            ash::ext::debug_utils::NAME.as_ptr(),
        ];

        let layer_names: Vec<CString> = vec![
            #[cfg(debug_assertions)]
            CString::new("VK_LAYER_KHRONOS_validation").unwrap(),
        ];

        let layer_name_ptrs: Vec<*const i8> =
            layer_names.iter().map(|name| name.as_ptr()).collect();

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layer_name_ptrs);

        let handle = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| format!("Failed to create Vulkan instance: {}", e))?;

        info!("Vulkan 1.3 Instance created");

        #[cfg(debug_assertions)]
        let (debug_utils, debug_messenger) = {
            let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &handle);
            let debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_messenger =
                unsafe { debug_utils.create_debug_utils_messenger(&debug_create_info, None) }
                    .map_err(|e| format!("Failed to create debug messenger: {}", e))?;

            (debug_utils, debug_messenger)
        };

        Ok(Arc::new(VulkanInstance {
            entry,
            handle,
            #[cfg(debug_assertions)]
            debug_utils,
            #[cfg(debug_assertions)]
            debug_messenger,
        }))
    }
}

impl Drop for VulkanInstance {
    fn drop(&mut self) {
        unsafe {
            #[cfg(debug_assertions)]
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.handle.destroy_instance(None);
        }
        info!("Vulkan Instance destroyed");
    }
}
