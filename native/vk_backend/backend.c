#include "native/core/array.h"

#include "backend.h"

#include "instance_ext.h"
#include "device.h"

static const i3_rbk_device_desc_t* i3_vk_backend_get_device_desc(i3_render_backend_o* self, uint32_t index)
{
    assert(self != NULL);
    i3_vk_backend_o* backend = (i3_vk_backend_o*)self;

    if (index >= i3_array_count(&backend->physical_devices))
        return NULL;

    return i3_array_at(&backend->physical_devices, index);
}

static uint32_t i3_vk_backend_get_device_count(i3_render_backend_o* self)
{
    assert(self != NULL);
    i3_vk_backend_o* backend = (i3_vk_backend_o*)self;

    return i3_array_count(&backend->physical_devices);
}

static i3_render_window_i* i3_vk_backend_create_render_window(i3_render_backend_o* self, const char* title, uint32_t width, uint32_t height)
{
    assert(self != NULL);
    i3_vk_backend_o* backend = (i3_vk_backend_o*)self;

    return i3_render_window_create_vulkan(backend->instance, title, width, height);
}

static i3_rbk_device_i* i3_vk_backend_create_device(i3_render_backend_o* self, uint32_t desc_index)
{
    assert(self != NULL);

    i3_vk_backend_o* backend = (i3_vk_backend_o*)self;

    if (desc_index >= i3_array_count(&backend->physical_devices))
    {
        i3_log_err(backend->log, "invalid device index");
        return NULL;
    }

    i3_vk_device_desc* desc = i3_array_at(&backend->physical_devices, desc_index);

    return i3_vk_device_create(backend, desc);
}

// destroy
static void i3_vk_backend_destroy(i3_render_backend_o* self)
{
    assert(self != NULL);

    i3_vk_backend_o* backend = (i3_vk_backend_o*)self;

    // destroy physical devices array
    i3_array_free(&backend->physical_devices);

    // destroy vulkan objects
    if (backend->debug_msg)
        backend->ext.vkDestroyDebugUtilsMessengerEXT(backend->instance, backend->debug_msg, NULL);

    vkDestroyInstance(backend->instance, NULL);
    
    i3_log_inf(backend->log, "Vulkan backend destroyed");
    i3_free(self);
}

static i3_render_backend_i vk_render_backend_iface_ = {
    .self = NULL,
    .get_device_desc = i3_vk_backend_get_device_desc,
    .get_device_count = i3_vk_backend_get_device_count,
    .create_render_window = i3_vk_backend_create_render_window,
    .create_device = i3_vk_backend_create_device,
    .destroy = i3_vk_backend_destroy
};

static VkBool32 i3_vk_backend_instance_debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    assert(pUserData != NULL);
    i3_vk_backend_o* backend = (i3_vk_backend_o*)pUserData;

    switch (messageSeverity)
    {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        i3_log_dbg(backend->log, "%s", pCallbackData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        i3_log_inf(backend->log, "%s", pCallbackData->pMessage);
        break;

    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        i3_log_wrn(backend->log, "%s", pCallbackData->pMessage);
        break;

    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        i3_log_err(backend->log, "%s", pCallbackData->pMessage);
        break;

    default:
        break;
    }

    return VK_FALSE;
}

i3_render_backend_i* i3_vk_backend_create(bool enable_validation)
{
    i3_vk_backend_o* backend = i3_zalloc(sizeof(i3_vk_backend_o));
    assert(backend != NULL);

    backend->iface = vk_render_backend_iface_;
    backend->iface.self = (i3_render_backend_o*)backend;
    backend->log = i3_vk_get_logger();

    // check supported version
    backend->api_version = VK_API_VERSION_1_0;
    if (vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceVersion") != NULL)
        i3_vk_check(vkEnumerateInstanceVersion(&backend->api_version));

    i3_log_inf(backend->log, "Vulkan API version: %d.%d.%d", VK_VERSION_MAJOR(backend->api_version), VK_VERSION_MINOR(backend->api_version), VK_VERSION_PATCH(backend->api_version));

    // enumerate layers
    uint32_t inst_layer_count = 0;
    i3_vk_check(vkEnumerateInstanceLayerProperties(&inst_layer_count, NULL));
    VkLayerProperties* inst_layers = i3_alloc(inst_layer_count * sizeof(VkLayerProperties));
    assert(inst_layers != NULL);
    i3_vk_check(vkEnumerateInstanceLayerProperties(&inst_layer_count, inst_layers));

    // enumarate extensions
    uint32_t inst_ext_count = 0;
    i3_vk_check(vkEnumerateInstanceExtensionProperties(NULL, &inst_ext_count, NULL));
    VkExtensionProperties* inst_exts = i3_alloc(inst_ext_count * sizeof(VkExtensionProperties));
    assert(inst_exts != NULL);
    i3_vk_check(vkEnumerateInstanceExtensionProperties(NULL, &inst_ext_count, inst_exts));

    // enabled layers
    i3_array_t enabled_layers;
    i3_array_init(&enabled_layers, sizeof(char**));

    // enabled extensions
    i3_array_t enabled_extensions;
    i3_array_init(&enabled_extensions, sizeof(char*));

    // required instance WSI extensions
    uint32_t required_wsi_extension_count = 0;
    const char** required_wsi_extensions = i3_render_window_get_required_vk_instance_extensions(&required_wsi_extension_count);

    // enable validation layer if needed
    if (enable_validation)
    {
        for (uint32_t i = 0; i < inst_layer_count; i++)
            if (!strcmp(VK_VALIDATION_LAYER_NAME, inst_layers[i].layerName))
            {
                char* layer_name = inst_layers[i].layerName;
                i3_array_push(&enabled_layers, &layer_name);
                i3_log_wrn(backend->log, "Enabled layer: %s", VK_VALIDATION_LAYER_NAME);
                break;
            }
    }

    for (uint32_t i = 0; i < inst_ext_count; i++)
    {
        char* ext_name = inst_exts[i].extensionName;

        // enable WSI extensions
        for (uint32_t j = 0; j < required_wsi_extension_count; j++)
        {
            if (!strcmp(ext_name, required_wsi_extensions[j]))
            {
                i3_array_push(&enabled_extensions, &ext_name);
                i3_log_dbg(backend->log, "Enabled instance extension: %s", ext_name);
                break;
            }
        }

        // enable supported extensions
        if (i3_vk_backend_is_instance_ext_supported(ext_name))
        {
            i3_array_push(&enabled_extensions, &ext_name);
            i3_log_dbg(backend->log, "Enabled instance extension: %s", ext_name);
        }
    }

    // create instance
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = NULL,
        .apiVersion = backend->api_version
    };

    VkInstanceCreateInfo instance_ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = i3_array_count(&enabled_layers),
        .ppEnabledLayerNames = i3_array_data(&enabled_layers),
        .enabledExtensionCount = i3_array_count(&enabled_extensions),
        .ppEnabledExtensionNames = i3_array_data(&enabled_extensions)
    };

    i3_vk_check(vkCreateInstance(&instance_ci, NULL, &backend->instance));
    i3_log_dbg(backend->log, "Vulkan instance created");

    // cleanup
    i3_free(inst_layers);
    i3_free(inst_exts);
    i3_array_free(&enabled_layers);
    i3_array_free(&enabled_extensions);

    // load ext
    i3_vk_backend_instance_ext_load(backend->instance, &backend->ext);

    // create debug util messenger
    if (enable_validation && backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsMessengerCreateInfoEXT debug_msg_ci =
        {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity =
            //VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            //VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = i3_vk_backend_instance_debug_callback,
        .pUserData = backend,
        };
        i3_vk_check(backend->ext.vkCreateDebugUtilsMessengerEXT(backend->instance, &debug_msg_ci, NULL, &backend->debug_msg));
    }

    // iterate over physical devices    
    uint32_t device_count = 0;
    i3_vk_check(vkEnumeratePhysicalDevices(backend->instance, &device_count, NULL));
    VkPhysicalDevice* devices = i3_alloc(device_count * sizeof(VkPhysicalDevice));
    assert(devices != NULL);
    i3_vk_check(vkEnumeratePhysicalDevices(backend->instance, &device_count, devices));

    // create physical device array
    i3_array_init_capacity(&backend->physical_devices, sizeof(i3_vk_device_desc), device_count);
    for (uint32_t i = 0; i < device_count; ++i)
    {
        i3_vk_device_desc* desc = i3_array_addn(&backend->physical_devices, 1);
        desc->physical_device = devices[i];
        vkGetPhysicalDeviceProperties(devices[i], &desc->properties);

        // set device name
        desc->base.name = desc->properties.deviceName;

        i3_log_dbg(backend->log, "Supported Device: %s, API version %d.%d.%d", desc->base.name,
            VK_VERSION_MAJOR(desc->properties.apiVersion),
            VK_VERSION_MINOR(desc->properties.apiVersion),
            VK_VERSION_PATCH(desc->properties.apiVersion));
    }

    i3_free(devices);

    i3_log_inf(i3_vk_get_logger(), "Vulkan backend created");

    return &backend->iface;
}
