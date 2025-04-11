#include "common.h"

// vk check
#define VK_RESULT_LIST()                                             \
    VK_RESULT(VK_SUCCESS)                                            \
    VK_RESULT(VK_NOT_READY)                                          \
    VK_RESULT(VK_TIMEOUT)                                            \
    VK_RESULT(VK_EVENT_SET)                                          \
    VK_RESULT(VK_EVENT_RESET)                                        \
    VK_RESULT(VK_INCOMPLETE)                                         \
    VK_RESULT(VK_ERROR_OUT_OF_HOST_MEMORY)                           \
    VK_RESULT(VK_ERROR_OUT_OF_DEVICE_MEMORY)                         \
    VK_RESULT(VK_ERROR_INITIALIZATION_FAILED)                        \
    VK_RESULT(VK_ERROR_DEVICE_LOST)                                  \
    VK_RESULT(VK_ERROR_MEMORY_MAP_FAILED)                            \
    VK_RESULT(VK_ERROR_LAYER_NOT_PRESENT)                            \
    VK_RESULT(VK_ERROR_EXTENSION_NOT_PRESENT)                        \
    VK_RESULT(VK_ERROR_FEATURE_NOT_PRESENT)                          \
    VK_RESULT(VK_ERROR_INCOMPATIBLE_DRIVER)                          \
    VK_RESULT(VK_ERROR_TOO_MANY_OBJECTS)                             \
    VK_RESULT(VK_ERROR_FORMAT_NOT_SUPPORTED)                         \
    VK_RESULT(VK_ERROR_FRAGMENTED_POOL)                              \
    VK_RESULT(VK_ERROR_UNKNOWN)                                      \
    VK_RESULT(VK_ERROR_OUT_OF_POOL_MEMORY)                           \
    VK_RESULT(VK_ERROR_INVALID_EXTERNAL_HANDLE)                      \
    VK_RESULT(VK_ERROR_FRAGMENTATION)                                \
    VK_RESULT(VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS)               \
    VK_RESULT(VK_PIPELINE_COMPILE_REQUIRED)                          \
    VK_RESULT(VK_ERROR_SURFACE_LOST_KHR)                             \
    VK_RESULT(VK_ERROR_NATIVE_WINDOW_IN_USE_KHR)                     \
    VK_RESULT(VK_SUBOPTIMAL_KHR)                                     \
    VK_RESULT(VK_ERROR_OUT_OF_DATE_KHR)                              \
    VK_RESULT(VK_ERROR_INCOMPATIBLE_DISPLAY_KHR)                     \
    VK_RESULT(VK_ERROR_VALIDATION_FAILED_EXT)                        \
    VK_RESULT(VK_ERROR_INVALID_SHADER_NV)                            \
    VK_RESULT(VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT) \
    VK_RESULT(VK_ERROR_NOT_PERMITTED_KHR)                            \
    VK_RESULT(VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT)          \
    VK_RESULT(VK_THREAD_IDLE_KHR)                                    \
    VK_RESULT(VK_THREAD_DONE_KHR)                                    \
    VK_RESULT(VK_OPERATION_DEFERRED_KHR)                             \
    VK_RESULT(VK_OPERATION_NOT_DEFERRED_KHR)

static const char* i3_vk_result_to_string(VkResult result)
{
#define VK_RESULT(result) \
    case result:          \
        return #result;
    switch (result)
    {
        VK_RESULT_LIST()
    default:
        return "Unknown";
    }
#undef VK_RESULT
}
#undef VK_RESULT_LIST

// logger
i3_logger_i* i3_vk_get_logger()
{
    static i3_logger_i* logger;
    if (logger == NULL)
        logger = i3_get_logger(I3_VK_BACKEND_LOGGER_NAME);

    return logger;
}

// vk check
void i3_vk_check__(VkResult result, const char* file, int line)
{
    if (result != VK_SUCCESS)
        i3_vk_log_fatal("Vulkan fatal error: %s (0x%x) at %s:%d", i3_vk_result_to_string(result), result, file, line);
}

