#pragma once

#include "common.h"

#define I3_VKBK_DEV_EXTS()                                                  \
    /* swapchain */                                                         \
    I3_VKBK_DEV_EXT_NAME(VK_KHR_swapchain)

typedef struct
{
#define I3_VKBK_DEV_EXT_NAME(ext_name) bool ext_name##_supported;
#define I3_VKBK_DEV_EXT_FN(func_name) PFN_##func_name func_name;
    I3_VKBK_DEV_EXTS()
#undef I3_VKBK_DEV_EXT_NAME
#undef I3_VKBK_DEV_EXT_FN

} i3_vkbk_device_ext_t;

void i3_vk_device_ext_load(VkDevice device, i3_vkbk_device_ext_t* ext);
bool i3_vk_device_ext_supported(const char* name);