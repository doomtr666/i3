#pragma once

#include "device_ext.h"

void i3_vk_device_ext_load(VkDevice device, i3_vkbk_device_ext_t* ext)
{
    i3_logger_i* log = i3_vk_get_logger();

    bool* ext_supported;

#define I3_VKBK_DEV_EXT_NAME(ext_name)          \
    ext_supported = &ext->ext_name##_supported; \
    *ext_supported = true;
#define I3_VKBK_DEV_EXT_FN(func_name) \
    *ext_supported &= ((ext->func_name = (PFN_##func_name)vkGetDeviceProcAddr(device, #func_name)) != NULL);
    I3_VKBK_DEV_EXTS()
#undef I3_VKBK_DEV_EXT_NAME
#undef I3_VKBK_DEV_EXT_FN

#define I3_VKBK_DEV_EXT_NAME(ext_name) \
    i3_log_dbg(log, "Device ext " #ext_name " loaded : %s", ext->ext_name##_supported ? "TRUE" : "FALSE");
#define I3_VKBK_DEV_EXT_FN(func_name)
        I3_VKBK_DEV_EXTS()
#undef I3_VKBK_DEV_EXT_NAME
#undef I3_VKBK_DEV_EXT_FN
}

bool i3_vk_device_ext_supported(const char* name)
{
#define I3_VKBK_DEV_EXT_NAME(ext_name) \
    if (!strcmp(name, #ext_name))      \
        return true;
#define I3_VKBK_DEV_EXT_FN(func_name)
    I3_VKBK_DEV_EXTS()
#undef I3_VKBK_DEV_EXT_NAME
#undef I3_VKBK_DEV_EXT_FN
        return false;
}

