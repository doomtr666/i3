#pragma once

#include "native/core/array.h"

#include "common.h"
#include "instance_ext.h"

typedef struct i3_vk_backend_o
{
    i3_render_backend_i iface;

    i3_logger_i* log;
    uint32_t api_version;
    VkInstance instance;
    i3_vk_backend_instance_ext_t ext;
    VkDebugUtilsMessengerEXT debug_msg;
    i3_array_t physical_devices;

} i3_vk_backend_o;
