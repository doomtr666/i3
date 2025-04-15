#pragma once

#include "device.h"

typedef struct i3_vk_cmd_buffer_o
{
    i3_rbk_resource_i base;
    i3_rbk_cmd_buffer_i iface;
    i3_vk_device_o* device;
    uint32_t use_count;
} i3_vk_cmd_buffer_o;

i3_rbk_cmd_buffer_i* i3_vk_device_create_cmd_buffer(i3_rbk_device_o* self);