#pragma once

#include "device.h"

typedef struct i3_vk_buffer_o
{
    i3_rbk_resource_i base;
    i3_rbk_buffer_i iface;
    i3_vk_device_o* device;
    i3_rbk_buffer_desc_t desc;
    uint32_t use_count;
    VmaAllocation allocation;
    VkBuffer handle;
} i3_vk_buffer_o;

i3_rbk_buffer_i* i3_vk_device_create_buffer(i3_rbk_device_o* self, const i3_rbk_buffer_desc_t* desc);