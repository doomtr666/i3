#pragma once

#include "device.h"

typedef struct i3_vk_descriptor_set_layout_o
{
    i3_rbk_resource_i base;
    i3_rbk_descriptor_set_layout_i iface;
    i3_vk_device_o* device;
    uint32_t use_count;
    VkDescriptorSetLayout handle;
} i3_vk_descriptor_set_layout_o;

i3_rbk_descriptor_set_layout_i* i3_vk_device_create_descriptor_set_layout(i3_rbk_device_o* self, const i3_rbk_descriptor_set_layout_desc_t* desc);