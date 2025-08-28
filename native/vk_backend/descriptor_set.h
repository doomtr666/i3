#pragma once

#include "descriptor_set_layout.h"

typedef struct i3_vk_descriptor_set_o
{
    i3_rbk_resource_i base;
    i3_rbk_descriptor_set_i iface;
    i3_vk_device_o* device;
    uint32_t use_count;
    i3_rbk_descriptor_set_layout_i* layout;
    VkDescriptorSet handle;
} i3_vk_descriptor_set_o;

i3_rbk_descriptor_set_i* i3_vk_device_create_descriptor_set(i3_rbk_device_o* self,
                                                            i3_rbk_descriptor_set_layout_i* layout);