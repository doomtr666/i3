#pragma once

#include "device.h"

typedef struct i3_vk_sampler_o
{
    i3_rbk_resource_i base;
    i3_rbk_sampler_i iface;
    i3_vk_device_o* device;
    i3_rbk_sampler_desc_t desc;
    uint32_t use_count;
    VkSampler handle;
} i3_vk_sampler_o;

i3_rbk_sampler_i* i3_vk_device_create_sampler(i3_rbk_device_o* self, const i3_rbk_sampler_desc_t* desc);