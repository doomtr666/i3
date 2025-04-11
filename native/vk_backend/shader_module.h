#pragma once

#include "device.h"

typedef struct i3_vk_shader_module_o
{
    i3_rbk_resource_i base;
    i3_rbk_shader_module_i iface;
    i3_vk_device_o* device;
    i3_rbk_shader_module_desc_t desc;
    uint32_t use_count;
    VkShaderModule handle;
} i3_vk_shader_module_o;

i3_rbk_shader_module_i* i3_vk_device_create_shader_module(i3_rbk_device_o* self, const i3_rbk_shader_module_desc_t* desc);