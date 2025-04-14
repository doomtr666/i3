#pragma once

#include "descriptor_set_layout.h"

#define I3_RBK_PIPELINE_LAYOUT_MAX_DESCRIPTOR_SET_COUNT 16

typedef struct i3_vk_pipeline_layout_o
{
    i3_rbk_resource_i base;
    i3_rbk_pipeline_layout_i iface;
    i3_vk_device_o* device;
    uint32_t use_count;

    // keep ref to descriptor set layouts
    uint32_t set_layout_count;
    i3_rbk_descriptor_set_layout_i* set_layouts[I3_RBK_PIPELINE_LAYOUT_MAX_DESCRIPTOR_SET_COUNT];

    VkPipelineLayout handle;
} i3_vk_pipeline_layout_o;

i3_rbk_pipeline_layout_i* i3_vk_device_create_pipeline_layout(i3_rbk_device_o* self, const i3_rbk_pipeline_layout_desc_t* desc);