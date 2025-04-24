#pragma once

#include "descriptor_set_layout.h"
#include "use_list.h"

typedef struct i3_vk_pipeline_layout_o
{
    i3_rbk_resource_i base;
    i3_rbk_pipeline_layout_i iface;
    i3_vk_device_o* device;
    uint32_t use_count;

    // use list for set layouts
    i3_vk_use_list_t use_list;

    VkPipelineLayout handle;
} i3_vk_pipeline_layout_o;

i3_rbk_pipeline_layout_i* i3_vk_device_create_pipeline_layout(i3_rbk_device_o* self,
                                                              const i3_rbk_pipeline_layout_desc_t* desc);