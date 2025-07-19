#pragma once

#include "device.h"

typedef struct i3_vk_pipeline_o
{
    i3_rbk_resource_i base;
    i3_rbk_pipeline_i iface;
    i3_vk_device_o* device;
    uint32_t use_count;
    i3_rbk_pipeline_layout_i* layout;  // keep ref to pipeline layout
    VkPipelineBindPoint bind_point;
    VkPipeline handle;
    VkRenderPass render_pass;  // for graphics pipelines
} i3_vk_pipeline_o;

i3_rbk_pipeline_i* i3_vk_device_create_graphics_pipeline(i3_rbk_device_o* self,
                                                         const i3_rbk_graphics_pipeline_desc_t* desc);
i3_rbk_pipeline_i* i3_vk_device_create_compute_pipeline(i3_rbk_device_o* self,
                                                        const i3_rbk_compute_pipeline_desc_t* desc);
