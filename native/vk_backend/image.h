#pragma once

#include "barrier.h"
#include "device.h"

typedef struct i3_vk_image_o
{
    i3_rbk_resource_i base;
    i3_rbk_image_i iface;
    i3_vk_device_o* device;
    i3_rbk_image_desc_t desc;
    uint32_t use_count;
    VmaAllocation allocation;
    VkImage handle;

    i3_vk_image_barrier_info_t barrier_info;

} i3_vk_image_o;

i3_rbk_image_i* i3_vk_device_create_image(i3_rbk_device_o* self, const i3_rbk_image_desc_t* desc);