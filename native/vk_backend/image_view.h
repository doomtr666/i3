#pragma once

#include "image.h"

typedef struct i3_vk_image_view_o
{
    i3_rbk_resource_i base;
    i3_rbk_image_view_i iface;
    i3_vk_device_o* device;
    i3_rbk_image_view_desc_t info;
    i3_vk_image_o* image;
    uint32_t use_count;
    VkImageView handle;
} i3_vk_image_view_o;

i3_rbk_image_view_i* i3_vk_device_create_image_view(i3_rbk_device_o* self, i3_rbk_image_i* image, const i3_rbk_image_view_desc_t* info);
