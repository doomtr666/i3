#pragma once

#include "common.h"

typedef struct i3_vk_device_desc
{
    i3_rbk_device_desc_t base;
    VkPhysicalDevice physical_device;
    VkPhysicalDeviceProperties properties;
} i3_vk_device_desc;
