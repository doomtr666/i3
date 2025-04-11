#pragma once

#include "device.h"

typedef struct i3_vk_swapchain_o
{
    i3_rbk_resource_i base;
    i3_rbk_swapchain_i iface;
    i3_vk_device_o* device;
    i3_rbk_swapchain_desc_t desc;
    uint32_t use_count;
    i3_logger_i* log;
    VkSurfaceKHR surface;
    VkSwapchainCreateInfoKHR create_info;
    VkSwapchainKHR handle;
} i3_vk_swapchain_o;


i3_rbk_swapchain_i* i3_vk_device_create_swapchain(i3_rbk_device_o* self, i3_render_window_i* window, const i3_rbk_swapchain_desc_t* desc);