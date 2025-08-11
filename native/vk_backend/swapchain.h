#pragma once

#include "device.h"

#define I3_VK_SWAPCHAIN_MAX_IMAGE_COUNT 16
#define I3_VK_SWAPCHAIN_MAX_SEM_COUNT I3_VK_SWAPCHAIN_MAX_IMAGE_COUNT

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

    // present info
    uint32_t image_count;
    VkImage images[I3_VK_SWAPCHAIN_MAX_IMAGE_COUNT];
    VkExtent2D extent;

    // semaphores
    uint32_t sem_count;
    VkSemaphore acquire_sems[I3_VK_SWAPCHAIN_MAX_SEM_COUNT];
    VkSemaphore present_sems[I3_VK_SWAPCHAIN_MAX_SEM_COUNT];

    // current semaphores
    uint32_t sem_index;
    VkSemaphore acquire_sem;
    VkSemaphore present_sem;

    bool out_of_date;

} i3_vk_swapchain_o;

i3_rbk_swapchain_i* i3_vk_device_create_swapchain(i3_rbk_device_o* self,
                                                  i3_render_window_i* window,
                                                  const i3_rbk_swapchain_desc_t* desc);

// for presentation
uint32_t i3_vk_swapchain_acquire_image(i3_vk_swapchain_o* swapchain);
void i3_vk_swapchain_present(i3_vk_swapchain_o* swapchain, uint32_t image_index);
VkSemaphore i3_vk_swapchain_get_acquire_semaphore(i3_vk_swapchain_o* swapchain);
VkSemaphore i3_vk_swapchain_get_present_semaphore(i3_vk_swapchain_o* swapchain);
