#include "swapchain.h"
#include "convert.h"

// resource interface
static void i3_vk_swapchain_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_swapchain_o* swapchain = (i3_vk_swapchain_o*)self;

    swapchain->use_count++;
}

static void i3_vk_swapchain_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_swapchain_o* swapchain = (i3_vk_swapchain_o*)self;

    swapchain->use_count--;

    if (swapchain->use_count == 0)
    {
        // destroy semaphores
        vkDestroySemaphore(swapchain->device->handle, swapchain->acquire_sem, NULL);
        vkDestroySemaphore(swapchain->device->handle, swapchain->present_sem, NULL);
        vkDestroySwapchainKHR(swapchain->device->handle, swapchain->handle, NULL);
        i3_free(swapchain);
    }
}

static int32_t i3_vk_swapchain_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_swapchain_o* swapchain = (i3_vk_swapchain_o*)self;

    return swapchain->use_count;
}

static void i3_vk_swapchain_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);
    i3_vk_swapchain_o* swapchain = (i3_vk_swapchain_o*)self;

    if (swapchain->device->backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsObjectNameInfoEXT name_info = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_SWAPCHAIN_KHR,
                                                   .objectHandle = (uintptr_t)swapchain->handle,
                                                   .pObjectName = name};
        swapchain->device->backend->ext.vkSetDebugUtilsObjectNameEXT(swapchain->device->handle, &name_info);
    }
}

// swapchain interface

static const i3_rbk_swapchain_desc_t* i3_vk_swapchain_get_desc(i3_rbk_swapchain_o* self)
{
    assert(self != NULL);
    i3_vk_swapchain_o* swapchain = (i3_vk_swapchain_o*)self;

    return &swapchain->desc;
}

static i3_rbk_resource_i* i3_vk_swapchain_get_resource_i(i3_rbk_swapchain_o* self)
{
    assert(self != NULL);
    i3_vk_swapchain_o* swapchain = (i3_vk_swapchain_o*)self;

    return &swapchain->base;
}

static void i3_vk_swapchain_destroy(i3_rbk_swapchain_o* self)
{
    assert(self != NULL);
    i3_vk_swapchain_o* swapchain = (i3_vk_swapchain_o*)self;

    swapchain->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_swapchain_o i3_vk_swapchain_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_swapchain_add_ref,
        .release = i3_vk_swapchain_release,
        .get_use_count = i3_vk_swapchain_get_use_count,
        .set_debug_name = i3_vk_swapchain_set_debug_name,
    },
    .iface =
    {
        .get_desc = i3_vk_swapchain_get_desc,
        .get_resource_i = i3_vk_swapchain_get_resource_i,
        .destroy = i3_vk_swapchain_destroy,
    }
};

bool i3_vk_recreate_swapchain(i3_vk_swapchain_o* swapchain)
{
    assert(swapchain != NULL);

    // get physical device capabilities
    VkSurfaceCapabilitiesKHR surface_caps;
    i3_vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(swapchain->device->desc.physical_device, swapchain->surface,
                                                          &surface_caps));

    // image extents
    swapchain->create_info.imageExtent = surface_caps.currentExtent;

    // old swapchain
    swapchain->create_info.oldSwapchain = swapchain->handle;

    // create swapchain
    VkSwapchainKHR old_swapchain = swapchain->handle;
    i3_vk_check(vkCreateSwapchainKHR(swapchain->device->handle, &swapchain->create_info, NULL, &swapchain->handle));
    if (old_swapchain != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(swapchain->device->handle, old_swapchain, NULL);

    // get swapchain images
    i3_vk_check(vkGetSwapchainImagesKHR(swapchain->device->handle, swapchain->handle, &swapchain->image_count, NULL));
    assert(swapchain->image_count > 0 && swapchain->image_count <= I3_VK_SWAPCHAIN_MAX_IMAGE_COUNT);
    i3_vk_check(vkGetSwapchainImagesKHR(swapchain->device->handle, swapchain->handle, &swapchain->image_count,
                                        swapchain->images));

    // reset out of date flag
    swapchain->out_of_date = false;

    return true;
}

static bool i3_vk_prensent_mode_supported(VkPresentModeKHR* present_modes,
                                          uint32_t present_mode_count,
                                          VkPresentModeKHR present_mode)
{
    for (uint32_t i = 0; i < present_mode_count; i++)
    {
        if (present_modes[i] == present_mode)
            return true;
    }
    return false;
}

i3_rbk_swapchain_i* i3_vk_device_create_swapchain(i3_rbk_device_o* self,
                                                  i3_render_window_i* window,
                                                  const i3_rbk_swapchain_desc_t* desc)
{
    assert(self != NULL);
    assert(window != NULL);
    assert(desc != NULL);

    i3_vk_swapchain_o* swapchain = i3_alloc(sizeof(i3_vk_swapchain_o));
    assert(swapchain != NULL);

    *swapchain = i3_vk_swapchain_iface_;
    swapchain->base.self = (i3_rbk_resource_o*)swapchain;
    swapchain->iface.self = (i3_rbk_swapchain_o*)swapchain;
    swapchain->device = (i3_vk_device_o*)self;
    swapchain->desc = *desc;
    swapchain->use_count = 1;
    swapchain->log = i3_vk_get_logger();

    // get surface
    swapchain->surface = window->get_vk_surface(window->self);
    if (swapchain->surface == NULL)
    {
        i3_log_err(swapchain->log, "Failed to get window surface");
        return NULL;
    }

    // swapchain create info
    swapchain->create_info = (VkSwapchainCreateInfoKHR){
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = swapchain->surface,
        .imageArrayLayers = 1,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .clipped = VK_TRUE,
    };

    // get physical device capabilities
    VkSurfaceCapabilitiesKHR surface_caps;
    i3_vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(swapchain->device->desc.physical_device, swapchain->surface,
                                                          &surface_caps));

    // min image count
    swapchain->create_info.minImageCount
        = i3_clamp(swapchain->desc.requested_image_count, surface_caps.minImageCount, surface_caps.maxImageCount);

    // image extents
    swapchain->create_info.imageExtent = surface_caps.currentExtent;

    // image usage
    if (surface_caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        swapchain->create_info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    else
        i3_vk_log_fatal("No supported image usage");

    // pre transform
    swapchain->create_info.preTransform = surface_caps.currentTransform;

    // composite alpha
    if (surface_caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
        swapchain->create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    else if (surface_caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
        swapchain->create_info.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    else
        i3_vk_log_fatal("No supported composite alpha");

    // format and color space
    VkSurfaceFormatKHR* surface_formats = NULL;
    uint32_t format_count = 0;
    i3_vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(swapchain->device->desc.physical_device, swapchain->surface,
                                                     &format_count, NULL));
    if (format_count == 0)
        i3_vk_log_fatal("No supported surface formats");
    surface_formats = i3_alloc(format_count * sizeof(VkSurfaceFormatKHR));
    assert(surface_formats != NULL);
    i3_vk_check(vkGetPhysicalDeviceSurfaceFormatsKHR(swapchain->device->desc.physical_device, swapchain->surface,
                                                     &format_count, surface_formats));

    swapchain->create_info.imageFormat = surface_formats[0].format;
    swapchain->create_info.imageColorSpace = surface_formats[0].colorSpace;

    // TODO: better format selection
    for (uint32_t i = 0; i < format_count; i++)
    {
        bool srgb = i3_vk_is_srgb_format(surface_formats[i].format);
        if (srgb == swapchain->desc.srgb)
        {
            swapchain->create_info.imageFormat = surface_formats[i].format;
            swapchain->create_info.imageColorSpace = surface_formats[i].colorSpace;
            break;
        }
    }

    i3_free(surface_formats);

    // present mode
    VkPresentModeKHR* present_modes = NULL;
    uint32_t present_mode_count = 0;
    i3_vk_check(vkGetPhysicalDeviceSurfacePresentModesKHR(swapchain->device->desc.physical_device, swapchain->surface,
                                                          &present_mode_count, NULL));
    if (present_mode_count == 0)
        i3_vk_log_fatal("No supported present modes");
    present_modes = i3_alloc(present_mode_count * sizeof(VkPresentModeKHR));
    assert(present_modes != NULL);
    i3_vk_check(vkGetPhysicalDeviceSurfacePresentModesKHR(swapchain->device->desc.physical_device, swapchain->surface,
                                                          &present_mode_count, present_modes));

    // always supported
    swapchain->create_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;

    if (swapchain->desc.vsync)
    {
        // prefer mailbox if available
        if (i3_vk_prensent_mode_supported(present_modes, present_mode_count, VK_PRESENT_MODE_MAILBOX_KHR))
            swapchain->create_info.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    }
    else
    {
        // prefer immediate if available
        if (i3_vk_prensent_mode_supported(present_modes, present_mode_count, VK_PRESENT_MODE_IMMEDIATE_KHR))
            swapchain->create_info.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        // or fifo_relaxed
        else if (i3_vk_prensent_mode_supported(present_modes, present_mode_count, VK_PRESENT_MODE_FIFO_RELAXED_KHR))
            swapchain->create_info.presentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
    }

    i3_free(present_modes);

    // create swapchain
    if (!i3_vk_recreate_swapchain(swapchain))
        i3_vk_log_fatal("Failed to create swapchain");

    // create semaphores
    VkSemaphoreCreateInfo sem_ci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    i3_vk_check(vkCreateSemaphore(swapchain->device->handle, &sem_ci, NULL, &swapchain->acquire_sem));
    i3_vk_check(vkCreateSemaphore(swapchain->device->handle, &sem_ci, NULL, &swapchain->present_sem));

    return &swapchain->iface;
}

uint32_t i3_vk_swapchain_acquire_image(i3_vk_swapchain_o* swapchain)
{
    assert(swapchain != NULL);

    if (swapchain->out_of_date)
        i3_vk_recreate_swapchain(swapchain);

    // acquire image
    uint32_t image_index = 0;

    VkResult result = vkAcquireNextImageKHR(swapchain->device->handle, swapchain->handle, UINT64_MAX,
                                            swapchain->acquire_sem, VK_NULL_HANDLE, &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        swapchain->out_of_date = true;
    else if (result != VK_SUCCESS)
    {
        i3_vk_log_fatal("Failed to acquire swapchain image: %d", result);
        return UINT32_MAX;
    }

    return image_index;
}

void i3_vk_swapchain_present(i3_vk_swapchain_o* swapchain, uint32_t image_index)
{
    assert(swapchain != NULL);
    assert(image_index < swapchain->image_count);

    VkPresentInfoKHR present_info = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                     .waitSemaphoreCount = 1,
                                     .pWaitSemaphores = &swapchain->present_sem,
                                     .swapchainCount = 1,
                                     .pSwapchains = &swapchain->handle,
                                     .pImageIndices = &image_index};

    VkResult result = vkQueuePresentKHR(swapchain->device->graphics_queue, &present_info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        swapchain->out_of_date = true;
    else if (result != VK_SUCCESS)
        i3_vk_log_fatal("Failed to present swapchain image: %d", result);
}