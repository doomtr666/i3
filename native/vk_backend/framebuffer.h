#pragma once

#include "device.h"
#include "use_list.h"

#define I3_VK_FRAMEBUFFER_MAX_COLOR_ATTACHMENTS 16

typedef struct i3_vk_framebuffer_o
{
    i3_rbk_resource_i base;
    i3_rbk_framebuffer_i iface;
    i3_vk_device_o* device;
    uint32_t use_count;
    VkRenderPass render_pass;
    VkFramebuffer handle;

    // use list for attachments
    i3_vk_use_list_t use_list;

    uint32_t color_attachment_count;
    i3_rbk_image_view_i* color_attachments[I3_VK_FRAMEBUFFER_MAX_COLOR_ATTACHMENTS];
    i3_rbk_image_view_i* depth_attachment;

} i3_vk_framebuffer_o;

i3_rbk_framebuffer_i* i3_vk_device_create_framebuffer(i3_rbk_device_o* self, const i3_rbk_framebuffer_desc_t* desc);
