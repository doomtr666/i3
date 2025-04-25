#include "native/core/arena.h"

#include "convert.h"
#include "framebuffer.h"
#include "image_view.h"

// resource interface

static void i3_vk_framebuffer_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_framebuffer_o* framebuffer = (i3_vk_framebuffer_o*)self;

    framebuffer->use_count++;
}

static void i3_vk_framebuffer_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_framebuffer_o* framebuffer = (i3_vk_framebuffer_o*)self;

    if (--framebuffer->use_count == 0)
    {
        vkDestroyRenderPass(framebuffer->device->handle, framebuffer->render_pass, NULL);
        vkDestroyFramebuffer(framebuffer->device->handle, framebuffer->handle, NULL);

        // destroy use list
        i3_vk_use_list_destroy(&framebuffer->use_list);

        i3_memory_pool_free(&framebuffer->device->framebuffer_pool, framebuffer);
    }
}

static int32_t i3_vk_framebuffer_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_framebuffer_o* framebuffer = (i3_vk_framebuffer_o*)self;

    return framebuffer->use_count;
}

static void i3_vk_framebuffer_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);
    i3_vk_framebuffer_o* framebuffer = (i3_vk_framebuffer_o*)self;

    if (framebuffer->device->backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsObjectNameInfoEXT name_info = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_FRAMEBUFFER,
                                                   .objectHandle = (uintptr_t)framebuffer->handle,
                                                   .pObjectName = name};
        framebuffer->device->backend->ext.vkSetDebugUtilsObjectNameEXT(framebuffer->device->handle, &name_info);
    }
}

//  framebuffer interface
static i3_rbk_resource_i* i3_vk_framebuffer_get_resource(i3_rbk_framebuffer_o* self)
{
    assert(self != NULL);
    i3_vk_framebuffer_o* framebuffer = (i3_vk_framebuffer_o*)self;

    return &framebuffer->base;
}

static void i3_vk_framebuffer_destroy(i3_rbk_framebuffer_o* self)
{
    assert(self != NULL);
    i3_vk_framebuffer_o* framebuffer = (i3_vk_framebuffer_o*)self;

    framebuffer->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_framebuffer_o i3_vk_framebuffer_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_framebuffer_add_ref,
        .release = i3_vk_framebuffer_release,
        .get_use_count = i3_vk_framebuffer_get_use_count,
        .set_debug_name = i3_vk_framebuffer_set_debug_name,
    },
    .iface =
    {
        .get_resource = i3_vk_framebuffer_get_resource,
        .destroy = i3_vk_framebuffer_destroy,
    },
};

// create framebuffer
i3_rbk_framebuffer_i* i3_vk_device_create_framebuffer(i3_rbk_device_o* self, const i3_rbk_framebuffer_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;
    i3_vk_framebuffer_o* framebuffer = i3_memory_pool_alloc(&device->framebuffer_pool);

    *framebuffer = i3_vk_framebuffer_iface_;
    framebuffer->base.self = (i3_rbk_resource_o*)framebuffer;
    framebuffer->iface.self = (i3_rbk_framebuffer_o*)framebuffer;
    framebuffer->device = device;
    framebuffer->use_count = 1;

    // intialize use list
    i3_vk_use_list_init(&framebuffer->use_list, device);

    i3_arena_t arena;
    i3_arena_init(&arena, I3_KB);

    uint32_t attachment_count = desc->color_attachment_count + (desc->depth_attachment ? 1 : 0);

    VkAttachmentReference* attachment_ref = i3_arena_alloc(&arena, sizeof(VkAttachmentReference) * attachment_count);
    VkAttachmentDescription* attachments = i3_arena_alloc(&arena, sizeof(VkAttachmentDescription) * attachment_count);
    VkImageView* image_views = i3_arena_alloc(&arena, sizeof(VkImageView) * attachment_count);

    uint32_t i;
    for (i = 0; i < desc->color_attachment_count; i++)
    {
        // retain image view
        i3_rbk_image_view_i* image_view = desc->color_attachments[i].image_view;
        i3_vk_use_list_add(&framebuffer->use_list, image_view);

        const i3_rbk_image_view_desc_t* image_view_desc = image_view->get_desc(image_view->self);
        i3_rbk_image_i* image = image_view->get_image(image_view->self);
        const i3_rbk_image_desc_t* image_desc = image->get_desc(image->self);

        attachment_ref[i] = (VkAttachmentReference){
            .attachment = i,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        attachments[i] = (VkAttachmentDescription){
            .format = i3_vk_convert_format(image_view_desc->format),
            .samples = image_desc->samples,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        image_views[i] = ((i3_vk_image_view_o*)desc->color_attachments[i].image_view->self)->handle;
    }

    if (desc->depth_attachment)
    {
        // retain image view
        i3_rbk_image_view_i* image_view = desc->depth_attachment->image_view;
        i3_vk_use_list_add(&framebuffer->use_list, image_view);

        const i3_rbk_image_view_desc_t* image_view_desc = image_view->get_desc(image_view->self);
        i3_rbk_image_i* image = image_view->get_image(image_view->self);
        const i3_rbk_image_desc_t* image_desc = image->get_desc(image->self);

        attachment_ref[i] = (VkAttachmentReference){
            .attachment = i,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        attachments[i] = (VkAttachmentDescription){
            .format = i3_vk_convert_format(image_view_desc->format),
            .samples = image_desc->samples,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        image_views[i] = ((i3_vk_image_view_o*)desc->depth_attachment->image_view->self)->handle;
    }

    // create render pass
    VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = desc->color_attachment_count,
        .pColorAttachments = attachment_ref,
        .pDepthStencilAttachment = desc->depth_attachment ? &attachment_ref[i] : NULL,
    };

    VkRenderPassCreateInfo render_pass_ci = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .attachmentCount = attachment_count,
        .pAttachments = attachments,
    };

    i3_vk_check(vkCreateRenderPass(device->handle, &render_pass_ci, NULL, &framebuffer->render_pass));

    // create framebuffer
    VkFramebufferCreateInfo framebuffer_ci = {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .width = desc->width,
        .height = desc->height,
        .layers = desc->layers,
        .renderPass = framebuffer->render_pass,
        .attachmentCount = attachment_count,
        .pAttachments = image_views,
    };

    i3_vk_check(vkCreateFramebuffer(device->handle, &framebuffer_ci, NULL, &framebuffer->handle));

    i3_arena_destroy(&arena);

    return &framebuffer->iface;
}