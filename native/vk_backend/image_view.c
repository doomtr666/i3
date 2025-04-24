#include "image_view.h"
#include "convert.h"

// resource interface

static void i3_vk_image_view_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_image_view_o* image_view = (i3_vk_image_view_o*)self;

    image_view->use_count++;
}

static void i3_vk_image_view_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_image_view_o* image_view = (i3_vk_image_view_o*)self;

    image_view->use_count--;

    if (image_view->use_count == 0)
    {
        // destroy image view
        vkDestroyImageView(image_view->device->handle, image_view->handle, NULL);

        // release image
        i3_rbk_resource_i* image_res = &image_view->image->base;
        image_res->release(image_res->self);

        // free memory
        i3_memory_pool_free(&image_view->device->image_view_pool, image_view);
    }
}

static int32_t i3_vk_image_view_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_image_view_o* image_view = (i3_vk_image_view_o*)self;

    return image_view->use_count;
}

static void i3_vk_image_view_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);
    i3_vk_image_view_o* image_view = (i3_vk_image_view_o*)self;

    if (image_view->device->backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsObjectNameInfoEXT name_info = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_IMAGE_VIEW,
                                                   .objectHandle = (uintptr_t)image_view->handle,
                                                   .pObjectName = name};
        image_view->device->backend->ext.vkSetDebugUtilsObjectNameEXT(image_view->device->handle, &name_info);
    }
}

// image view interface

static const i3_rbk_image_view_desc_t* i3_vk_image_view_get_desc(i3_rbk_image_view_o* self)
{
    assert(self != NULL);
    i3_vk_image_view_o* image_view = (i3_vk_image_view_o*)self;

    return &image_view->info;
}

static i3_rbk_image_i* i3_vk_image_view_get_image(i3_rbk_image_view_o* self)
{
    assert(self != NULL);
    i3_vk_image_view_o* image_view = (i3_vk_image_view_o*)self;

    return &image_view->image->iface;
}

static i3_rbk_resource_i* i3_vk_image_view_get_resource(i3_rbk_image_view_o* self)
{
    assert(self != NULL);
    i3_vk_image_view_o* image_view = (i3_vk_image_view_o*)self;

    return &image_view->base;
}

static void i3_vk_image_view_destroy(i3_rbk_image_view_o* self)
{
    assert(self != NULL);
    i3_vk_image_view_o* image_view = (i3_vk_image_view_o*)self;

    image_view->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_image_view_o i3_vk_image_view_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_image_view_add_ref,
        .release = i3_vk_image_view_release,
        .get_use_count = i3_vk_image_view_get_use_count,
        .set_debug_name = i3_vk_image_view_set_debug_name,
    },
    .iface =
    {
        .get_desc = i3_vk_image_view_get_desc,
        .get_image = i3_vk_image_view_get_image,
        .get_resource = i3_vk_image_view_get_resource,
        .destroy = i3_vk_image_view_destroy,
    },
};

i3_rbk_image_view_i* i3_vk_device_create_image_view(i3_rbk_device_o* self,
                                                    i3_rbk_image_i* image,
                                                    const i3_rbk_image_view_desc_t* info)
{
    assert(self != NULL);
    assert(image != NULL);
    assert(info != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;

    i3_vk_image_view_o* image_view = i3_memory_pool_alloc(&device->image_view_pool);
    assert(image_view != NULL);

    *image_view = i3_vk_image_view_iface_;
    image_view->base.self = (i3_rbk_resource_o*)image_view;
    image_view->iface.self = (i3_rbk_image_view_o*)image_view;
    image_view->device = device;
    image_view->info = *info;
    image_view->image = (i3_vk_image_o*)image->self;
    image_view->use_count = 1;

    // add ref to image
    i3_rbk_resource_i* image_res = &image_view->image->base;
    image_res->add_ref(image_res->self);

    // create image view
    VkImageViewCreateInfo image_view_ci
        = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
           .image = image_view->image->handle,
           .viewType = i3_vk_convert_image_view_type(info->type),
           .format = i3_vk_convert_format(info->format),
           .components = {.r = i3_vk_convert_component_swizzle(info->r),
                          .g = i3_vk_convert_component_swizzle(info->g),
                          .b = i3_vk_convert_component_swizzle(info->b),
                          .a = i3_vk_convert_component_swizzle(info->a)},
           .subresourceRange = {.aspectMask = i3_vk_convert_image_aspect_flags(info->aspect_mask),
                                .baseMipLevel = info->base_mip_level,
                                .levelCount = info->level_count,
                                .baseArrayLayer = info->base_array_layer,
                                .layerCount = info->layer_count}};

    i3_vk_check(vkCreateImageView(device->handle, &image_view_ci, NULL, &image_view->handle));

    // return image view interface
    return &image_view->iface;
}