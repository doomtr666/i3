#include "image.h"
#include "convert.h"

// resource interface

static void i3_vk_image_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);

    i3_vk_image_o* image = (i3_vk_image_o*)self;

    image->use_count++;
}

static void i3_vk_image_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);

    i3_vk_image_o* image = (i3_vk_image_o*)self;

    image->use_count--;

    if (image->use_count == 0)
    {
        vmaDestroyImage(image->device->vma, image->handle, image->allocation);
        i3_vk_destroy_image_state(&image->state);
        i3_memory_pool_free(&image->device->image_pool, image);
    }
}

static int32_t i3_vk_image_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);

    i3_vk_image_o* image = (i3_vk_image_o*)self;

    return image->use_count;
}

static void i3_vk_image_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);

    i3_vk_image_o* image = (i3_vk_image_o*)self;

    if (image->device->backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsObjectNameInfoEXT name_info = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_IMAGE,
                                                   .objectHandle = (uintptr_t)image->handle,
                                                   .pObjectName = name};
        image->device->backend->ext.vkSetDebugUtilsObjectNameEXT(image->device->handle, &name_info);
    }
}

// image interface

static const i3_rbk_image_desc_t* i3_vk_image_get_desc(i3_rbk_image_o* self)
{
    assert(self != NULL);

    i3_vk_image_o* image = (i3_vk_image_o*)self;
    return &image->desc;
}

static i3_rbk_resource_i* i3_vk_image_get_resource(i3_rbk_image_o* self)
{
    assert(self != NULL);

    i3_vk_image_o* image = (i3_vk_image_o*)self;
    return &image->base;
}

static void i3_vk_image_destroy(i3_rbk_image_o* self)
{
    assert(self != NULL);

    i3_vk_image_o* image = (i3_vk_image_o*)self;

    image->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_image_o i3_vk_image_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_image_add_ref,
        .release = i3_vk_image_release,
        .get_use_count = i3_vk_image_get_use_count,
        .set_debug_name = i3_vk_image_set_debug_name,
    },
    .iface =
    {
        .get_desc = i3_vk_image_get_desc,
        .get_resource = i3_vk_image_get_resource,
        .destroy = i3_vk_image_destroy,
    },
};

i3_rbk_image_i* i3_vk_device_create_image(i3_rbk_device_o* self, const i3_rbk_image_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;

    i3_vk_image_o* image = i3_memory_pool_alloc(&device->image_pool);
    assert(image != NULL);

    *image = i3_vk_image_iface_;
    image->base.self = (i3_rbk_resource_o*)image;
    image->iface.self = (i3_rbk_image_o*)image;
    image->device = device;
    image->desc = *desc;
    image->use_count = 1;

    VkImageUsageFlags usage
        = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (i3_vk_is_depth_format(desc->format))
        usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    else
        usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    // image info
    VkImageCreateInfo image_ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                  .imageType = i3_vk_convert_image_type(desc->type),
                                  .format = i3_vk_convert_format(desc->format),
                                  .extent = {.width = desc->width, .height = desc->height, .depth = desc->depth},
                                  .mipLevels = desc->mip_levels,
                                  .arrayLayers = desc->array_layers,
                                  .samples = i3_vk_convert_sample_count(desc->samples),
                                  .tiling = VK_IMAGE_TILING_OPTIMAL,
                                  .usage = usage,
                                  .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                  .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

    // allocation info
    VmaAllocationCreateInfo alloc_ci = {.usage = VMA_MEMORY_USAGE_GPU_ONLY};

    i3_vk_check(vmaCreateImage(device->vma, &image_ci, &alloc_ci, &image->handle, &image->allocation, NULL));

    // create barrier info
    i3_vk_init_image_state(&image->state, desc->array_layers, desc->mip_levels);

    return &image->iface;
}