#include "sampler.h"
#include "convert.h"

// resource interface
static void i3_vk_sampler_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);

    i3_vk_sampler_o* sampler = (i3_vk_sampler_o*)self;

    sampler->use_count++;
}

static void i3_vk_sampler_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_sampler_o* sampler = (i3_vk_sampler_o*)self;

    sampler->use_count--;

    if (sampler->use_count == 0)
    {
        vkDestroySampler(sampler->device->handle, sampler->handle, NULL);
        i3_memory_pool_free(&sampler->device->sampler_pool, sampler);
    }
}

static int32_t i3_vk_sampler_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_sampler_o* sampler = (i3_vk_sampler_o*)self;

    return sampler->use_count;
}

static void i3_vk_sampler_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);

    i3_vk_sampler_o* sampler = (i3_vk_sampler_o*)self;

    if (sampler->device->backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsObjectNameInfoEXT name_info = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_SAMPLER,
                                                   .objectHandle = (uintptr_t)sampler->handle,
                                                   .pObjectName = name };
        sampler->device->backend->ext.vkSetDebugUtilsObjectNameEXT(sampler->device->handle, &name_info);
    }
}

// sampler interface

static const i3_rbk_sampler_desc_t* i3_vk_sampler_get_desc(i3_rbk_sampler_o* self)
{
    assert(self != NULL);

    i3_vk_sampler_o* sampler = (i3_vk_sampler_o*)self;

    return &sampler->desc;
}

static i3_rbk_resource_i* i3_vk_sampler_get_resource_i(i3_rbk_sampler_o* self)
{
    assert(self != NULL);

    i3_vk_sampler_o* sampler = (i3_vk_sampler_o*)self;

    return &sampler->base;
}

static void i3_vk_sampler_destroy(i3_rbk_sampler_o* self)
{
    assert(self != NULL);

    i3_vk_sampler_o* sampler = (i3_vk_sampler_o*)self;

    sampler->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_sampler_o i3_vk_sampler_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_sampler_add_ref,
        .release = i3_vk_sampler_release,
        .get_use_count = i3_vk_sampler_get_use_count,
        .set_debug_name = i3_vk_sampler_set_debug_name,
    },
    .iface =
    {
        .get_desc = i3_vk_sampler_get_desc,
        .get_resource_i = i3_vk_sampler_get_resource_i,
        .destroy = i3_vk_sampler_destroy,
    },
};

// device interface

i3_rbk_sampler_i* i3_vk_device_create_sampler(i3_rbk_device_o* self, const i3_rbk_sampler_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;

    i3_vk_sampler_o* sampler = i3_memory_pool_alloc(&device->sampler_pool);
    assert(sampler != NULL);

    *sampler = i3_vk_sampler_iface_;
    sampler->base.self = (i3_rbk_resource_o*)sampler;
    sampler->iface.self = (i3_rbk_sampler_o*)sampler;
    sampler->device = device;
    sampler->desc = *desc;
    sampler->use_count = 1;

    VkSamplerCreateInfo sampler_ci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = i3_vk_convert_filter(desc->mag_filter),
        .minFilter = i3_vk_convert_filter(desc->min_filter),
        .mipmapMode = i3_vk_convert_sampler_mipmap_mode(desc->mipmap_mode),
        .addressModeU = i3_vk_convert_sampler_address_mode(desc->address_mode_u),
        .addressModeV = i3_vk_convert_sampler_address_mode(desc->address_mode_v),
        .addressModeW = i3_vk_convert_sampler_address_mode(desc->address_mode_w),
        .mipLodBias = desc->mip_lod_bias,
        .anisotropyEnable = desc->anisotropy_enable,
        .maxAnisotropy = desc->max_anisotropy,
        .compareEnable = desc->compare_enable,
        .compareOp = i3_vk_convert_compare_op(desc->compare_op),
        .minLod = desc->min_lod,
        .maxLod = desc->max_lod,
        .borderColor = i3_vk_convert_border_color(desc->border_color),
        .unnormalizedCoordinates = desc->unnormalized_coordinates,
    };

    i3_vk_check(vkCreateSampler(device->handle, &sampler_ci, NULL, &sampler->handle));

    return &sampler->iface;
}

