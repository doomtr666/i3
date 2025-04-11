#include "buffer.h"

// resource interface
static void i3_vk_buffer_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_buffer_o* buffer = (i3_vk_buffer_o*)self;

    buffer->use_count++;
}

static void i3_vk_buffer_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_buffer_o* buffer = (i3_vk_buffer_o*)self;

    buffer->use_count--;

    if (buffer->use_count == 0)
    {
        vmaDestroyBuffer(buffer->device->vma, buffer->handle, buffer->allocation);
        i3_memory_pool_free(&buffer->device->buffer_pool, buffer);
    }
}

static int32_t i3_vk_buffer_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_buffer_o* buffer = (i3_vk_buffer_o*)self;

    return buffer->use_count;
}

static void i3_vk_buffer_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);
    i3_vk_buffer_o* buffer = (i3_vk_buffer_o*)self;

    if (buffer->device->backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsObjectNameInfoEXT name_info = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_BUFFER,
                                                   .objectHandle = (uintptr_t)buffer->handle,
                                                   .pObjectName = name };
        buffer->device->backend->ext.vkSetDebugUtilsObjectNameEXT(buffer->device->handle, &name_info);
    }
}

// buffer interface

static const i3_rbk_buffer_desc_t* i3_vk_buffer_get_desc(i3_rbk_buffer_o* self)
{
    assert(self != NULL);
    i3_vk_buffer_o* buffer = (i3_vk_buffer_o*)self;

    return &buffer->desc;
}

static i3_rbk_resource_i* i3_vk_buffer_get_resource_i(i3_rbk_buffer_o* self)
{
    assert(self != NULL);
    i3_vk_buffer_o* buffer = (i3_vk_buffer_o*)self;

    return &buffer->base;
}

static void i3_vk_buffer_destroy(i3_rbk_buffer_o* self)
{
    assert(self != NULL);
    i3_vk_buffer_o* buffer = (i3_vk_buffer_o*)self;

    buffer->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_buffer_o i3_vk_buffer_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_buffer_add_ref,
        .release = i3_vk_buffer_release,
        .get_use_count = i3_vk_buffer_get_use_count,
        .set_debug_name = i3_vk_buffer_set_debug_name,
    },
    .iface =
    {
        .get_desc = i3_vk_buffer_get_desc,
        .get_resource_i = i3_vk_buffer_get_resource_i,
        .destroy = i3_vk_buffer_destroy,
    },
};

i3_rbk_buffer_i* i3_vk_device_create_buffer(i3_rbk_device_o* self, const i3_rbk_buffer_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;

    i3_vk_buffer_o* buffer = i3_memory_pool_alloc(&device->buffer_pool);
    assert(buffer != NULL);

    *buffer = i3_vk_buffer_iface_;
    buffer->base.self = (i3_rbk_resource_o*)buffer;
    buffer->iface.self = (i3_rbk_buffer_o*)buffer;
    buffer->device = device;
    buffer->desc = *desc;
    buffer->use_count = 1;

    // default VM flags
    VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_GPU_ONLY;

    // buffer ci
    VkBufferCreateInfo buffer_ci =
    {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = desc->size,
        .usage = buffer_usage,
    };

    // alloc ci
    VmaAllocationCreateInfo alloc_ci = { .usage = memory_usage };

    i3_vk_check(vmaCreateBuffer(device->vma, &buffer_ci, &alloc_ci, &buffer->handle, &buffer->allocation, NULL));

    return &buffer->iface;
}