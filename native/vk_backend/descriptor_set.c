#include "native/core/arena.h"

#include "buffer.h"
#include "descriptor_set.h"
#include "descriptor_set_layout.h"
#include "image_view.h"
#include "sampler.h"

// resource interface
static void i3_vk_descriptor_set_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_descriptor_set_o* descriptor_set = (i3_vk_descriptor_set_o*)self;

    descriptor_set->use_count++;
}

static void i3_vk_descriptor_set_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_descriptor_set_o* descriptor_set = (i3_vk_descriptor_set_o*)self;

    if (--descriptor_set->use_count == 0)
    {
        // destroy the descriptor set
        vkFreeDescriptorSets(descriptor_set->device->handle, descriptor_set->device->descriptor_pool, 1,
                             &descriptor_set->handle);

        // release the descriptor set layout
        i3_rbk_resource_release(descriptor_set->layout);

        // free the descriptor set
        i3_memory_pool_free(&descriptor_set->device->descriptor_set_pool, descriptor_set);
    }
}

static int32_t i3_vk_descriptor_set_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_descriptor_set_o* descriptor_set = (i3_vk_descriptor_set_o*)self;

    return descriptor_set->use_count;
}

static void i3_vk_descriptor_set_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);
    i3_vk_descriptor_set_o* descriptor_set = (i3_vk_descriptor_set_o*)self;

    if (descriptor_set->device->backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsObjectNameInfoEXT name_info = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_DESCRIPTOR_SET,
                                                   .objectHandle = (uintptr_t)descriptor_set->handle,
                                                   .pObjectName = name};
        descriptor_set->device->backend->ext.vkSetDebugUtilsObjectNameEXT(descriptor_set->device->handle, &name_info);
    }
}

// descriptor set interface

static i3_rbk_resource_i* i3_vk_descriptor_set_get_resource(i3_rbk_descriptor_set_o* self)
{
    assert(self != NULL);
    i3_vk_descriptor_set_o* descriptor_set = (i3_vk_descriptor_set_o*)self;

    return &descriptor_set->base;
}

static void i3_vk_descriptor_set_destroy(i3_rbk_descriptor_set_o* self)
{
    assert(self != NULL);
    i3_vk_descriptor_set_o* descriptor_set = (i3_vk_descriptor_set_o*)self;

    // release the resource
    descriptor_set->base.release((i3_rbk_resource_o*)self);
}

static void i3_vk_descriptor_set_update(i3_rbk_descriptor_set_o* self,
                                        uint32_t write_count,
                                        const i3_rbk_descriptor_set_write_t* writes)
{
    assert(self != NULL);
    assert(writes != NULL);
    assert(write_count > 0);

    i3_vk_descriptor_set_o* descriptor_set = (i3_vk_descriptor_set_o*)self;
    i3_vk_device_o* device = descriptor_set->device;

    i3_arena_t arena;
    i3_arena_init(&arena, I3_KB);

    VkWriteDescriptorSet* vk_writes = i3_arena_alloc(&arena, sizeof(VkWriteDescriptorSet) * write_count);
    VkDescriptorImageInfo* image_infos = i3_arena_alloc(&arena, sizeof(VkDescriptorImageInfo) * write_count);
    VkDescriptorBufferInfo* buffer_infos = i3_arena_alloc(&arena, sizeof(VkDescriptorBufferInfo) * write_count);

    for (uint32_t i = 0; i < write_count; i++)
    {
        const i3_rbk_descriptor_set_write_t* write = &writes[i];
        VkWriteDescriptorSet* vk_write = &vk_writes[i];
        *vk_write = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = NULL,
            .dstSet = descriptor_set->handle,
            .dstBinding = write->binding,
            .dstArrayElement = write->array_element,
            .descriptorCount = 1,
            .descriptorType = i3_vk_convert_descriptor_type(write->descriptor_type),
        };

        switch (write->descriptor_type)
        {
            case I3_RBK_DESCRIPTOR_TYPE_SAMPLER:
                vk_write->pImageInfo = &image_infos[i];
                image_infos[i] = (VkDescriptorImageInfo){
                    .sampler = ((i3_vk_sampler_o*)write->sampler->self)->handle,
                };
                break;

            case I3_RBK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                vk_write->pImageInfo = &image_infos[i];
                image_infos[i] = (VkDescriptorImageInfo){
                    .sampler = ((i3_vk_sampler_o*)write->sampler->self)->handle,
                    .imageView = ((i3_vk_image_view_o*)write->image->self)->handle,
                    // TODO: .imageLayout = 0,
                };
                break;

            case I3_RBK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case I3_RBK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                vk_write->pImageInfo = &image_infos[i];
                image_infos[i] = (VkDescriptorImageInfo){
                    .imageView = ((i3_vk_image_view_o*)write->image->self)->handle,
                    // TODO: .imageLayout = 0,
                };
                break;

            case I3_RBK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case I3_RBK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                vk_write->pBufferInfo = &buffer_infos[i];
                buffer_infos[i] = (VkDescriptorBufferInfo){
                    .buffer = ((i3_vk_buffer_o*)write->buffer->self)->handle,
                    .range = ((i3_vk_buffer_o*)write->buffer->self)->desc.size,
                };
                break;
            default:
                assert(0);
        }
    }

    vkUpdateDescriptorSets(device->handle, write_count, vk_writes, 0, NULL);

    i3_arena_destroy(&arena);
}

static i3_vk_descriptor_set_o i3_vk_descriptor_set_iface_ = {
    .base = {
        .add_ref = i3_vk_descriptor_set_add_ref,
        .release = i3_vk_descriptor_set_release,
        .get_use_count = i3_vk_descriptor_set_get_use_count,
        .set_debug_name = i3_vk_descriptor_set_set_debug_name,
    },
    .iface = {
        .get_resource = i3_vk_descriptor_set_get_resource,
        .update = i3_vk_descriptor_set_update,
        .destroy = i3_vk_descriptor_set_destroy,
    },
};

i3_rbk_descriptor_set_i* i3_vk_device_create_descriptor_set(i3_rbk_device_o* self,
                                                            i3_rbk_descriptor_set_layout_i* layout)
{
    assert(self != NULL);
    assert(layout != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;
    i3_vk_descriptor_set_o* descriptor_set = i3_memory_pool_alloc(&device->descriptor_set_pool);

    *descriptor_set = i3_vk_descriptor_set_iface_;
    descriptor_set->base.self = (i3_rbk_resource_o*)descriptor_set;
    descriptor_set->iface.self = (i3_rbk_descriptor_set_o*)descriptor_set;
    descriptor_set->device = device;
    descriptor_set->use_count = 1;
    descriptor_set->layout = layout;

    // retain the descriptor set layout
    i3_rbk_resource_add_ref(descriptor_set->layout);

    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = device->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &((i3_vk_descriptor_set_layout_o*)(layout->self))->handle,
    };

    i3_vk_check(vkAllocateDescriptorSets(device->handle, &alloc_info, &descriptor_set->handle));

    return &descriptor_set->iface;
}
