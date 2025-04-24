#include "native/core/arena.h"

#include "descriptor_set_layout.h"

// resource interface
static void i3_vk_descriptor_set_layout_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_descriptor_set_layout_o* descriptor_set_layout = (i3_vk_descriptor_set_layout_o*)self;

    descriptor_set_layout->use_count++;
}

static void i3_vk_descriptor_set_layout_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_descriptor_set_layout_o* descriptor_set_layout = (i3_vk_descriptor_set_layout_o*)self;

    descriptor_set_layout->use_count--;

    if (descriptor_set_layout->use_count == 0)
    {
        vkDestroyDescriptorSetLayout(descriptor_set_layout->device->handle, descriptor_set_layout->handle, NULL);
        i3_memory_pool_free(&descriptor_set_layout->device->descriptor_set_layout_pool, descriptor_set_layout);
    }
}

static int32_t i3_vk_descriptor_set_layout_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_descriptor_set_layout_o* descriptor_set_layout = (i3_vk_descriptor_set_layout_o*)self;

    return descriptor_set_layout->use_count;
}

static void i3_vk_descriptor_set_layout_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);
    i3_vk_descriptor_set_layout_o* descriptor_set_layout = (i3_vk_descriptor_set_layout_o*)self;

    if (descriptor_set_layout->device->backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsObjectNameInfoEXT name_info = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                                                   .objectHandle = (uintptr_t)descriptor_set_layout->handle,
                                                   .pObjectName = name};
        descriptor_set_layout->device->backend->ext.vkSetDebugUtilsObjectNameEXT(descriptor_set_layout->device->handle,
                                                                                 &name_info);
    }
}

// descriptor set layout interface

static i3_rbk_resource_i* i3_vk_descriptor_set_layout_get_resource(i3_rbk_descriptor_set_layout_o* self)
{
    assert(self != NULL);
    i3_vk_descriptor_set_layout_o* descriptor_set_layout = (i3_vk_descriptor_set_layout_o*)self;

    return &descriptor_set_layout->base;
}

static void i3_vk_descriptor_set_layout_destroy(i3_rbk_descriptor_set_layout_o* self)
{
    assert(self != NULL);
    i3_vk_descriptor_set_layout_o* descriptor_set_layout = (i3_vk_descriptor_set_layout_o*)self;

    descriptor_set_layout->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_descriptor_set_layout_o i3_vk_descriptor_set_layout_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_descriptor_set_layout_add_ref,
        .release = i3_vk_descriptor_set_layout_release,
        .get_use_count = i3_vk_descriptor_set_layout_get_use_count,
        .set_debug_name = i3_vk_descriptor_set_layout_set_debug_name,
    },
    .iface =
    {
        .get_resource = i3_vk_descriptor_set_layout_get_resource,
        .destroy = i3_vk_descriptor_set_layout_destroy,
    },
};

// create descriptor set layout
i3_rbk_descriptor_set_layout_i* i3_vk_device_create_descriptor_set_layout(
    i3_rbk_device_o* self,
    const i3_rbk_descriptor_set_layout_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;
    i3_vk_descriptor_set_layout_o* descriptor_set_layout = i3_memory_pool_alloc(&device->descriptor_set_layout_pool);
    assert(descriptor_set_layout != NULL);
    *descriptor_set_layout = i3_vk_descriptor_set_layout_iface_;
    descriptor_set_layout->base.self = (i3_rbk_resource_o*)descriptor_set_layout;
    descriptor_set_layout->iface.self = (i3_rbk_descriptor_set_layout_o*)descriptor_set_layout;
    descriptor_set_layout->device = device;
    descriptor_set_layout->use_count = 1;

    // create layout
    VkDescriptorSetLayoutCreateInfo layout_ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = desc->binding_count,
    };

    i3_arena_t arena;
    i3_arena_init(&arena, I3_KB);

    if (desc->binding_count > 0)
    {
        VkDescriptorSetLayoutBinding* bindings
            = i3_arena_alloc(&arena, sizeof(VkDescriptorSetLayoutBinding) * desc->binding_count);
        for (uint32_t i = 0; i < desc->binding_count; i++)
        {
            const i3_rbk_descriptor_set_layout_binding_t* binding = &desc->bindings[i];
            bindings[i] = (VkDescriptorSetLayoutBinding){
                .binding = binding->binding,
                .descriptorType = i3_vk_convert_descriptor_type(binding->descriptor_type),
                .descriptorCount = binding->descriptor_count,
                .stageFlags = i3_vk_convert_shader_stage_flags(binding->stage_flags),
            };

            // TODO: immutable samplers
        }

        layout_ci.pBindings = bindings;
    }

    i3_vk_check(vkCreateDescriptorSetLayout(device->handle, &layout_ci, NULL, &descriptor_set_layout->handle));

    i3_arena_free(&arena);

    return &descriptor_set_layout->iface;
}