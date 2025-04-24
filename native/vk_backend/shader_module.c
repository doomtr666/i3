#include "shader_module.h"

// resource interface
static void i3_vk_shader_module_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_shader_module_o* module = (i3_vk_shader_module_o*)self;

    module->use_count++;
}

static void i3_vk_shader_module_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_shader_module_o* module = (i3_vk_shader_module_o*)self;

    module->use_count--;

    if (module->use_count == 0)
    {
        vkDestroyShaderModule(module->device->handle, module->handle, NULL);
        i3_memory_pool_free(&module->device->shader_module_pool, module);
    }
}

static int32_t i3_vk_shader_module_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_shader_module_o* module = (i3_vk_shader_module_o*)self;

    return module->use_count;
}

static void i3_vk_shader_module_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);
    i3_vk_shader_module_o* module = (i3_vk_shader_module_o*)self;

    if (module->device->backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsObjectNameInfoEXT name_info = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_SHADER_MODULE,
                                                   .objectHandle = (uintptr_t)module->handle,
                                                   .pObjectName = name};
        module->device->backend->ext.vkSetDebugUtilsObjectNameEXT(module->device->handle, &name_info);
    }
}

// module interface

static const i3_rbk_shader_module_desc_t* i3_vk_shader_module_get_desc(i3_rbk_shader_module_o* self)
{
    assert(self != NULL);
    i3_vk_shader_module_o* module = (i3_vk_shader_module_o*)self;

    return &module->desc;
}

static i3_rbk_resource_i* i3_vk_shader_module_get_resource(i3_rbk_shader_module_o* self)
{
    assert(self != NULL);
    i3_vk_shader_module_o* module = (i3_vk_shader_module_o*)self;

    return &module->base;
}

static void i3_vk_shader_module_destroy(i3_rbk_shader_module_o* self)
{
    assert(self != NULL);
    i3_vk_shader_module_o* module = (i3_vk_shader_module_o*)self;

    module->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_shader_module_o i3_vk_shader_module_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_shader_module_add_ref,
        .release = i3_vk_shader_module_release,
        .get_use_count = i3_vk_shader_module_get_use_count,
        .set_debug_name = i3_vk_shader_module_set_debug_name,
    },
    .iface =
    {
        .get_desc = i3_vk_shader_module_get_desc,
        .get_resource = i3_vk_shader_module_get_resource,
        .destroy = i3_vk_shader_module_destroy,
    },
};

i3_rbk_shader_module_i* i3_vk_device_create_shader_module(i3_rbk_device_o* self,
                                                          const i3_rbk_shader_module_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;

    i3_vk_shader_module_o* module = i3_memory_pool_alloc(&device->shader_module_pool);
    assert(module != NULL);

    *module = i3_vk_shader_module_iface_;
    module->base.self = (i3_rbk_resource_o*)module;
    module->iface.self = (i3_rbk_shader_module_o*)module;
    module->device = device;
    module->desc = *desc;
    module->use_count = 1;

    VkShaderModuleCreateInfo shader_module_ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                                 .codeSize = desc->code_size,
                                                 .pCode = (const uint32_t*)desc->code};

    i3_vk_check(vkCreateShaderModule(device->handle, &shader_module_ci, NULL, &module->handle));

    return &module->iface;
}
