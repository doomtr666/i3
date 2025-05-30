#include "native/core/arena.h"

#include "descriptor_set_layout.h"
#include "pipeline_layout.h"

// resource interface

static void i3_vk_pipeline_layout_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_pipeline_layout_o* pipeline_layout = (i3_vk_pipeline_layout_o*)self;

    pipeline_layout->use_count++;
}

static void i3_vk_pipeline_layout_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_pipeline_layout_o* pipeline_layout = (i3_vk_pipeline_layout_o*)self;

    if (--pipeline_layout->use_count == 0)
    {
        // destroy pipeline layout
        vkDestroyPipelineLayout(pipeline_layout->device->handle, pipeline_layout->handle, NULL);

        // destroy use list
        i3_vk_use_list_destroy(&pipeline_layout->use_list);

        // free pipeline layout
        i3_memory_pool_free(&pipeline_layout->device->pipeline_layout_pool, pipeline_layout);
    }
}

static int32_t i3_vk_pipeline_layout_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_pipeline_layout_o* pipeline_layout = (i3_vk_pipeline_layout_o*)self;

    return pipeline_layout->use_count;
}

static void i3_vk_pipeline_layout_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);
    i3_vk_pipeline_layout_o* pipeline_layout = (i3_vk_pipeline_layout_o*)self;

    if (pipeline_layout->device->backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsObjectNameInfoEXT name_info = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                                   .objectHandle = (uintptr_t)pipeline_layout->handle,
                                                   .pObjectName = name};
        pipeline_layout->device->backend->ext.vkSetDebugUtilsObjectNameEXT(pipeline_layout->device->handle, &name_info);
    }
}

// pipeline layout interface

static i3_rbk_resource_i* i3_vk_pipeline_layout_get_resource(i3_rbk_pipeline_layout_o* self)
{
    assert(self != NULL);
    i3_vk_pipeline_layout_o* pipeline_layout = (i3_vk_pipeline_layout_o*)self;

    return &pipeline_layout->base;
}

static void i3_vk_pipeline_layout_destroy(i3_rbk_pipeline_layout_o* self)
{
    assert(self != NULL);
    i3_vk_pipeline_layout_o* pipeline_layout = (i3_vk_pipeline_layout_o*)self;

    pipeline_layout->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_pipeline_layout_o i3_vk_pipeline_layout_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_pipeline_layout_add_ref,
        .release = i3_vk_pipeline_layout_release,
        .get_use_count = i3_vk_pipeline_layout_get_use_count,
        .set_debug_name = i3_vk_pipeline_layout_set_debug_name,
    },
    .iface =
    {
        .get_resource = i3_vk_pipeline_layout_get_resource,
        .destroy = i3_vk_pipeline_layout_destroy,
    },
};

// create pipeline layout
i3_rbk_pipeline_layout_i* i3_vk_device_create_pipeline_layout(i3_rbk_device_o* self,
                                                              const i3_rbk_pipeline_layout_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;
    i3_vk_pipeline_layout_o* pipeline_layout = i3_memory_pool_alloc(&device->pipeline_layout_pool);
    assert(pipeline_layout != NULL);
    *pipeline_layout = i3_vk_pipeline_layout_iface_;
    pipeline_layout->base.self = (i3_rbk_resource_o*)pipeline_layout;
    pipeline_layout->iface.self = (i3_rbk_pipeline_layout_o*)pipeline_layout;
    pipeline_layout->device = device;
    pipeline_layout->use_count = 1;

    // initialize use list
    i3_vk_use_list_init(&pipeline_layout->use_list, device);

    i3_arena_t arena;
    i3_arena_init(&arena, I3_KB);

    // create layout
    VkPipelineLayoutCreateInfo layout_ci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = desc->set_layout_count,
        .pushConstantRangeCount = desc->push_constant_range_count,
    };

    //  descriptor set layouts
    if (desc->set_layout_count > 0)
    {
        VkDescriptorSetLayout* set_layouts
            = i3_arena_alloc(&arena, sizeof(VkDescriptorSetLayout) * desc->set_layout_count);

        for (uint32_t i = 0; i < desc->set_layout_count; ++i)
        {
            i3_rbk_descriptor_set_layout_i* set_layout = (i3_rbk_descriptor_set_layout_i*)desc->set_layouts[i];

            // get handle
            set_layouts[i] = ((i3_vk_descriptor_set_layout_o*)set_layout->self)->handle;

            // retain the set layout
            i3_vk_use_list_add(&pipeline_layout->use_list, set_layout);
        }

        layout_ci.pSetLayouts = set_layouts;
    }

    if (desc->push_constant_range_count > 0)
    {
        VkPushConstantRange* push_constant_ranges
            = i3_arena_alloc(&arena, sizeof(VkPushConstantRange) * desc->push_constant_range_count);
        for (uint32_t i = 0; i < desc->push_constant_range_count; ++i)
        {
            const i3_rbk_push_constant_range_t* range = &desc->push_constant_ranges[i];
            push_constant_ranges[i] = (VkPushConstantRange){
                .stageFlags = i3_vk_convert_shader_stage_flags(range->stage_flags),
                .offset = range->offset,
                .size = range->size,
            };
        }
        layout_ci.pPushConstantRanges = push_constant_ranges;
    }

    i3_vk_check(vkCreatePipelineLayout(device->handle, &layout_ci, NULL, &pipeline_layout->handle));

    i3_arena_destroy(&arena);

    return &pipeline_layout->iface;
}