#include "native/core/arena.h"

#include "pipeline.h"
#include "convert.h"

// resource interface

static void i3_vk_pipeline_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_pipeline_o* pipeline = (i3_vk_pipeline_o*)self;

    pipeline->use_count++;
}

static void i3_vk_pipeline_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_pipeline_o* pipeline = (i3_vk_pipeline_o*)self;

    pipeline->use_count--;
    if (pipeline->use_count == 0)
    {
        vkDestroyPipelineLayout(pipeline->device->handle, pipeline->layout, NULL);
        //vkDestroyPipeline(pipeline->device->handle, pipeline->handle, NULL);
        i3_memory_pool_free(&pipeline->device->pipeline_pool, pipeline);
    }
}

static int32_t i3_vk_pipeline_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_pipeline_o* pipeline = (i3_vk_pipeline_o*)self;

    return pipeline->use_count;
}

static void i3_vk_buffer_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);
    i3_vk_pipeline_o* pipeline = (i3_vk_pipeline_o*)self;

    if (pipeline->device->backend->ext.VK_EXT_debug_utils_supported)
    {
        VkDebugUtilsObjectNameInfoEXT name_info = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_PIPELINE,
                                                   .objectHandle = (uintptr_t)pipeline->handle,
                                                   .pObjectName = name };
        pipeline->device->backend->ext.vkSetDebugUtilsObjectNameEXT(pipeline->device->handle, &name_info);
    }
}

// pipeline interface

static i3_rbk_resource_i* i3_vk_pipeline_get_resource_i(i3_rbk_pipeline_o* self)
{
    assert(self != NULL);
    i3_vk_pipeline_o* pipeline = (i3_vk_pipeline_o*)self;

    return &pipeline->base;
}

static void i3_vk_pipeline_destroy(i3_rbk_pipeline_o* self)
{
    assert(self != NULL);
    i3_vk_pipeline_o* pipeline = (i3_vk_pipeline_o*)self;

    pipeline->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_pipeline_o i3_vk_pipeline_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_pipeline_add_ref,
        .release = i3_vk_pipeline_release,
        .get_use_count = i3_vk_pipeline_get_use_count,
        .set_debug_name = i3_vk_buffer_set_debug_name,
    },
    .iface =
    {
        .get_resource_i = i3_vk_pipeline_get_resource_i,
        .destroy = i3_vk_pipeline_destroy,
    },
};

// allocate pipeline
static i3_vk_pipeline_o* i3_vk_allocate_pipeline(i3_vk_device_o* device)
{
    assert(device != NULL);

    i3_vk_pipeline_o* pipeline = i3_memory_pool_alloc(&device->pipeline_pool);
    assert(pipeline != NULL);
    *pipeline = i3_vk_pipeline_iface_;
    pipeline->base.self = (i3_rbk_resource_o*)pipeline;
    pipeline->iface.self = (i3_rbk_pipeline_o*)pipeline;
    pipeline->device = device;
    pipeline->use_count = 1;
    return pipeline;
}

// create pipeline layout
static VkPipelineLayout i3_vk_create_pipeline_layout(i3_arena_t* arena, i3_vk_device_o* device, const i3_rbk_pipeline_layout_desc_t* desc)
{
    assert(device != NULL);
    assert(desc != NULL);

    // create layout
    VkPipelineLayoutCreateInfo layout_ci =
    {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    };

    VkPipelineLayout layout;
    i3_vk_check(vkCreatePipelineLayout(device->handle, &layout_ci, NULL, &layout));

    return layout;
}

// create shader module
static VkShaderModule i3_vk_create_shader_module(i3_vk_device_o* device, const i3_rbk_pipeline_shader_stage_desc_t* desc)
{
    assert(device != NULL);
    assert(desc != NULL);

    VkShaderModuleCreateInfo module_ci =
    {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = desc->code_size,
        .pCode = desc->code,
    };

    VkShaderModule module;
    i3_vk_check(vkCreateShaderModule(device->handle, &module_ci, NULL, &module));
    return module;
}

// create stages
static void i3_vk_create_stages(i3_vk_device_o* device, uint32_t stage_count, const i3_rbk_pipeline_shader_stage_desc_t* descs, VkShaderModule* modules, VkPipelineShaderStageCreateInfo* stages)
{
    assert(device != NULL);
    assert(stage_count == 0 || descs != NULL);
    assert(stage_count == 0 || stages != NULL);

    for (uint32_t i = 0; i < stage_count; ++i)
    {
        // create stage
        stages[i] = (VkPipelineShaderStageCreateInfo)
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = i3_vk_convert_shader_stage(descs[i].stage),
            .module = modules[i],
            .pName = descs[i].entry_point,
        };
    }
}

// vertex input state
static void i3_vk_create_vertex_input_states(i3_arena_t* arena, const i3_rbk_pipeline_vertex_input_state_t* desc, VkPipelineVertexInputStateCreateInfo* ci)
{
    assert(arena != NULL);
    assert(desc != NULL);
    assert(ci != NULL);

    // create binding descriptions
    VkVertexInputBindingDescription* bindings = i3_arena_alloc(arena, sizeof(VkVertexInputBindingDescription) * desc->binding_count);
    for (uint32_t i = 0; i < desc->binding_count; i++)
    {
        const i3_rbk_pipeline_vertex_input_binding_desc_t* binding = &desc->bindings[i];
        bindings[i] = (VkVertexInputBindingDescription)
        {
            .binding = binding->binding,
            .stride = binding->stride,
            .inputRate = i3_vk_convert_vertex_input_rate(binding->input_rate),
        };
    }

    // create attribute descriptions
    VkVertexInputAttributeDescription* attributes = i3_arena_alloc(arena, sizeof(VkVertexInputAttributeDescription) * desc->attribute_count);
    for (uint32_t i = 0; i < desc->attribute_count; i++)
    {
        const i3_rbk_pipeline_vertex_input_attribute_desc_t* attribute = &desc->attributes[i];
        attributes[i] = (VkVertexInputAttributeDescription)
        {
            .location = attribute->location,
            .binding = attribute->binding,
            .format = i3_vk_convert_format(attribute->format),
            .offset = attribute->offset,
        };
    }

    // create vertex input state
    *ci = (VkPipelineVertexInputStateCreateInfo)
    {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = desc->binding_count,
        .pVertexBindingDescriptions = bindings,
        .vertexAttributeDescriptionCount = desc->attribute_count,
        .pVertexAttributeDescriptions = attributes,
    };
}

// input assembly state
static void i3_vk_create_input_assembly_states(i3_arena_t* arena, const i3_rbk_pipeline_input_assembly_state_t* desc, VkPipelineInputAssemblyStateCreateInfo* ci)
{
    assert(arena != NULL);
    assert(desc != NULL);
    assert(ci != NULL);
    *ci = (VkPipelineInputAssemblyStateCreateInfo)
    {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = i3_vk_convert_primitive_topology(desc->topology),
        .primitiveRestartEnable = desc->primitive_restart_enable ? VK_TRUE : VK_FALSE,
    };
}

// create graphics pipeline
i3_rbk_pipeline_i* i3_vk_device_create_graphics_pipeline(i3_rbk_device_o* self, const i3_rbk_graphics_pipeline_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;
    i3_vk_pipeline_o* pipeline = i3_vk_allocate_pipeline(device);

    // allocate arena
    i3_arena_t arena;
    i3_arena_init(&arena, 16 * I3_KB);

    // create pipeline layout
    pipeline->layout = i3_vk_create_pipeline_layout(&arena, device, &desc->layout);

    // create pipeline
    VkGraphicsPipelineCreateInfo pipeline_ci =
    {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .layout = pipeline->layout,
    };

    // create shader modules
    VkShaderModule* modules = i3_arena_alloc(&arena, sizeof(VkShaderModule) * desc->stage_count);
    for (uint32_t i = 0; i < desc->stage_count; i++)
        modules[i] = i3_vk_create_shader_module(device, &desc->stages[i]);

    // create stages
    void* stages = i3_arena_alloc(&arena, desc->stage_count * sizeof(VkPipelineShaderStageCreateInfo));
    i3_vk_create_stages(device, desc->stage_count, desc->stages, modules, stages);
    pipeline_ci.pStages = stages;
    pipeline_ci.stageCount = desc->stage_count;

    // vertex input state
    if (desc->vertex_input != NULL)
    {
        VkPipelineVertexInputStateCreateInfo* vertex_input = i3_arena_alloc(&arena, sizeof(VkPipelineVertexInputStateCreateInfo));
        i3_vk_create_vertex_input_states(&arena, desc->vertex_input, vertex_input);
        pipeline_ci.pVertexInputState = vertex_input;
    }

    // input assembly state
    if (desc->input_assembly != NULL)
    {
        VkPipelineInputAssemblyStateCreateInfo* input_assembly = i3_arena_alloc(&arena, sizeof(VkPipelineInputAssemblyStateCreateInfo));
        i3_vk_create_input_assembly_states(&arena, desc->input_assembly, input_assembly);
        pipeline_ci.pInputAssemblyState = input_assembly;
    }

    // create pipeline
    i3_vk_check(vkCreateGraphicsPipelines(device->handle, VK_NULL_HANDLE, 1, &pipeline_ci, NULL, &pipeline->handle));

    // destroy shader modules
    for (uint32_t i = 0; i < desc->stage_count; i++)
        vkDestroyShaderModule(device->handle, modules[i], NULL);

    // free arena
    i3_arena_free(&arena);

    return &pipeline->iface;
}

// create compute pipeline
i3_rbk_pipeline_i* i3_vk_device_create_compute_pipeline(i3_rbk_device_o* self, const i3_rbk_compute_pipeline_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;
    i3_vk_pipeline_o* pipeline = i3_vk_allocate_pipeline(device);

    // allocate arena
    i3_arena_t arena;
    i3_arena_init(&arena, 16 * I3_KB);

    // create pipeline layout
    pipeline->layout = i3_vk_create_pipeline_layout(&arena, device, &desc->layout);

    // create shader module
    VkShaderModule module = i3_vk_create_shader_module(device, &desc->stage);

    // create pipeline
    VkComputePipelineCreateInfo pipeline_ci =
    {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .layout = pipeline->layout,
    };

    // create stage
    i3_vk_create_stages(device, 1, &desc->stage, &module, &pipeline_ci.stage);

    // create pipeline
    i3_vk_check(vkCreateComputePipelines(device->handle, VK_NULL_HANDLE, 1, &pipeline_ci, NULL, &pipeline->handle));

    // destroy shader module
    vkDestroyShaderModule(device->handle, module, NULL);

    // free arena
    i3_arena_free(&arena);

    return &pipeline->iface;
}