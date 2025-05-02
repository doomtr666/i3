#include "native/core/arena.h"

#include "convert.h"
#include "framebuffer.h"
#include "pipeline.h"
#include "pipeline_layout.h"
#include "shader_module.h"

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
        vkDestroyPipeline(pipeline->device->handle, pipeline->handle, NULL);

        // release pipeline layout
        i3_rbk_resource_release(pipeline->layout);

        // release framebuffer
        i3_rbk_resource_release(pipeline->framebuffer);

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
        VkDebugUtilsObjectNameInfoEXT name_info = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                                                   .objectType = VK_OBJECT_TYPE_PIPELINE,
                                                   .objectHandle = (uintptr_t)pipeline->handle,
                                                   .pObjectName = name};
        pipeline->device->backend->ext.vkSetDebugUtilsObjectNameEXT(pipeline->device->handle, &name_info);
    }
}

// pipeline interface

static i3_rbk_resource_i* i3_vk_pipeline_get_resource(i3_rbk_pipeline_o* self)
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
        .get_resource = i3_vk_pipeline_get_resource,
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

// create stages
static VkPipelineShaderStageCreateInfo* i3_vk_create_stages(i3_arena_t* arena,
                                                            uint32_t stage_count,
                                                            const i3_rbk_pipeline_shader_stage_desc_t* descs)
{
    assert(arena != NULL);
    assert(descs != NULL);
    assert(stage_count > 0);

    VkPipelineShaderStageCreateInfo* ci = i3_arena_alloc(arena, sizeof(VkPipelineShaderStageCreateInfo) * stage_count);

    for (uint32_t i = 0; i < stage_count; ++i)
    {
        // create stage
        ci[i] = (VkPipelineShaderStageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = i3_vk_convert_shader_stage(descs[i].stage),
            .module = ((i3_vk_shader_module_o*)descs[i].shader_module->self)->handle,
            .pName = descs[i].entry_point,
        };
    }

    return ci;
}

// vertex input state
static VkPipelineVertexInputStateCreateInfo* i3_vk_create_vertex_input_states(
    i3_arena_t* arena,
    const i3_rbk_pipeline_vertex_input_state_t* desc)
{
    assert(arena != NULL);
    assert(desc != NULL);

    VkPipelineVertexInputStateCreateInfo* ci = i3_arena_alloc(arena, sizeof(VkPipelineVertexInputStateCreateInfo));

    // create binding descriptions
    VkVertexInputBindingDescription* bindings
        = i3_arena_alloc(arena, sizeof(VkVertexInputBindingDescription) * desc->binding_count);
    for (uint32_t i = 0; i < desc->binding_count; i++)
    {
        const i3_rbk_pipeline_vertex_input_binding_desc_t* binding = &desc->bindings[i];
        bindings[i] = (VkVertexInputBindingDescription){
            .binding = binding->binding,
            .stride = binding->stride,
            .inputRate = i3_vk_convert_vertex_input_rate(binding->input_rate),
        };
    }

    // create attribute descriptions
    VkVertexInputAttributeDescription* attributes
        = i3_arena_alloc(arena, sizeof(VkVertexInputAttributeDescription) * desc->attribute_count);
    for (uint32_t i = 0; i < desc->attribute_count; i++)
    {
        const i3_rbk_pipeline_vertex_input_attribute_desc_t* attribute = &desc->attributes[i];
        attributes[i] = (VkVertexInputAttributeDescription){
            .location = attribute->location,
            .binding = attribute->binding,
            .format = i3_vk_convert_format(attribute->format),
            .offset = attribute->offset,
        };
    }

    // create vertex input state
    *ci = (VkPipelineVertexInputStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = desc->binding_count,
        .pVertexBindingDescriptions = bindings,
        .vertexAttributeDescriptionCount = desc->attribute_count,
        .pVertexAttributeDescriptions = attributes,
    };

    return ci;
}

// input assembly state
static VkPipelineInputAssemblyStateCreateInfo* i3_vk_create_input_assembly_states(
    i3_arena_t* arena,
    const i3_rbk_pipeline_input_assembly_state_t* desc)
{
    assert(arena != NULL);
    assert(desc != NULL);

    VkPipelineInputAssemblyStateCreateInfo* ci = i3_arena_alloc(arena, sizeof(VkPipelineInputAssemblyStateCreateInfo));

    *ci = (VkPipelineInputAssemblyStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = i3_vk_convert_primitive_topology(desc->topology),
        .primitiveRestartEnable = desc->primitive_restart_enable ? VK_TRUE : VK_FALSE,
    };

    return ci;
}

// tessellation state
static VkPipelineTessellationStateCreateInfo* i3_vk_create_tessellation_states(
    i3_arena_t* arena,
    const i3_rbk_pipeline_tessellation_state_t* desc)
{
    assert(arena != NULL);
    assert(desc != NULL);

    VkPipelineTessellationStateCreateInfo* ci = i3_arena_alloc(arena, sizeof(VkPipelineTessellationStateCreateInfo));

    *ci = (VkPipelineTessellationStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
        .patchControlPoints = desc->path_control_points,
    };

    return ci;
}

// viewport state
static VkPipelineViewportStateCreateInfo* i3_vk_create_viewport_states(i3_arena_t* arena,
                                                                       const i3_rbk_pipeline_viewport_state_t* desc)
{
    assert(arena != NULL);
    assert(desc != NULL);

    VkPipelineViewportStateCreateInfo* ci = i3_arena_alloc(arena, sizeof(VkPipelineViewportStateCreateInfo));

    *ci = (VkPipelineViewportStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = desc->viewport_count,
        .scissorCount = desc->scissor_count,
    };

    // create viewports
    if (desc->viewport_count > 0)
    {
        VkViewport* viewports = i3_arena_alloc(arena, sizeof(VkViewport) * desc->viewport_count);
        for (uint32_t i = 0; i < desc->viewport_count; i++)
        {
            const i3_rbk_viewport_t* viewport = &desc->viewports[i];
            viewports[i] = (VkViewport){
                .x = viewport->x,
                .y = viewport->y,
                .width = viewport->width,
                .height = viewport->height,
                .minDepth = viewport->min_depth,
                .maxDepth = viewport->max_depth,
            };
        }

        ci->pViewports = viewports;
    }

    // create scissors
    if (desc->scissor_count > 0)
    {
        VkRect2D* scissors = i3_arena_alloc(arena, sizeof(VkRect2D) * desc->scissor_count);
        for (uint32_t i = 0; i < desc->scissor_count; i++)
        {
            const i3_rbk_rect_t* rect = &desc->scissors[i];
            scissors[i] = (VkRect2D){
                .offset = {rect->offset.x, rect->offset.y},
                .extent = {rect->extent.width, rect->extent.height},
            };
        }

        ci->pScissors = scissors;
    }

    return ci;
}

// rasterization state
static VkPipelineRasterizationStateCreateInfo* i3_vk_create_rasterization_states(
    i3_arena_t* arena,
    const i3_rbk_pipeline_rasterization_state_t* desc)
{
    assert(arena != NULL);
    assert(desc != NULL);

    VkPipelineRasterizationStateCreateInfo* ci = i3_arena_alloc(arena, sizeof(VkPipelineRasterizationStateCreateInfo));

    *ci = (VkPipelineRasterizationStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = desc->depth_clamp_enable ? VK_TRUE : VK_FALSE,
        .rasterizerDiscardEnable = desc->rasterizer_discard_enable ? VK_TRUE : VK_FALSE,
        .polygonMode = i3_vk_convert_polygon_mode(desc->polygon_mode),
        .cullMode = i3_vk_convert_cull_mode_flags(desc->cull_mode),
        .frontFace = i3_vk_convert_front_face(desc->front_face),
        .depthBiasEnable = desc->depth_bias_enable ? VK_TRUE : VK_FALSE,
        .depthBiasConstantFactor = desc->depth_bias_constant_factor,
        .depthBiasClamp = desc->depth_bias_clamp,
        .depthBiasSlopeFactor = desc->depth_bias_slope_factor,
        .lineWidth = desc->line_width,
    };

    return ci;
}

// multisample state
static VkPipelineMultisampleStateCreateInfo* i3_vk_create_multisample_states(
    i3_arena_t* arena,
    const i3_rbk_pipeline_multisample_state_t* desc)
{
    assert(arena != NULL);
    assert(desc != NULL);

    VkPipelineMultisampleStateCreateInfo* ci = i3_arena_alloc(arena, sizeof(VkPipelineMultisampleStateCreateInfo));

    *ci = (VkPipelineMultisampleStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = i3_vk_convert_sample_count(desc->rasterization_samples),
        .sampleShadingEnable = desc->sample_shading_enable ? VK_TRUE : VK_FALSE,
        .minSampleShading = desc->min_sample_shading,
        .pSampleMask = desc->sample_mask,
        .alphaToCoverageEnable = desc->alpha_to_coverage_enable ? VK_TRUE : VK_FALSE,
        .alphaToOneEnable = desc->alpha_to_one_enable ? VK_TRUE : VK_FALSE,
    };

    return ci;
}

// depth stencil state
static VkPipelineDepthStencilStateCreateInfo* i3_vk_create_depth_stencil_states(
    i3_arena_t* arena,
    const i3_rbk_pipeline_depth_stencil_state_t* desc)
{
    assert(arena != NULL);
    assert(desc != NULL);

    VkPipelineDepthStencilStateCreateInfo* ci = i3_arena_alloc(arena, sizeof(VkPipelineDepthStencilStateCreateInfo));

    *ci = (VkPipelineDepthStencilStateCreateInfo)
    {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = desc->depth_test_enable ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = desc->depth_write_enable ? VK_TRUE : VK_FALSE,
        .depthCompareOp = i3_vk_convert_compare_op(desc->depth_compare_op),
        .depthBoundsTestEnable = desc->depth_bounds_test_enable ? VK_TRUE : VK_FALSE,
        .stencilTestEnable = desc->stencil_test_enable ? VK_TRUE : VK_FALSE,
        .front = (VkStencilOpState)
        {
            .failOp = i3_vk_convert_stencil_op(desc->front.fail_op),
            .passOp = i3_vk_convert_stencil_op(desc->front.pass_op),
            .depthFailOp = i3_vk_convert_stencil_op(desc->front.depth_fail_op),
            .compareOp = i3_vk_convert_compare_op(desc->front.compare_op),
            .compareMask = desc->front.compare_mask,
            .writeMask = desc->front.write_mask,
            .reference = desc->front.reference,
        },
        .back = (VkStencilOpState)
        {
            .failOp = i3_vk_convert_stencil_op(desc->back.fail_op),
            .passOp = i3_vk_convert_stencil_op(desc->back.pass_op),
            .depthFailOp = i3_vk_convert_stencil_op(desc->back.depth_fail_op),
            .compareOp = i3_vk_convert_compare_op(desc->back.compare_op),
            .compareMask = desc->back.compare_mask,
            .writeMask = desc->back.write_mask,
            .reference = desc->back.reference,
        },
        .minDepthBounds = desc->min_depth_bounds,
        .maxDepthBounds = desc->max_depth_bounds,
    };

    return ci;
}

// color blend state
static VkPipelineColorBlendStateCreateInfo* i3_vk_create_color_blend_states(
    i3_arena_t* arena,
    const i3_rbk_pipeline_color_blend_state_t* desc)
{
    assert(arena != NULL);
    assert(desc != NULL);

    VkPipelineColorBlendStateCreateInfo* ci = i3_arena_alloc(arena, sizeof(VkPipelineColorBlendStateCreateInfo));

    *ci = (VkPipelineColorBlendStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = desc->logic_op_enable ? VK_TRUE : VK_FALSE,
        .logicOp = i3_vk_convert_logic_op(desc->logic_op),
        .attachmentCount = desc->attachment_count,
        .blendConstants
        = {desc->blend_constants[0], desc->blend_constants[1], desc->blend_constants[2], desc->blend_constants[3]},
    };

    // create attachments
    if (desc->attachment_count > 0)
    {
        VkPipelineColorBlendAttachmentState* attachments
            = i3_arena_alloc(arena, sizeof(VkPipelineColorBlendAttachmentState) * desc->attachment_count);
        for (uint32_t i = 0; i < desc->attachment_count; i++)
        {
            const i3_rbk_pipeline_color_blend_attachment_state_t* attachment = &desc->attachments[i];
            attachments[i] = (VkPipelineColorBlendAttachmentState){
                .blendEnable = attachment->blend_enable ? VK_TRUE : VK_FALSE,
                .srcColorBlendFactor = i3_vk_convert_blend_factor(attachment->src_color_blend_factor),
                .dstColorBlendFactor = i3_vk_convert_blend_factor(attachment->dst_color_blend_factor),
                .colorBlendOp = i3_vk_convert_blend_op(attachment->color_blend_op),
                .srcAlphaBlendFactor = i3_vk_convert_blend_factor(attachment->src_alpha_blend_factor),
                .dstAlphaBlendFactor = i3_vk_convert_blend_factor(attachment->dst_alpha_blend_factor),
                .alphaBlendOp = i3_vk_convert_blend_op(attachment->alpha_blend_op),
                .colorWriteMask = i3_vk_convert_color_component_flags(attachment->color_write_mask),
            };
        }

        ci->pAttachments = attachments;
    }

    return ci;
}

// dynamic state
static VkPipelineDynamicStateCreateInfo* i3_vk_create_dynamic_states(i3_arena_t* arena,
                                                                     const i3_rbk_pipeline_dynamic_state_t* desc)
{
    assert(arena != NULL);
    assert(desc != NULL);

    VkPipelineDynamicStateCreateInfo* ci = i3_arena_alloc(arena, sizeof(VkPipelineDynamicStateCreateInfo));

    *ci = (VkPipelineDynamicStateCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = desc->dynamic_state_count,
    };

    // create dynamic states
    if (desc->dynamic_state_count > 0)
    {
        VkDynamicState* dynamic_states = i3_arena_alloc(arena, sizeof(VkDynamicState) * desc->dynamic_state_count);
        for (uint32_t i = 0; i < desc->dynamic_state_count; i++)
            dynamic_states[i] = i3_vk_convert_dynamic_state(desc->dynamic_states[i]);

        ci->pDynamicStates = dynamic_states;
    }

    return ci;
}

// create graphics pipeline
i3_rbk_pipeline_i* i3_vk_device_create_graphics_pipeline(i3_rbk_device_o* self,
                                                         const i3_rbk_graphics_pipeline_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;
    i3_vk_pipeline_o* pipeline = i3_vk_allocate_pipeline(device);
    pipeline->bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;

    // allocate arena
    i3_arena_t arena;
    i3_arena_init(&arena, 4 * I3_KB);

    // keep ref to framebuffer
    pipeline->framebuffer = desc->framebuffer;
    i3_rbk_resource_add_ref(desc->framebuffer);

    // keep ref to pipeline layout
    pipeline->layout = desc->layout;
    i3_rbk_resource_add_ref(desc->layout);

    // create pipeline
    VkGraphicsPipelineCreateInfo pipeline_ci = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .layout = ((i3_vk_pipeline_layout_o*)desc->layout->self)->handle,
        .renderPass = ((i3_vk_framebuffer_o*)desc->framebuffer->self)->render_pass,
    };

    // create stages
    if (desc->stage_count > 0 && desc->stages != NULL)
    {
        pipeline_ci.pStages = i3_vk_create_stages(&arena, desc->stage_count, desc->stages);
        pipeline_ci.stageCount = desc->stage_count;
    }

    // vertex input state
    if (desc->vertex_input != NULL)
        pipeline_ci.pVertexInputState = i3_vk_create_vertex_input_states(&arena, desc->vertex_input);

    // input assembly state
    if (desc->input_assembly != NULL)
        pipeline_ci.pInputAssemblyState = i3_vk_create_input_assembly_states(&arena, desc->input_assembly);

    // tessellation state
    if (desc->tessellation != NULL)
        pipeline_ci.pTessellationState = i3_vk_create_tessellation_states(&arena, desc->tessellation);

    // viewport state
    if (desc->viewport != NULL)
        pipeline_ci.pViewportState = i3_vk_create_viewport_states(&arena, desc->viewport);

    // rasterization state
    if (desc->rasterization != NULL)
        pipeline_ci.pRasterizationState = i3_vk_create_rasterization_states(&arena, desc->rasterization);

    // multisample state
    if (desc->multisample != NULL)
        pipeline_ci.pMultisampleState = i3_vk_create_multisample_states(&arena, desc->multisample);

    // depth stencil state
    if (desc->depth_stencil != NULL)
        pipeline_ci.pDepthStencilState = i3_vk_create_depth_stencil_states(&arena, desc->depth_stencil);

    // color blend state
    if (desc->color_blend != NULL)
        pipeline_ci.pColorBlendState = i3_vk_create_color_blend_states(&arena, desc->color_blend);

    // dynamic state
    if (desc->dynamic != NULL)
        pipeline_ci.pDynamicState = i3_vk_create_dynamic_states(&arena, desc->dynamic);

    // create pipeline
    i3_vk_check(vkCreateGraphicsPipelines(device->handle, VK_NULL_HANDLE, 1, &pipeline_ci, NULL, &pipeline->handle));

    // free arena
    i3_arena_destroy(&arena);

    return &pipeline->iface;
}

// create compute pipeline
i3_rbk_pipeline_i* i3_vk_device_create_compute_pipeline(i3_rbk_device_o* self,
                                                        const i3_rbk_compute_pipeline_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;
    i3_vk_pipeline_o* pipeline = i3_vk_allocate_pipeline(device);
    pipeline->bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;

    // allocate arena
    i3_arena_t arena;
    i3_arena_init(&arena, I3_KB);

    // create pipeline layout
    pipeline->layout = desc->layout;
    // i3_rbk_resource_add_ref(desc->layout);

    // create pipeline
    VkComputePipelineCreateInfo pipeline_ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .layout = ((i3_vk_pipeline_layout_o*)desc->layout->self)->handle,
    };

    // create stage
    pipeline_ci.stage = *i3_vk_create_stages(&arena, 1, &desc->stage);

    // create pipeline
    i3_vk_check(vkCreateComputePipelines(device->handle, VK_NULL_HANDLE, 1, &pipeline_ci, NULL, &pipeline->handle));

    // free arena
    i3_arena_destroy(&arena);

    return &pipeline->iface;
}