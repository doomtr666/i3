#include "opaque_pass.h"

typedef struct i3_renderer_opaque_pass_ctx_t
{
    i3_render_target_t g_depth;
    i3_render_target_t g_normal;
    i3_render_target_t g_albedo;
    i3_render_target_t g_metalic_roughness;
    i3_rbk_buffer_i* u_main_constants;

    i3_rbk_pipeline_i* opaque_pipeline;
    i3_rbk_framebuffer_i* framebuffer;
    i3_rbk_descriptor_set_i* descriptor_set;
} opaque_pass_ctx_t;

static void i3_renderer_opaque_pass_init(i3_render_pass_i* pass)
{
    opaque_pass_ctx_t* ctx = i3_alloc(sizeof(opaque_pass_ctx_t));
    *ctx = (opaque_pass_ctx_t){0};
    pass->set_user_data(pass->self, ctx);

    i3_rbk_device_i* device = pass->get_device(pass->self);

    // get main uniform buffer
    pass->get(pass->self, "u_main_constants", &ctx->u_main_constants);

    // create descriptor set layout
    i3_rbk_descriptor_set_layout_binding_t descriptor_set_layout_bindings[] = {{
        .binding = 0,
        .descriptor_type = I3_RBK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptor_count = 1,
        .stage_flags = I3_RBK_SHADER_STAGE_VERTEX,
    }};

    i3_rbk_descriptor_set_layout_desc_t descriptor_set_layout_desc = {
        .binding_count = sizeof(descriptor_set_layout_bindings) / sizeof(descriptor_set_layout_bindings[0]),
        .bindings = descriptor_set_layout_bindings,
    };

    i3_rbk_descriptor_set_layout_i* descriptor_set_layout
        = device->create_descriptor_set_layout(device->self, &descriptor_set_layout_desc);

    // create descriptor set
    ctx->descriptor_set = device->create_descriptor_set(device->self, descriptor_set_layout);

    // update descriptor set
    i3_rbk_descriptor_set_write_t descriptor_set_write = {
        .binding = 0,
        .array_element = 0,
        .descriptor_type = I3_RBK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .buffer = ctx->u_main_constants,
    };
    ctx->descriptor_set->update(ctx->descriptor_set->self, 1, &descriptor_set_write);

    // create pipeline layout
    i3_rbk_push_constant_range_t push_constant_range = {
        .stage_flags = I3_RBK_SHADER_STAGE_VERTEX,
        .offset = 0,
        .size = sizeof(float) * 16,
    };

    i3_rbk_pipeline_layout_desc_t pipeline_layout_desc = {
        .set_layout_count = 1,
        .set_layouts = &descriptor_set_layout,
        .push_constant_range_count = 1,
        .push_constant_ranges = &push_constant_range,
    };

    i3_rbk_pipeline_layout_i* pipeline_layout = device->create_pipeline_layout(device->self, &pipeline_layout_desc);

    // create shader module
    i3_content_store_i* content_store = pass->get_content_store(pass->self);
    i3_content_i* shader_content = content_store->load(content_store->self, "opaque_pass.spv");

    i3_rbk_shader_module_desc_t shader_desc = {.code = shader_content->get_data(shader_content->self),
                                               .code_size = shader_content->get_size(shader_content->self)};
    i3_rbk_shader_module_i* shader_module = device->create_shader_module(device->self, &shader_desc);
    shader_content->release(shader_content->self);

    // pipeline stages
    i3_rbk_pipeline_shader_stage_desc_t stages[] = {
        {.stage = I3_RBK_SHADER_STAGE_VERTEX, .shader_module = shader_module, .entry_point = "vertexMain"},
        {.stage = I3_RBK_SHADER_STAGE_FRAGMENT, .shader_module = shader_module, .entry_point = "fragmentMain"},
    };

    // vertex input state
    i3_rbk_pipeline_vertex_input_binding_desc_t bindings[] = {
        {.binding = 0, .stride = 3 * sizeof(float), .input_rate = I3_RBK_VERTEX_INPUT_RATE_VERTEX},
        {.binding = 1, .stride = 3 * sizeof(float), .input_rate = I3_RBK_VERTEX_INPUT_RATE_VERTEX},
        {.binding = 2, .stride = 3 * sizeof(float), .input_rate = I3_RBK_VERTEX_INPUT_RATE_VERTEX},
        {.binding = 3, .stride = 3 * sizeof(float), .input_rate = I3_RBK_VERTEX_INPUT_RATE_VERTEX},
        {.binding = 4, .stride = 2 * sizeof(float), .input_rate = I3_RBK_VERTEX_INPUT_RATE_VERTEX},
    };

    i3_rbk_pipeline_vertex_input_attribute_desc_t attributes[] = {
        {.location = 0, .binding = 0, .format = I3_RBK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
        {.location = 1, .binding = 1, .format = I3_RBK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
        {.location = 2, .binding = 2, .format = I3_RBK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
        {.location = 3, .binding = 3, .format = I3_RBK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
        {.location = 4, .binding = 4, .format = I3_RBK_FORMAT_R32G32_SFLOAT, .offset = 0},
    };

    i3_rbk_pipeline_vertex_input_state_t vertex_input = {
        .binding_count = sizeof(bindings) / sizeof(bindings[0]),
        .bindings = bindings,
        .attribute_count = sizeof(attributes) / sizeof(attributes[0]),
        .attributes = attributes,
    };

    // input assembly state
    i3_rbk_pipeline_input_assembly_state_t input_assembly = {
        .topology = I3_RBK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    // viewport state
    i3_rbk_viewport_t viewport_data
        = {.x = 0.0f, .y = 0.0f, .width = 800.0f, .height = 600.0f, .min_depth = 0.0f, .max_depth = 1.0f};

    i3_rbk_rect_t scissor_data = {.offset = {0, 0}, .extent = {800, 600}};

    i3_rbk_pipeline_viewport_state_t viewport
        = {.viewport_count = 1, .viewports = &viewport_data, .scissor_count = 1, .scissors = &scissor_data};

    // rasterization state
    i3_rbk_pipeline_rasterization_state_t rasterization = {
        .polygon_mode = I3_RBK_POLYGON_MODE_FILL,
        .cull_mode = I3_RBK_CULL_MODE_BACK_BIT,
        .front_face = I3_RBK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depth_clamp_enable = false,
        .rasterizer_discard_enable = false,
        .depth_bias_enable = false,
        .line_width = 1.0f,
    };

    // multisample state
    i3_rbk_pipeline_multisample_state_t multisample = {
        .rasterization_samples = 1,
        .sample_shading_enable = false,
        .min_sample_shading = 0.0f,
        .alpha_to_coverage_enable = false,
        .alpha_to_one_enable = false,
    };

    // depth stencil state
    i3_rbk_pipeline_depth_stencil_state_t depth_stencil = {
        .depth_test_enable = true,
        .depth_write_enable = true,
        .depth_compare_op = I3_RBK_COMPARE_OP_LESS,
    };

    // color blend state
    i3_rbk_pipeline_color_blend_attachment_state_t color_blend_attachments[] = {
        {
            .blend_enable = false,
            .color_write_mask = I3_RBK_COLOR_COMPONENT_R_BIT | I3_RBK_COLOR_COMPONENT_G_BIT
                                | I3_RBK_COLOR_COMPONENT_B_BIT | I3_RBK_COLOR_COMPONENT_A_BIT,
        },
        {
            .blend_enable = false,
            .color_write_mask = I3_RBK_COLOR_COMPONENT_R_BIT | I3_RBK_COLOR_COMPONENT_G_BIT
                                | I3_RBK_COLOR_COMPONENT_B_BIT | I3_RBK_COLOR_COMPONENT_A_BIT,
        },
        {

            .blend_enable = false,
            .color_write_mask = I3_RBK_COLOR_COMPONENT_R_BIT | I3_RBK_COLOR_COMPONENT_G_BIT
                                | I3_RBK_COLOR_COMPONENT_B_BIT | I3_RBK_COLOR_COMPONENT_A_BIT,
        },
    };

    i3_rbk_pipeline_color_blend_state_t color_blend = {
        .logic_op_enable = false,
        .attachment_count = 3,
        .attachments = color_blend_attachments,
        .blend_constants = {0.0f, 0.0f, 0.0f, 0.0f},
    };

    // dynamic state
    i3_rbk_dynamic_state_t dynamic_states[] = {
        I3_RBK_DYNAMIC_STATE_VIEWPORT,
        I3_RBK_DYNAMIC_STATE_SCISSOR,
    };

    i3_rbk_pipeline_dynamic_state_t dynamic = {
        .dynamic_state_count = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
        .dynamic_states = dynamic_states,
    };

    // create attachment description
    i3_rbk_attachment_desc_t color_attachments[] = {
        {.format = I3_RBK_FORMAT_A2R10G10B10_UNORM, .samples = 1},
        {.format = I3_RBK_FORMAT_R8G8B8A8_UNORM, .samples = 1},
        {.format = I3_RBK_FORMAT_R8G8B8A8_UNORM, .samples = 1},
    };

    i3_rbk_attachment_desc_t depth_attachment = {.format = I3_RBK_FORMAT_D24_UNORM_S8_UINT, .samples = 1};

    // graphics pipeline
    i3_rbk_graphics_pipeline_desc_t pipeline_desc = {
        .color_attachment_count = 3,
        .color_attachments = color_attachments,
        .depth_stencil_attachment = &depth_attachment,
        .stage_count = 2,
        .stages = stages,
        .vertex_input = &vertex_input,
        .input_assembly = &input_assembly,
        .viewport = &viewport,
        .rasterization = &rasterization,
        .multisample = &multisample,
        .depth_stencil = &depth_stencil,
        .color_blend = &color_blend,
        .dynamic = &dynamic,
        .layout = pipeline_layout,
    };

    ctx->opaque_pipeline = device->create_graphics_pipeline(device->self, &pipeline_desc);

    // cleanup
    descriptor_set_layout->destroy(descriptor_set_layout->self);
    pipeline_layout->destroy(pipeline_layout->self);
    shader_module->destroy(shader_module->self);
}

static void i3_renderer_opaque_pass_destroy(i3_render_pass_i* pass)
{
    opaque_pass_ctx_t* ctx = (opaque_pass_ctx_t*)pass->get_user_data(pass->self);

    ctx->opaque_pipeline->destroy(ctx->opaque_pipeline->self);
    ctx->framebuffer->destroy(ctx->framebuffer->self);
    ctx->descriptor_set->destroy(ctx->descriptor_set->self);

    i3_free(ctx);
}

static void i3_renderer_opaque_pass_resolution_change(i3_render_pass_i* pass)
{
    opaque_pass_ctx_t* ctx = (opaque_pass_ctx_t*)pass->get_user_data(pass->self);

    // retrieve the render targets
    pass->get(pass->self, "g_depth", &ctx->g_depth);
    pass->get(pass->self, "g_normal", &ctx->g_normal);
    pass->get(pass->self, "g_albedo", &ctx->g_albedo);
    pass->get(pass->self, "g_metalic_roughness", &ctx->g_metalic_roughness);

    // get render size
    uint32_t width, height;
    pass->get_render_size(pass->self, &width, &height);

    i3_renderer_i* renderer = pass->get_renderer(pass->self);

    // color attachments
    i3_rbk_image_view_i* color_attachments[] = {
        ctx->g_normal.image_view,
        ctx->g_albedo.image_view,
        ctx->g_metalic_roughness.image_view,
    };

    // create framebuffer
    i3_rbk_framebuffer_desc_t framebuffer_desc = {
        .width = width,
        .height = height,
        .layers = 1,
        .color_attachment_count = 3,
        .color_attachments = color_attachments,
        .depth_stencil_attachment = ctx->g_depth.image_view,
        .graphics_pipeline = ctx->opaque_pipeline,
    };

    renderer->create_framebuffer(renderer->self, &ctx->framebuffer, &framebuffer_desc);
}

static void i3_renderer_opaque_pass_update(i3_render_pass_i* pass)
{
    opaque_pass_ctx_t* ctx = (opaque_pass_ctx_t*)pass->get_user_data(pass->self);

    // get the scene
    i3_renderer_i* renderer = pass->get_renderer(pass->self);
    i3_scene_i* scene = renderer->get_scene(renderer->self);

    // update the scene
    i3_rbk_cmd_buffer_i* cmd_buffer = pass->get_cmd_buffer(pass->self);
    i3_game_time_t* game_time = pass->get_game_time(pass->self);
    scene->update(scene->self, cmd_buffer, game_time);
    pass->submit_cmd_buffers(pass->self, 1, &cmd_buffer);
    cmd_buffer->destroy(cmd_buffer->self);
}

static void i3_renderer_opaque_pass_instance_render(void* ctx,
                                                    i3_rbk_cmd_buffer_i* cmd_buffer,
                                                    i3_model_instance_i* instance)
{
    opaque_pass_ctx_t* opaque_ctx = (opaque_pass_ctx_t*)ctx;

    // bind vertex buffers
    i3_model_i* model = instance->get_model(instance->self);
    model->bind_buffers(model->self, cmd_buffer);

    // draw meshes
    i3_node_t* nodes = model->get_nodes(model->self);
    i3_mesh_t* meshes = model->get_meshes(model->self);
    i3_mat4_t* transforms = instance->get_transforms(instance->self);

    for (uint32_t i = 0; i < model->get_node_count(model->self); i++)
    {
        // transform as a push constant
        cmd_buffer->push_constants(cmd_buffer->self,
                                   opaque_ctx->opaque_pipeline->get_layout(opaque_ctx->opaque_pipeline->self),
                                   I3_RBK_SHADER_STAGE_VERTEX, 0, sizeof(i3_mat4_t), &transforms[i]);

        i3_node_t* node = &nodes[i];
        for (uint32_t j = 0; j < node->mesh_count; j++)
        {
            i3_mesh_t* mesh = &meshes[node->mesh_offset + j];

            cmd_buffer->draw_indexed(cmd_buffer->self, mesh->index_count, 1, mesh->index_offset, mesh->vertex_offset,
                                     0);
        }
    }
}

static void i3_renderer_opaque_pass_render(i3_render_pass_i* pass)
{
    opaque_pass_ctx_t* ctx = (opaque_pass_ctx_t*)pass->get_user_data(pass->self);

    uint32_t width, height;
    pass->get_render_size(pass->self, &width, &height);

    // render the scene
    i3_renderer_i* renderer = pass->get_renderer(pass->self);
    i3_scene_i* scene = renderer->get_scene(renderer->self);

    i3_rbk_cmd_buffer_i* cmd_buffer = pass->get_cmd_buffer(pass->self);

    cmd_buffer->begin_rendering(cmd_buffer->self, ctx->framebuffer,
                                &(i3_rbk_rect_t){.offset = {0, 0}, .extent = {width, height}});

    cmd_buffer->bind_pipeline(cmd_buffer->self, ctx->opaque_pipeline);
    cmd_buffer->bind_descriptor_sets(cmd_buffer->self, ctx->opaque_pipeline, 0, 1, &ctx->descriptor_set);
    cmd_buffer->set_viewports(
        cmd_buffer->self, 0, 1,
        &(i3_rbk_viewport_t){
            .x = 0.0f, .y = 0.0f, .width = width, .height = height, .min_depth = 0.0f, .max_depth = 1.0f});
    cmd_buffer->set_scissors(cmd_buffer->self, 0, 1, &(i3_rbk_rect_t){.offset = {0, 0}, .extent = {width, height}});

    scene->render(scene->self, cmd_buffer, ctx, i3_renderer_opaque_pass_instance_render);

    cmd_buffer->end_rendering(cmd_buffer->self);

    pass->submit_cmd_buffers(pass->self, 1, &cmd_buffer);
    cmd_buffer->destroy(cmd_buffer->self);
}

i3_render_pass_desc_t* i3_get_opaque_pass_desc(void)
{
    static i3_render_pass_desc_t opaque_pass_desc = {
        .name = I3_RENDERER_OPAQUE_PASS_NAME,
        .init = i3_renderer_opaque_pass_init,
        .destroy = i3_renderer_opaque_pass_destroy,
        .resolution_change = i3_renderer_opaque_pass_resolution_change,
        .update = i3_renderer_opaque_pass_update,
        .render = i3_renderer_opaque_pass_render,
    };

    return &opaque_pass_desc;
}