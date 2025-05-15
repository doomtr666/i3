#include <stdio.h>

#include "native/core/log.h"
#include "native/core/time.h"
#include "native/math/transform.h"
#include "native/vk_backend/vk_backend.h"

#include <direct.h>

// cube vertex data
static float cube_vertex_data[] = {
    -1.000000, 1.000000,  1.000000,  0.000000,  0.000000,  1.000000,  1.000000,  -1.000000, 1.000000,  0.000000,
    0.000000,  1.000000,  1.000000,  1.000000,  1.000000,  0.000000,  0.000000,  1.000000,  1.000000,  -1.000000,
    1.000000,  0.000000,  -1.000000, 0.000000,  -1.000000, -1.000000, -1.000000, 0.000000,  -1.000000, 0.000000,
    1.000000,  -1.000000, -1.000000, 0.000000,  -1.000000, 0.000000,  -1.000000, -1.000000, 1.000000,  -1.000000,
    0.000000,  0.000000,  -1.000000, 1.000000,  -1.000000, -1.000000, 0.000000,  0.000000,  -1.000000, -1.000000,
    -1.000000, -1.000000, 0.000000,  0.000000,  1.000000,  1.000000,  -1.000000, 0.000000,  0.000000,  -1.000000,
    -1.000000, -1.000000, -1.000000, 0.000000,  0.000000,  -1.000000, -1.000000, 1.000000,  -1.000000, 0.000000,
    0.000000,  -1.000000, 1.000000,  1.000000,  1.000000,  1.000000,  0.000000,  -0.000000, 1.000000,  -1.000000,
    -1.000000, 1.000000,  0.000000,  -0.000000, 1.000000,  1.000000,  -1.000000, 1.000000,  0.000000,  -0.000000,
    -1.000000, 1.000000,  1.000000,  0.000000,  1.000000,  -0.000000, 1.000000,  1.000000,  -1.000000, 0.000000,
    1.000000,  -0.000000, -1.000000, 1.000000,  -1.000000, 0.000000,  1.000000,  -0.000000, -1.000000, -1.000000,
    1.000000,  0.000000,  -0.000000, 1.000000,  -1.000000, -1.000000, 1.000000,  0.000000,  -1.000000, 0.000000,
    -1.000000, 1.000000,  1.000000,  -1.000000, 0.000000,  0.000000,  1.000000,  -1.000000, -1.000000, 0.000000,
    0.000000,  -1.000000, 1.000000,  -1.000000, 1.000000,  1.000000,  0.000000,  0.000000,  1.000000,  1.000000,
    1.000000,  0.000000,  1.000000,  -0.000000,
};

// cube index data
static uint32_t cube_index_data[] = {
    0, 1,  2, 3, 4,  5, 6, 7,  8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    0, 18, 1, 3, 19, 4, 6, 20, 7, 9, 21, 10, 12, 22, 13, 15, 23, 16,
};

int main()
{
    // set vk backend logger level to debug
    i3_logger_i* vk_logger = i3_get_logger(I3_VK_BACKEND_LOGGER_NAME);
    vk_logger->set_level(vk_logger->self, I3_LOG_LEVEL_DEBUG);

    i3_render_backend_i* backend = i3_vk_backend_create();
    if (backend->get_device_count(backend->self) == 0)
    {
        i3_log_err(vk_logger, "No Vulkan devices found");
        return -1;
    }

    // create render window
    i3_render_window_i* window = backend->create_render_window(backend->self, "Hello, Vulkan!", 800, 600);

    // create default device
    i3_rbk_device_i* device = backend->create_device(backend->self, 0);

    // create swapchain
    i3_rbk_swapchain_desc_t swapchain_desc = {
        .requested_image_count = 2,
        .srgb = false,
        .vsync = false,
    };

    i3_rbk_swapchain_i* swapchain = device->create_swapchain(device->self, window, &swapchain_desc);

    // create sampler
    i3_rbk_sampler_desc_t sampler_desc = {.mag_filter = I3_RBK_FILTER_LINEAR,
                                          .min_filter = I3_RBK_FILTER_LINEAR,
                                          .mipmap_mode = I3_RBK_SAMPLER_MIPMAP_MODE_LINEAR};

    i3_rbk_sampler_i* sampler = device->create_sampler(device->self, &sampler_desc);

    // create vertex buffer
    i3_rbk_buffer_desc_t vertex_buffer_desc = {
        .flags = I3_RBK_BUFFER_FLAG_VERTEX_BUFFER,
        .size = sizeof(cube_vertex_data),
    };

    i3_rbk_buffer_i* vertex_buffer = device->create_buffer(device->self, &vertex_buffer_desc);

    // create index buffer
    i3_rbk_buffer_desc_t index_buffer_desc = {
        .flags = I3_RBK_BUFFER_FLAG_INDEX_BUFFER,
        .size = sizeof(cube_index_data),
    };

    i3_rbk_buffer_i* index_buffer = device->create_buffer(device->self, &index_buffer_desc);

    // copy data to buffers
    i3_rbk_cmd_buffer_i* cmd_buffer = device->create_cmd_buffer(device->self);
    cmd_buffer->write_buffer(cmd_buffer->self, vertex_buffer, 0, sizeof(cube_vertex_data), cube_vertex_data);
    cmd_buffer->write_buffer(cmd_buffer->self, index_buffer, 0, sizeof(cube_index_data), cube_index_data);
    device->submit_cmd_buffers(device->self, &cmd_buffer, 1);
    cmd_buffer->destroy(cmd_buffer->self);

    // create image
    i3_rbk_image_desc_t image_desc = {
        .type = I3_RBK_IMAGE_TYPE_2D,
        .format = I3_RBK_FORMAT_R8G8B8A8_UNORM,
        .width = 800,
        .height = 600,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1,
    };

    i3_rbk_image_i* image = device->create_image(device->self, &image_desc);

    // create image view
    i3_rbk_image_view_desc_t image_view_info = {
        .type = I3_RBK_IMAGE_VIEW_TYPE_2D,
        .format = I3_RBK_FORMAT_R8G8B8A8_UNORM,
        .aspect_mask = I3_RBK_IMAGE_ASPECT_COLOR,
        .base_mip_level = 0,
        .level_count = 1,
        .base_array_layer = 0,
        .layer_count = 1,
    };

    i3_rbk_image_view_i* image_view = device->create_image_view(device->self, image, &image_view_info);

    // create framebuffer
    i3_rbk_framebuffer_attachment_desc_t framebuffer_attachment = {
        .image_view = image_view,
    };

    i3_rbk_framebuffer_desc_t framebuffer_desc = {
        .width = 800,
        .height = 600,
        .layers = 1,
        .color_attachment_count = 1,
        .color_attachments = &framebuffer_attachment,
    };

    i3_rbk_framebuffer_i* frame_buffer = device->create_framebuffer(device->self, &framebuffer_desc);

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

    // create pipeline layout
    i3_rbk_push_constant_range_t push_constant_range = {
        .stage_flags = I3_RBK_SHADER_STAGE_VERTEX,
        .offset = 0,
        .size = sizeof(float) * 16,
    };

    i3_rbk_pipeline_layout_desc_t pipeline_layout_desc = {
        //.set_layout_count = 1,
        //.set_layouts = &descriptor_set_layout,
        .push_constant_range_count = 1,
        .push_constant_ranges = &push_constant_range,
    };

    i3_rbk_pipeline_layout_i* pipeline_layout = device->create_pipeline_layout(device->self, &pipeline_layout_desc);

    // create shader module
    FILE* file = fopen("samples/vk_draw_cubes/shaders.spv", "rb");
    if (file == NULL)
    {
        i3_log_err(vk_logger, "Failed to open shader file");
        return -1;
    }
    fseek(file, 0, SEEK_END);
    uint32_t shader_data_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* shaders_data = i3_alloc(shader_data_size);
    fread(shaders_data, 1, shader_data_size, file);
    fclose(file);

    i3_rbk_shader_module_desc_t shader_desc = {.code = shaders_data, .code_size = shader_data_size};
    i3_rbk_shader_module_i* shader_module = device->create_shader_module(device->self, &shader_desc);

    i3_free(shaders_data);

    // pipeline stages
    i3_rbk_pipeline_shader_stage_desc_t stages[] = {
        {.stage = I3_RBK_SHADER_STAGE_VERTEX, .shader_module = shader_module, .entry_point = "vertexMain"},
        {.stage = I3_RBK_SHADER_STAGE_FRAGMENT, .shader_module = shader_module, .entry_point = "fragmentMain"},
    };

    // vertex input state
    i3_rbk_pipeline_vertex_input_binding_desc_t bindings[] = {
        {.binding = 0, .stride = 6 * sizeof(float), .input_rate = I3_RBK_VERTEX_INPUT_RATE_VERTEX},
    };

    i3_rbk_pipeline_vertex_input_attribute_desc_t attributes[] = {
        {.location = 0, .binding = 0, .format = I3_RBK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
        {.location = 1, .binding = 0, .format = I3_RBK_FORMAT_R32G32B32_SFLOAT, .offset = 3 * sizeof(float)},
    };

    i3_rbk_pipeline_vertex_input_state_t vertex_input = {
        .binding_count = 1,
        .bindings = bindings,
        .attribute_count = 2,
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

    // color blend state
    i3_rbk_pipeline_color_blend_attachment_state_t color_blend_attachment = {

        .blend_enable = false,
        .color_write_mask = I3_RBK_COLOR_COMPONENT_R_BIT | I3_RBK_COLOR_COMPONENT_G_BIT | I3_RBK_COLOR_COMPONENT_B_BIT
                            | I3_RBK_COLOR_COMPONENT_A_BIT,

    };

    i3_rbk_pipeline_color_blend_state_t color_blend = {
        .logic_op_enable = false,
        .attachment_count = 1,
        .attachments = &color_blend_attachment,
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

    // graphics pipeline
    i3_rbk_graphics_pipeline_desc_t pipeline_desc = {
        .stage_count = 2,
        .stages = stages,
        .vertex_input = &vertex_input,
        .input_assembly = &input_assembly,
        .viewport = &viewport,
        .rasterization = &rasterization,
        .multisample = &multisample,
        .color_blend = &color_blend,
        .dynamic = &dynamic,
        .layout = pipeline_layout,
        .framebuffer = frame_buffer,
    };

    i3_rbk_pipeline_i* pipeline = device->create_graphics_pipeline(device->self, &pipeline_desc);

    i3_game_time_t game_time;
    i3_game_time_init(&game_time);

    while (!window->should_close(window->self))
    {
        i3_game_time_update(&game_time);

        // create cmd buffer
        i3_rbk_cmd_buffer_i* cmd_buffer = device->create_cmd_buffer(device->self);

        i3_rbk_clear_color_t clear_color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}};
        cmd_buffer->clear_image(cmd_buffer->self, image_view, &clear_color);

        i3_rbk_rect_t render_area = {.offset = {0, 0}, .extent = {800, 600}};
        i3_rbk_rect_t scissor = {.offset = {0, 0}, .extent = {800, 600}};
        i3_rbk_viewport_t viewport
            = {.x = 0.0f, .y = 600.0f, .width = 800.0f, .height = -600.0f, .min_depth = 0.0f, .max_depth = 1.0f};

        cmd_buffer->bind_vertex_buffers(cmd_buffer->self, 0, 1, &vertex_buffer, NULL);
        cmd_buffer->bind_index_buffer(cmd_buffer->self, index_buffer, 0, I3_RBK_INDEX_TYPE_UINT32);

        cmd_buffer->bind_pipeline(cmd_buffer->self, pipeline);

        cmd_buffer->set_viewports(cmd_buffer->self, 0, 1, &viewport);
        cmd_buffer->set_scissors(cmd_buffer->self, 0, 1, &scissor);

        i3_mat4_t world = i3_mat4_rotation_euler(i3_vec3(game_time.total_time, 2 * game_time.total_time, 0));
        i3_mat4_t view = i3_mat4_translation(i3_vec3(0.0f, 0.0f, -10.0f));
        i3_mat4_t proj = i3_mat4_persective_fov_rh(i3_deg_to_radf(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

        i3_mat4_t wvp = i3_mat4_mult(i3_mat4_mult(world, view), proj);
        i3_mat4_t pvw = i3_mat4_mult(i3_mat4_mult(proj, view), world);

        cmd_buffer->push_constants(cmd_buffer->self, pipeline_layout, I3_RBK_SHADER_STAGE_VERTEX, 0, sizeof(wvp), &wvp);

        cmd_buffer->begin_rendering(cmd_buffer->self, frame_buffer, &render_area);

        cmd_buffer->draw_indexed(cmd_buffer->self, sizeof(cube_index_data) / sizeof(uint32_t), 1, 0, 0, 0);

        cmd_buffer->end_rendering(cmd_buffer->self);

        // submit cmd buffer
        device->submit_cmd_buffers(device->self, &cmd_buffer, 1);

        // destroy cmd buffer
        cmd_buffer->destroy(cmd_buffer->self);

        // present image view to swapchain
        device->present(device->self, swapchain, image_view);

        // perform cleanup
        device->end_frame(device->self);

        i3_render_window_poll_events();
    }

    // wait for last frame to complete
    device->wait_idle(device->self);

    printf("avg fps: %.2f\n", 1.0f / (game_time.total_time / game_time.frame_count));

    frame_buffer->destroy(frame_buffer->self);
    shader_module->destroy(shader_module->self);
    descriptor_set_layout->destroy(descriptor_set_layout->self);
    pipeline_layout->destroy(pipeline_layout->self);
    pipeline->destroy(pipeline->self);
    image_view->destroy(image_view->self);
    image->destroy(image->self);
    vertex_buffer->destroy(vertex_buffer->self);
    index_buffer->destroy(index_buffer->self);
    sampler->destroy(sampler->self);

    swapchain->destroy(swapchain->self);
    device->destroy(device->self);
    window->destroy(window->self);
    backend->destroy(backend->self);

    i3_log_dbg(vk_logger, "Application finished successfully");

    return 0;
}