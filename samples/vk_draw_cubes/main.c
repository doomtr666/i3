#include <stdio.h>

#include "native/vk_backend/vk_backend.h"
#include "native/core/log.h"

#include <direct.h>

int main()
{
    // set vk backend logger level to debug
    i3_logger_i* vk_logger = i3_get_logger(I3_VK_BACKEND_LOGGER_NAME);
    vk_logger->set_level(vk_logger->self, I3_LOG_LEVEL_DEBUG);

    i3_render_backend_i* backend = i3_vk_backend_create(true);
    if (backend->get_device_count(backend->self) == 0)
    {
        i3_log_err(vk_logger, "No Vulkan devices found");
        return -1;
    }

    // create render window
    i3_render_window_i* window = backend->create_render_window(backend->self, "Hello, Vulkan!", 800, 600);

    // create default device
    i3_rbk_device_i* device = backend->create_device(backend->self, 0);

    // create sampler
    i3_rbk_sampler_desc_t sampler_desc =
    {
        .mag_filter = I3_RBK_FILTER_LINEAR,
        .min_filter = I3_RBK_FILTER_LINEAR,
        .mipmap_mode = I3_RBK_SAMPLER_MIPMAP_MODE_LINEAR
    };

    i3_rbk_sampler_i* sampler = device->create_sampler(device->self, &sampler_desc);

    // create buffer
    i3_rbk_buffer_desc_t buffer_desc =
    {
        .size = 1024
    };

    i3_rbk_buffer_i* buffer = device->create_buffer(device->self, &buffer_desc);

    // create image
    i3_rbk_image_desc_t image_desc =
    {
        .type = I3_RBK_IMAGE_TYPE_2D,
        .format = I3_RBK_FORMAT_R8G8B8A8_UNORM,
        .width = 800,
        .height = 600,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1
    };

    i3_rbk_image_i* image = device->create_image(device->self, &image_desc);

    // create image view
    i3_rbk_image_view_desc_t image_view_info =
    {
        .type = I3_RBK_IMAGE_VIEW_TYPE_2D,
        .format = I3_RBK_FORMAT_R8G8B8A8_UNORM,
        .aspect_mask = I3_RBK_IMAGE_ASPECT_COLOR,
        .base_mip_level = 0,
        .level_count = 1,
        .base_array_layer = 0,
        .layer_count = 1
    };

    i3_rbk_image_view_i* image_view = device->create_image_view(device->self, image, &image_view_info);

    // create swapchain
    i3_rbk_swapchain_desc_t swapchain_desc =
    {
        .requested_image_count = 2,
        .srgb = false,
        .vsync = false,
    };

    i3_rbk_swapchain_i* swapchain = device->create_swapchain(device->self, window, &swapchain_desc);


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

    i3_rbk_shader_module_desc_t shader_desc = {
        .code = shaders_data,
        .code_size = shader_data_size
    };
    i3_rbk_shader_module_i* shader_module = device->create_shader_module(device->self, &shader_desc);
    

    // pipeline stages
    i3_rbk_pipeline_shader_stage_desc_t stages[] =
    {
        {
            .stage = I3_RBK_SHADER_STAGE_VERTEX,
            .code = shaders_data,
            .code_size = shader_data_size,
            .entry_point = "vertexMain"
        },
        {
            .stage = I3_RBK_SHADER_STAGE_FRAGMENT,
            .code = shaders_data,
            .code_size = shader_data_size,
            .entry_point = "fragmentMain"
        }
    };

    // vertex input state
    i3_rbk_pipeline_vertex_input_binding_desc_t bindings[] =
    {
        {
            .binding = 0,
            .stride = 6 * sizeof(float),
            .input_rate = I3_RBK_VERTEX_INPUT_RATE_VERTEX
        },
    };

    i3_rbk_pipeline_vertex_input_attribute_desc_t attributes[] =
    {
        {
            .location = 0,
            .binding = 0,
            .format = I3_RBK_FORMAT_R32G32B32_SFLOAT,
            .offset = 0
        },
        {
            .location = 1,
            .binding = 0,
            .format = I3_RBK_FORMAT_R32G32B32_SFLOAT,
            .offset = 3 * sizeof(float)
        }
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

    i3_rbk_graphics_pipeline_desc_t pipeline_desc = {
        .stage_count = 2,
        .stages = stages,
        .vertex_input = &vertex_input,
        .input_assembly = &input_assembly,
    };
    i3_rbk_pipeline_i* pipeline = device->create_graphics_pipeline(device->self, &pipeline_desc);


    while (!window->should_close(window->self))
    {
        i3_render_window_poll_events();
    }

    pipeline->destroy(pipeline->self);
    swapchain->destroy(swapchain->self);
    image_view->destroy(image_view->self);
    image->destroy(image->self);
    buffer->destroy(buffer->self);
    sampler->destroy(sampler->self);
    device->destroy(device->self);
    window->destroy(window->self);
    backend->destroy(backend->self);

    return 0;
}