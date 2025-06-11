#include "../renderer.h"

#include "light_pass.h"

typedef struct i3_renderer_light_pass_ctx_t
{
    i3_render_target_t light_buffer;  // Light buffer for the light pass
} light_pass_ctx_t;

static void i3_renderer_light_pass_init(i3_render_pass_i* pass)
{
    // Initialize the light pass
    light_pass_ctx_t* ctx = i3_alloc(sizeof(light_pass_ctx_t));
    *ctx = (light_pass_ctx_t){0};
    pass->set_user_data(pass->self, ctx);
}

static void i3_renderer_light_pass_destroy(i3_render_pass_i* pass)
{
    // get context
    light_pass_ctx_t* ctx = (light_pass_ctx_t*)pass->get_user_data(pass->self);
    // get renderer
    i3_renderer_i* renderer = pass->get_renderer(pass->self);

    // destroy the light buffer render target
    renderer->destroy_render_target(renderer->self, &ctx->light_buffer);

    i3_free(ctx);
}

static void i3_renderer_light_pass_resolution_change(i3_render_pass_i* pass)
{
    // get context
    light_pass_ctx_t* ctx = (light_pass_ctx_t*)pass->get_user_data(pass->self);

    // get renderer
    i3_renderer_i* renderer = pass->get_renderer(pass->self);

    uint32_t width, height;
    pass->get_render_size(pass->self, &width, &height);

    // create light buffer
    i3_rbk_image_desc_t light_buffer_image_desc = {
        .type = I3_RBK_IMAGE_TYPE_2D,
        .format = I3_RBK_FORMAT_R16G16B16A16_FLOAT,
        .width = width,
        .height = height,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1,
    };

    i3_rbk_image_view_desc_t light_buffer_view_desc = {
        .type = I3_RBK_IMAGE_VIEW_TYPE_2D,
        .format = light_buffer_image_desc.format,
        .aspect_mask = I3_RBK_IMAGE_ASPECT_COLOR,
        .level_count = 1,
        .layer_count = 1,
    };

    // create the light buffer render target
    renderer->create_render_target(renderer->self, &ctx->light_buffer, &light_buffer_image_desc,
                                   &light_buffer_view_desc);
    // put the light buffer in the blackboard
    pass->put(pass->self, "light_buffer", &ctx->light_buffer, sizeof(ctx->light_buffer));

    // set graph output
    pass->put(pass->self, "output", &ctx->light_buffer, sizeof(ctx->light_buffer));
}

static void i3_renderer_light_pass_update(i3_render_pass_i* pass)
{
    // get context
    light_pass_ctx_t* ctx = (light_pass_ctx_t*)pass->get_user_data(pass->self);
}

static void i3_renderer_light_pass_render(i3_render_pass_i* pass)
{
    // get context
    light_pass_ctx_t* ctx = (light_pass_ctx_t*)pass->get_user_data(pass->self);

    i3_rbk_cmd_buffer_i* cmd_buffer = pass->get_cmd_buffer(pass->self);

    i3_rbk_clear_color_t clear_color = {
        .float32 = {0.0f, 0.0f, 1.0f, 1.0f},
    };

    cmd_buffer->clear_image(cmd_buffer->self, ctx->light_buffer.image_view, &clear_color);

    pass->submit_cmd_buffers(pass->self, 1, &cmd_buffer);
    cmd_buffer->destroy(cmd_buffer->self);
}

i3_render_pass_desc_t* i3_renderer_get_light_pass_desc(void)
{
    static i3_render_pass_desc_t light_pass_desc = {
        .name = I3_RENDERER_LIGHT_PASS_NAME,
        .init = i3_renderer_light_pass_init,
        .destroy = i3_renderer_light_pass_destroy,
        .resolution_change = i3_renderer_light_pass_resolution_change,
        .update = i3_renderer_light_pass_update,
        .render = i3_renderer_light_pass_render,
    };

    return &light_pass_desc;
}
