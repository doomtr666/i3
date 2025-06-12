#include "../renderer.h"

#include "gbuffer_pass.h"

typedef struct i3_renderer_gbuffer_pass_ctx_t
{
    i3_render_target_t g_depth;              // G-buffer depth
    i3_render_target_t g_normal;             // G-buffer normal
    i3_render_target_t g_albedo;             // G-buffer albedo
    i3_render_target_t g_metalic_roughness;  // G-buffer metallic and roughness
} gbuffer_pass_ctx_t;

static void i3_renderer_gbuffer_pass_init(i3_render_pass_i* pass)
{
    // Initialize the G-buffer pass
    gbuffer_pass_ctx_t* ctx = i3_alloc(sizeof(gbuffer_pass_ctx_t));
    *ctx = (gbuffer_pass_ctx_t){0};
    pass->set_user_data(pass->self, ctx);
}

static void i3_renderer_gbuffer_pass_destroy(i3_render_pass_i* pass)
{
    gbuffer_pass_ctx_t* ctx = (gbuffer_pass_ctx_t*)pass->get_user_data(pass->self);
    i3_renderer_i* renderer = pass->get_renderer(pass->self);

    // destroy render targets
    renderer->destroy_render_target(renderer->self, &ctx->g_depth);
    renderer->destroy_render_target(renderer->self, &ctx->g_normal);
    renderer->destroy_render_target(renderer->self, &ctx->g_albedo);
    renderer->destroy_render_target(renderer->self, &ctx->g_metalic_roughness);

    i3_free(ctx);
}

static void i3_renderer_gbuffer_pass_resolution_change(i3_render_pass_i* pass)
{
    gbuffer_pass_ctx_t* ctx = (gbuffer_pass_ctx_t*)pass->get_user_data(pass->self);
    i3_renderer_i* renderer = pass->get_renderer(pass->self);

    uint32_t width, height;
    pass->get_render_size(pass->self, &width, &height);

    // create g-buffer depth
    i3_rbk_image_desc_t depth_image_desc = {
        .type = I3_RBK_IMAGE_TYPE_2D,
        .format = I3_RBK_FORMAT_D24_UNORM_S8_UINT,
        .width = width,
        .height = height,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1,
    };
    i3_rbk_image_view_desc_t depth_view_desc = {
        .type = I3_RBK_IMAGE_VIEW_TYPE_2D,
        .format = depth_image_desc.format,
        .aspect_mask = I3_RBK_IMAGE_ASPECT_DEPTH | I3_RBK_IMAGE_ASPECT_STENCIL,
        .level_count = 1,
        .layer_count = 1,
    };

    // create the g-buffer depth render target
    renderer->create_render_target(renderer->self, &ctx->g_depth, &depth_image_desc, &depth_view_desc);
    // put the g-buffer depth in the blackboard
    pass->put(pass->self, "g_depth", &ctx->g_depth, sizeof(ctx->g_depth));

    // create g-buffer normal
    i3_rbk_image_desc_t normal_image_desc = {
        .type = I3_RBK_IMAGE_TYPE_2D,
        .format = I3_RBK_FORMAT_A2R10G10B10_UNORM,
        .width = width,
        .height = height,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1,
    };

    i3_rbk_image_view_desc_t normal_view_desc = {
        .type = I3_RBK_IMAGE_VIEW_TYPE_2D,
        .format = normal_image_desc.format,
        .aspect_mask = I3_RBK_IMAGE_ASPECT_COLOR,
        .level_count = 1,
        .layer_count = 1,
    };

    // create the g-buffer normal render target
    renderer->create_render_target(renderer->self, &ctx->g_normal, &normal_image_desc, &normal_view_desc);
    // put the g-buffer normal in the blackboard
    pass->put(pass->self, "g_normal", &ctx->g_normal, sizeof(ctx->g_normal));

    // create g-buffer albedo
    i3_rbk_image_desc_t albedo_image_desc = {
        .type = I3_RBK_IMAGE_TYPE_2D,
        .format = I3_RBK_FORMAT_R8G8B8A8_UNORM,
        .width = width,
        .height = height,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1,
    };

    i3_rbk_image_view_desc_t albedo_view_desc = {
        .type = I3_RBK_IMAGE_VIEW_TYPE_2D,
        .format = albedo_image_desc.format,
        .aspect_mask = I3_RBK_IMAGE_ASPECT_COLOR,
        .level_count = 1,
        .layer_count = 1,
    };

    // create the g-buffer albedo render target
    renderer->create_render_target(renderer->self, &ctx->g_albedo, &albedo_image_desc, &albedo_view_desc);
    // put the g-buffer albedo in the blackboard
    pass->put(pass->self, "g_albedo", &ctx->g_albedo, sizeof(ctx->g_albedo));

    // create g-buffer metallic roughness
    i3_rbk_image_desc_t metallic_roughness_image_desc = {
        .type = I3_RBK_IMAGE_TYPE_2D,
        .format = I3_RBK_FORMAT_R8G8B8A8_UNORM,
        .width = width,
        .height = height,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1,
    };

    i3_rbk_image_view_desc_t metallic_roughness_view_desc = {
        .type = I3_RBK_IMAGE_VIEW_TYPE_2D,
        .format = metallic_roughness_image_desc.format,
        .aspect_mask = I3_RBK_IMAGE_ASPECT_COLOR,
        .level_count = 1,
        .layer_count = 1,
    };

    // create the g-buffer metallic roughness render target
    renderer->create_render_target(renderer->self, &ctx->g_metalic_roughness, &metallic_roughness_image_desc,
                                   &metallic_roughness_view_desc);
    // put the g-buffer metallic roughness in the blackboard
    pass->put(pass->self, "g_metalic_roughness", &ctx->g_metalic_roughness, sizeof(ctx->g_metalic_roughness));
}

static void i3_renderer_gbuffer_pass_render(i3_render_pass_i* pass)
{
    gbuffer_pass_ctx_t* ctx = (gbuffer_pass_ctx_t*)pass->get_user_data(pass->self);
    i3_renderer_i* renderer = pass->get_renderer(pass->self);

    i3_rbk_cmd_buffer_i* cmd_buffer = pass->get_cmd_buffer(pass->self);

    // clear the G-buffer depth
    i3_rbk_clear_depth_stencil_value_t clear_depth = {.depth = 1.0f, .stencil = 0};
    cmd_buffer->clear_depth_stencil_image(cmd_buffer->self, ctx->g_depth.image_view, &clear_depth);

    // clear the G-buffer color images
    i3_rbk_clear_color_value_t clear_color = {.float32 = {0.0f, 0.0f, 0.0f, 0.0f}};
    cmd_buffer->clear_color_image(cmd_buffer->self, ctx->g_normal.image_view, &clear_color);
    cmd_buffer->clear_color_image(cmd_buffer->self, ctx->g_albedo.image_view, &clear_color);
    cmd_buffer->clear_color_image(cmd_buffer->self, ctx->g_metalic_roughness.image_view, &clear_color);

    pass->submit_cmd_buffers(pass->self, 1, &cmd_buffer);
    cmd_buffer->destroy(cmd_buffer->self);
}

i3_render_pass_desc_t* i3_renderer_get_gbuffer_pass_desc(void)
{
    static i3_render_pass_desc_t gbuffer_pass_desc = {
        .name = I3_RENDERER_GBUFFER_PASS_NAME,
        .init = i3_renderer_gbuffer_pass_init,
        .destroy = i3_renderer_gbuffer_pass_destroy,
        .resolution_change = i3_renderer_gbuffer_pass_resolution_change,
        .render = i3_renderer_gbuffer_pass_render,
    };

    return &gbuffer_pass_desc;
}
