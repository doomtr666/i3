#include "opaque_pass.h"

typedef struct i3_renderer_opaque_pass_ctx_t
{
    i3_render_target_t g_depth;
    i3_render_target_t g_normal;
    i3_render_target_t g_albedo;
    i3_render_target_t g_metalic_roughness;
} opaque_pass_ctx_t;

static void i3_renderer_opaque_pass_init(i3_render_pass_i* pass)
{
    opaque_pass_ctx_t* ctx = i3_alloc(sizeof(opaque_pass_ctx_t));
    *ctx = (opaque_pass_ctx_t){0};
    pass->set_user_data(pass->self, ctx);
}

static void i3_renderer_opaque_pass_destroy(i3_render_pass_i* pass)
{
    opaque_pass_ctx_t* ctx = (opaque_pass_ctx_t*)pass->get_user_data(pass->self);
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

static void i3_renderer_opaque_pass_render(i3_render_pass_i* pass)
{
    opaque_pass_ctx_t* ctx = (opaque_pass_ctx_t*)pass->get_user_data(pass->self);
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