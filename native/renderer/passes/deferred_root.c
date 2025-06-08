#include "../renderer.h"

#include "deferred_root.h"

typedef struct deferred_pass_ctx_t
{
    int dummy;
} deferred_pass_ctx_t;

static void i3_renderer_deferred_root_pass_init(i3_render_pass_i* pass)
{
    deferred_pass_ctx_t* ctx = i3_alloc(sizeof(deferred_pass_ctx_t));
    *ctx = (deferred_pass_ctx_t){0};
    pass->set_user_data(pass->self, ctx);
}

static void i3_renderer_deferred_root_pass_destroy(i3_render_pass_i* pass)
{
    // get context
    deferred_pass_ctx_t* ctx = (deferred_pass_ctx_t*)pass->get_user_data(pass->self);
    i3_free(ctx);
}

static void i3_renderer_deferred_root_pass_resolution_change(i3_render_pass_i* pass)
{
    // get context
    deferred_pass_ctx_t* ctx = (deferred_pass_ctx_t*)pass->get_user_data(pass->self);
}

static void i3_renderer_deferred_root_pass_update(i3_render_pass_i* pass)
{
    // get context
    deferred_pass_ctx_t* ctx = (deferred_pass_ctx_t*)pass->get_user_data(pass->self);
}

static void i3_renderer_deferred_root_pass_render(i3_render_pass_i* pass)
{
    // get context
    deferred_pass_ctx_t* ctx = (deferred_pass_ctx_t*)pass->get_user_data(pass->self);
}

i3_render_pass_desc_t* i3_renderer_get_deferred_root_pass_desc(void)
{
    static i3_render_pass_desc_t deferred_root_pass_desc = {
        .name = I3_RENDERER_DEFERRED_ROOT_PASS_NAME,
        .init = i3_renderer_deferred_root_pass_init,
        .destroy = i3_renderer_deferred_root_pass_destroy,
        .resolution_change = i3_renderer_deferred_root_pass_resolution_change,
        .update = i3_renderer_deferred_root_pass_update,
        .render = i3_renderer_deferred_root_pass_render,
    };

    return &deferred_root_pass_desc;
}