#include "deferred_root.h"

#include "native/math/cam.h"

typedef struct i3_deferred_root_uniform_buffer_t
{
    i3_mat4_t proj_view;  // projection view matrix
    float elapsed_time;   // elapsed time since last frame
    float total_time;     // total time since the start of the application
} i3_deferred_root_uniform_buffer_t;

typedef struct i3_deferred_root_pass_ctx_t
{
    i3_rbk_buffer_i* main_uniform_buffer;  // main uniform buffer for the deferred root pass
    i3_deferred_root_uniform_buffer_t values;
} i3_deferred_root_pass_ctx_t;

static void i3_renderer_deferred_root_pass_init(i3_render_pass_i* pass)
{
    i3_deferred_root_pass_ctx_t* ctx = i3_alloc(sizeof(i3_deferred_root_pass_ctx_t));
    *ctx = (i3_deferred_root_pass_ctx_t){0};
    pass->set_user_data(pass->self, ctx);

    // create main uniform buffer
    i3_rbk_device_i* device = pass->get_device(pass->self);
    i3_rbk_buffer_desc_t buffer_desc = {
        .flags = I3_RBK_BUFFER_FLAG_UNIFORM_BUFFER,
        .size = sizeof(i3_deferred_root_uniform_buffer_t),
    };
    ctx->main_uniform_buffer = device->create_buffer(device->self, &buffer_desc);

    // put the main uniform buffer into the render pass blackboard
    pass->put(pass->self, "u_main_constants", &ctx->main_uniform_buffer, sizeof(i3_rbk_buffer_i*));
}

static void i3_renderer_deferred_root_pass_destroy(i3_render_pass_i* pass)
{
    // get context
    i3_deferred_root_pass_ctx_t* ctx = (i3_deferred_root_pass_ctx_t*)pass->get_user_data(pass->self);

    // destroy main uniform buffer
    ctx->main_uniform_buffer->destroy(ctx->main_uniform_buffer->self);

    // free context
    i3_free(ctx);
}

static void i3_renderer_deferred_root_pass_update(i3_render_pass_i* pass)
{
    // get context
    i3_deferred_root_pass_ctx_t* ctx = (i3_deferred_root_pass_ctx_t*)pass->get_user_data(pass->self);

    // get camera
    i3_cam_t cam;
    if (!pass->get(pass->self, "cam", &cam))
        assert(false && "Deferred root pass requires i3_cam_t 'cam' to be set in the render graph.");

    // aspect ration
    uint32_t width, height;
    pass->get_render_size(pass->self, &width, &height);
    float aspect_ratio = (float)width / height;

    // game time
    i3_game_time_t* game_time = pass->get_game_time(pass->self);

    // set values
    ctx->values.proj_view = i3_cam_get_projection_view_matrix(&cam, aspect_ratio);
    ctx->values.elapsed_time = game_time->elapsed_time;
    ctx->values.total_time = game_time->total_time;

    // update the main uniform buffer
    i3_rbk_cmd_buffer_i* cmd_buffer = pass->get_cmd_buffer(pass->self);
    cmd_buffer->write_buffer(cmd_buffer->self, ctx->main_uniform_buffer, 0, sizeof(i3_deferred_root_uniform_buffer_t),
                             &ctx->values);
    pass->submit_cmd_buffers(pass->self, 1, &cmd_buffer);
    cmd_buffer->destroy(cmd_buffer->self);
}

i3_render_pass_desc_t* i3_get_deferred_root_pass_desc()
{
    static i3_render_pass_desc_t deferred_root_pass_desc = {
        .name = I3_RENDERER_DEFERRED_ROOT_PASS_NAME,
        .init = i3_renderer_deferred_root_pass_init,
        .destroy = i3_renderer_deferred_root_pass_destroy,
        .update = i3_renderer_deferred_root_pass_update,
    };

    return &deferred_root_pass_desc;
}