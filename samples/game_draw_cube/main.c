#include "native/core/log.h"
#include "native/game/game.h"

#include <stdio.h>

typedef struct i3_game_context_t
{
    i3_logger_i* log;
    i3_renderer_i* renderer;
    i3_render_graph_i* render_graph;
} i3_game_context_t;

// deferred pass
typedef struct deferred_pass_ctx_t
{
    i3_render_target_t g_depth;              // g-buffer depth
    i3_render_target_t g_normal;             // b-buffer normal
    i3_render_target_t g_albedo;             // g-buffer albedo
    i3_render_target_t g_metalic_roughness;  // g-buffer metallic and roughness
} deferred_pass_ctx_t;

static void deffered_pass_init(i3_render_pass_i* pass)
{
    deferred_pass_ctx_t* ctx = i3_alloc(sizeof(deferred_pass_ctx_t));
    *ctx = (deferred_pass_ctx_t){0};
    pass->set_user_data(pass->self, ctx);
}

static void deffered_pass_destroy(i3_render_pass_i* pass)
{
    // get context
    deferred_pass_ctx_t* ctx = (deferred_pass_ctx_t*)pass->get_user_data(pass->self);
    // get renderer
    i3_renderer_i* renderer = pass->get_renderer(pass->self);

    // destroy render targets
    renderer->destroy_render_target(renderer->self, &ctx->g_depth);

    i3_free(ctx);
}

static void deffered_pass_resolution_change(i3_render_pass_i* pass)
{
    // get context
    deferred_pass_ctx_t* ctx = (deferred_pass_ctx_t*)pass->get_user_data(pass->self);

    // get renderer
    i3_renderer_i* renderer = pass->get_renderer(pass->self);

    uint32_t width, height;
    pass->get_render_size(pass->self, &width, &height);

    // create g-buffer depth
    i3_rbk_image_desc_t depth_image_desc = {
        .flags = I3_RBK_IMAGE_FLAG_NONE,
        .type = I3_RBK_IMAGE_TYPE_2D,
        .format = I3_RBK_FORMAT_D24_UNORM_S8_UINT,
        .width = width,
        .height = height,
        .depth = 1,
        .mip_levels = 1,
        .array_layers = 1,
        .samples = 1,
    };

    // create g-buffer depth view
    i3_rbk_image_view_desc_t depth_view_desc = {
        .type = I3_RBK_IMAGE_VIEW_TYPE_2D,
        .format = depth_image_desc.format,
        .aspect_mask = I3_RBK_IMAGE_ASPECT_DEPTH,
        .level_count = 1,
        .layer_count = 1,
    };

    // create the g-buffer depth render target
    renderer->create_render_target(renderer->self, &ctx->g_depth, &depth_image_desc, &depth_view_desc);
    // put the g-buffer depth in the pass blackboard
    pass->put(pass->self, "g_depth", &ctx->g_depth, sizeof(ctx->g_depth));
}

static void deffered_pass_update(i3_render_pass_i* pass)
{
    // get context
    deferred_pass_ctx_t* ctx = (deferred_pass_ctx_t*)pass->get_user_data(pass->self);
}

static void deffered_pass_render(i3_render_pass_i* pass)
{
    // get context
    deferred_pass_ctx_t* ctx = (deferred_pass_ctx_t*)pass->get_user_data(pass->self);
}

static i3_render_pass_desc_t deffered_pass_desc = {
    .name = "Deffered",
    .init = deffered_pass_init,
    .destroy = deffered_pass_destroy,
    .resolution_change = deffered_pass_resolution_change,
    .update = deffered_pass_update,
    .render = deffered_pass_render,
};

// draw cube pass
static void draw_cube_pass_init(i3_render_pass_i* pass)
{
    // Initialize the draw cube pass
}

static void draw_cube_pass_destroy(i3_render_pass_i* pass)
{
    // Cleanup the draw cube pass
}

static void draw_cube_pass_resolution_change(i3_render_pass_i* pass)
{
    // Handle resolution change for the draw cube pass
}

static void draw_cube_pass_update(i3_render_pass_i* pass)
{
    // Update logic for the draw cube pass
    // printf("Draw Cube pass updated\n");
}

static void draw_cube_pass_render(i3_render_pass_i* pass)
{
    // Render logic for the draw cube pass
    // printf("Draw Cube pass rendered\n");
}

static i3_render_pass_desc_t draw_cube_pass_desc = {
    .name = "DrawCube",
    .init = draw_cube_pass_init,
    .destroy = draw_cube_pass_destroy,
    .resolution_change = draw_cube_pass_resolution_change,
    .update = draw_cube_pass_update,
    .render = draw_cube_pass_render,
};

// lighting pass
static void light_pass_init(i3_render_pass_i* pass)
{
    // Initialize the lighting pass
}

static void light_pass_destroy(i3_render_pass_i* pass)
{
    // Cleanup the lighting pass
}

static void light_pass_resolution_change(i3_render_pass_i* pass)
{
    // Handle resolution change for the lighting pass
}

static void light_pass_update(i3_render_pass_i* pass)
{
    // Update logic for the lighting pass
}

static void light_pass_render(i3_render_pass_i* pass)
{
    // Render logic for the lighting pass
}

static i3_render_pass_desc_t light_pass_desc = {
    .name = "Light",
    .init = light_pass_init,
    .destroy = light_pass_destroy,
    .resolution_change = light_pass_resolution_change,
    .update = light_pass_update,
    .render = light_pass_render,
};

static void init(i3_game_i* game)
{
    i3_game_context_t* ctx = (i3_game_context_t*)game->get_user_data(game->self);
    i3_log_inf(ctx->log, "Game initialized");

    ctx->renderer = game->get_renderer(game->self);

    // create the render graph
    i3_render_graph_builder_i* graph_builder = ctx->renderer->create_graph_builder(ctx->renderer->self);

    // create passes
    graph_builder->begin_pass(graph_builder->self, NULL, &deffered_pass_desc);
    graph_builder->add_pass(graph_builder->self, NULL, &draw_cube_pass_desc);
    graph_builder->add_pass(graph_builder->self, NULL, &light_pass_desc);
    graph_builder->end_pass(graph_builder->self);

    // build the render graph
    ctx->render_graph = graph_builder->build(graph_builder->self);

    // destroy the graph builder
    graph_builder->destroy(graph_builder->self);

    // set the render graph in the renderer
    ctx->renderer->set_render_graph(ctx->renderer->self, ctx->render_graph);
}

static void update(i3_game_i* game, i3_game_time_t* game_time) {}

static void cleanup(i3_game_i* game)
{
    i3_game_context_t* ctx = (i3_game_context_t*)game->get_user_data(game->self);

    // destroy the render graph
    ctx->render_graph->destroy(ctx->render_graph->self);

    i3_log_inf(ctx->log, "Game cleaned up");
}

int main()
{
    i3_game_context_t context;
    context.log = i3_get_logger("draw_cube");

    i3_game_desc_t game_desc = {
        .user_data = &context,
        .init = init,
        .update = update,
        .cleanup = cleanup,
    };

    i3_game_i* game = i3_game_create(&game_desc);

    game->run(game->self);

    game->destroy(game->self);

    return 0;
}