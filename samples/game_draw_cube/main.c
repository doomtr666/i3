#include "native/core/log.h"
#include "native/game/game.h"
#include "native/math/mat.h"

#include <stdio.h>

typedef struct i3_game_context_t
{
    i3_logger_i* log;
    i3_renderer_i* renderer;
    i3_render_graph_i* render_graph;
} i3_game_context_t;

// draw cube pass

typedef struct draw_cube_pass_ctx_t
{
    i3_render_target_t g_depth;
    i3_render_target_t g_normal;
    i3_render_target_t g_albedo;
    i3_render_target_t g_metalic_roughness;
} draw_cube_pass_ctx_t;

static void draw_cube_pass_init(i3_render_pass_i* pass)
{
    // initialize the context
    draw_cube_pass_ctx_t* ctx = i3_alloc(sizeof(draw_cube_pass_ctx_t));
    *ctx = (draw_cube_pass_ctx_t){0};
    pass->set_user_data(pass->self, ctx);
}

static void draw_cube_pass_destroy(i3_render_pass_i* pass)
{
    // free the context
    draw_cube_pass_ctx_t* ctx = (draw_cube_pass_ctx_t*)pass->get_user_data(pass->self);
    i3_free(ctx);
}

static void draw_cube_pass_resolution_change(i3_render_pass_i* pass)
{
    draw_cube_pass_ctx_t* ctx = (draw_cube_pass_ctx_t*)pass->get_user_data(pass->self);

    // retrieve the new render targets
    pass->get(pass->self, "g_depth", &ctx->g_depth);
    pass->get(pass->self, "g_normal", &ctx->g_normal);
    pass->get(pass->self, "g_albedo", &ctx->g_albedo);
    pass->get(pass->self, "g_metalic_roughness", &ctx->g_metalic_roughness);
}

static void draw_cube_pass_update(i3_render_pass_i* pass)
{
    // Update logic for the draw cube pass
}

static void draw_cube_pass_render(i3_render_pass_i* pass)
{
    // Render logic for the draw cube pass
}

static i3_render_pass_desc_t draw_cube_pass_desc = {
    .name = "DrawCube",
    .init = draw_cube_pass_init,
    .destroy = draw_cube_pass_destroy,
    .resolution_change = draw_cube_pass_resolution_change,
    .update = draw_cube_pass_update,
    .render = draw_cube_pass_render,
};

static void init(i3_game_i* game)
{
    i3_game_context_t* ctx = (i3_game_context_t*)game->get_user_data(game->self);
    i3_log_inf(ctx->log, "Game initialized");

    ctx->renderer = game->get_renderer(game->self);

    // create the render graph
    i3_render_graph_builder_i* graph_builder = ctx->renderer->create_graph_builder(ctx->renderer->self);

    // setup default passes
    ctx->renderer->setup_default_passes(ctx->renderer->self, graph_builder);

    // add cube render pass to the predefined gbuffer_pass
    graph_builder->add_pass(graph_builder->self, "gbuffer_pass", &draw_cube_pass_desc);

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