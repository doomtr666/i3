#include "native/core/log.h"
#include "native/game/game.h"

typedef struct i3_game_context_t
{
    i3_logger_i* log;
    i3_render_graph_i* render_graph;
} i3_game_context_t;

static void init(i3_game_i* game)
{
    i3_game_context_t* ctx = (i3_game_context_t*)game->get_user_data(game->self);
    i3_log_inf(ctx->log, "Game initialized");

    i3_renderer_i* renderer = game->get_renderer(game->self);

    // create the render graph
    i3_render_graph_builder_i* graph_builder = renderer->create_graph_builder(renderer->self);

    // create passes
    i3_render_pass_desc_t deffered_pass_desc = {
        .name = "Deffered",
        .user_data = NULL,
        .resolution_change = NULL,  // no resolution change handler
        .update = NULL,             // no update handler
        .render = NULL,             // no render handler
        .destroy = NULL,            // no destroy handler
    };
    graph_builder->begin_pass(graph_builder->self, &deffered_pass_desc);

    i3_render_pass_desc_t draw_cube_pass_desc = {
        .name = "DrawCube",
        .user_data = NULL,
        .resolution_change = NULL,  // no resolution change handler
        .update = NULL,             // no update handler
        .render = NULL,             // no render handler
        .destroy = NULL,            // no destroy handler
    };
    graph_builder->add_pass(graph_builder->self, &draw_cube_pass_desc);

    i3_render_pass_desc_t light_pass_desc = {
        .name = "Light",
        .user_data = NULL,
        .resolution_change = NULL,  // no resolution change handler
        .update = NULL,             // no update handler
        .render = NULL,             // no render handler
        .destroy = NULL,            // no destroy handler
    };
    graph_builder->add_pass(graph_builder->self, &light_pass_desc);

    // end the deffered pass
    graph_builder->end_pass(graph_builder->self);

    // build the render graph
    ctx->render_graph = graph_builder->build(graph_builder->self);

    // destroy the graph builder
    graph_builder->destroy(graph_builder->self);
}

static void update(i3_game_i* game, i3_game_time_t* game_time) {}
static void render(i3_game_i* game, i3_game_time_t* game_time) {}

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
        .render = render,
        .cleanup = cleanup,
    };

    i3_game_i* game = i3_game_create(&game_desc);

    game->run(game->self);

    game->destroy(game->self);

    return 0;
}