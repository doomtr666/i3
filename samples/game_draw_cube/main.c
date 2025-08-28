#include "native/core/log.h"
#include "native/game/game.h"
#include "native/math/cam.h"

#include "native/deferred_graph/deferred_graph.h"

typedef struct i3_game_context_t
{
    i3_logger_i* log;
    i3_renderer_i* renderer;
    i3_render_graph_i* render_graph;
    i3_scene_i* scene;
    i3_content_i* cube_model_content;
    i3_model_i* cube_model;
    i3_model_instance_i* cube_instance;
} i3_game_context_t;

static void init(i3_game_i* game)
{
    i3_game_context_t* ctx = (i3_game_context_t*)game->get_user_data(game->self);
    i3_log_inf(ctx->log, "Game initialized");

    ctx->renderer = game->get_renderer(game->self);

    // create the render graph
    i3_render_graph_builder_i* graph_builder = ctx->renderer->create_graph_builder(ctx->renderer->self);

    // setup deferred graph
    i3_setup_deferred_graph(graph_builder);

    // build the render graph
    ctx->render_graph = graph_builder->build(graph_builder->self);

    // destroy the graph builder
    graph_builder->destroy(graph_builder->self);

    // set the render graph in the renderer
    ctx->renderer->set_render_graph(ctx->renderer->self, ctx->render_graph);

    // create a scene
    ctx->scene = ctx->renderer->create_scene(ctx->renderer->self);
    ctx->renderer->set_scene(ctx->renderer->self, ctx->scene);

    // get the content store
    i3_content_store_i* content_store = game->get_content_store(game->self);

    // load the cube model
    ctx->cube_model_content = content_store->load(content_store->self, "cube.bin");

    // create the cube model
    ctx->cube_model = ctx->scene->add_model(ctx->scene->self, ctx->cube_model_content);

    // add the cube model instance to the scene
    ctx->cube_instance = ctx->scene->add_instance(ctx->scene->self, ctx->cube_model, i3_mat4_identity());

    // release the model instance
    ctx->cube_model_content->release(ctx->cube_model_content->self);
}

static void update(i3_game_i* game, i3_game_time_t* game_time)
{
    i3_game_context_t* ctx = (i3_game_context_t*)game->get_user_data(game->self);

    // TODO: update cam
    i3_cam_t cam;
    i3_cam_init_target(&cam, (i3_vec3_t){0.0f, 0.0f, -5.0f}, (i3_vec3_t){0.0f, 0.0f, 0.0f},
                       (i3_vec3_t){0.0f, 1.0f, 0.0f}, 60.0f, 0.1f, 100.0f);

    ctx->render_graph->put(ctx->render_graph->self, "cam", &cam, sizeof(i3_cam_t));
}

static void cleanup(i3_game_i* game)
{
    i3_game_context_t* ctx = (i3_game_context_t*)game->get_user_data(game->self);

    // destroy the render graph
    ctx->render_graph->destroy(ctx->render_graph->self);

    // destroy the scene
    ctx->scene->destroy(ctx->scene->self);

    i3_log_inf(ctx->log, "Game cleaned up");
}

int main(int argc, char** argv)
{
    i3_game_context_t context;
    context.log = i3_get_logger("draw_cube");

    i3_game_desc_t game_desc = {
        .user_data = &context,
        .init = init,
        .update = update,
        .cleanup = cleanup,
    };

    i3_game_i* game = i3_game_create(argc, argv, &game_desc);

    game->run(game->self);

    game->destroy(game->self);

    return 0;
}