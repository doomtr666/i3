#include "game.h"

#include "native/core/log.h"
#include "native/vk_backend/vk_backend.h"

typedef struct i3_game_o
{
    i3_game_i iface;
    i3_game_desc_t desc;
    i3_content_store_i* content_store;
    i3_render_backend_i* backend;
    i3_renderer_i* renderer;
    i3_render_window_i* window;
    i3_logger_i* log;
    bool is_running;
} i3_game_o;

static void i3_game_destroy(i3_game_o* self)
{
    assert(self != NULL);

    // destroy the renderer
    self->renderer->destroy(self->renderer->self);

    // destroy the render window
    self->window->destroy(self->window->self);

    // destroy the backend
    self->backend->destroy(self->backend->self);

    // destroy the content store
    self->content_store->destroy(self->content_store->self);

    i3_log_inf(self->log, "Game destroyed");

    // free the game object
    i3_free(self);
}

static void* i3_game_get_user_data(i3_game_o* self)
{
    assert(self != NULL);

    return self->desc.user_data;
}

static i3_content_store_i* i3_game_get_content_store(i3_game_o* self)
{
    assert(self != NULL);

    return self->content_store;
}

static i3_renderer_i* i3_game_get_renderer(i3_game_o* self)
{
    assert(self != NULL);

    return self->renderer;
}

static i3_render_window_i* i3_game_get_window(i3_game_o* self)
{
    assert(self != NULL);

    return self->window;
}

static void i3_game_terminate(i3_game_o* self)
{
    assert(self != NULL);
}

static void i3_game_run(i3_game_o* self)
{
    assert(self != NULL);

    if (self->desc.init)
        self->desc.init(&self->iface);

    i3_game_time_t game_time;
    i3_game_time_init(&game_time);

    while (self->is_running && !self->window->should_close(self->window->self))
    {
        // update the game time
        i3_game_time_update(&game_time);

        // update the game
        if (self->desc.update)
            self->desc.update(&self->iface, &game_time);

        // render
        self->renderer->render(self->renderer->self, &game_time);

        // process window events
        i3_render_window_poll_events();
    }

    if (self->desc.cleanup)
        self->desc.cleanup(&self->iface);
}

static i3_game_o i3_vk_game_iface_ = 
{
    .iface =
    {
        .get_user_data = i3_game_get_user_data,
        .get_content_store = i3_game_get_content_store,
        .get_renderer = i3_game_get_renderer,
        .get_window = i3_game_get_window,
        .terminate = i3_game_terminate,
        .run = i3_game_run,
        .destroy = i3_game_destroy,
    },
};

i3_game_i* i3_game_create(int argc, char** argv, i3_game_desc_t* desc)
{
    assert(desc != NULL);

    // create the game object
    i3_game_o* game = i3_alloc(sizeof(i3_game_o));
    assert(game != NULL);

    *game = i3_vk_game_iface_;
    game->iface.self = game;
    game->desc = *desc;
    game->is_running = true;

    // create the logger
    game->log = i3_get_logger(I3_GAME_LOGGER_NAME);

    // create the content store
    game->content_store = i3_content_store_create();

    // create the render backend, vulkan only for now
    game->backend = i3_vk_backend_create();

    // create the render window
    game->window = game->backend->create_render_window(game->backend->self,
                                                       game->desc.name ? game->desc.name : "I3 Game", 800, 600);

    // create the renderer
    game->renderer = i3_renderer_create(game->backend, game->window, game->content_store);

    i3_log_inf(game->log, "Game created");

    return &game->iface;
}