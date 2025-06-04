#pragma once

#include "native/core/time.h"
#include "native/render_window/render_window.h"
#include "native/renderer/renderer.h"

typedef struct i3_game_o i3_game_o;
typedef struct i3_game_i i3_game_i;

#define I3_GAME_LOGGER_NAME "game"

typedef struct i3_game_desc_t
{
    const char* name;
    void* user_data;
    void (*init)(i3_game_i* game);
    void (*update)(i3_game_i* game, i3_game_time_t* game_time);
    void (*cleanup)(i3_game_i* game);

} i3_game_desc_t;

struct i3_game_i
{
    i3_game_o* self;

    // get the game user data
    void* (*get_user_data)(i3_game_o* self);

    // get the game renderer
    i3_renderer_i* (*get_renderer)(i3_game_o* self);

    // get the render window
    i3_render_window_i* (*get_window)(i3_game_o* self);

    // terminate the game
    void (*terminate)(i3_game_o* self);

    // run the game
    void (*run)(i3_game_o* self);

    // destroy the game
    void (*destroy)(i3_game_o* self);
};

i3_game_i* i3_game_create(i3_game_desc_t* desc);
