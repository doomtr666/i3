#pragma once

#include "render_graph.h"

typedef struct i3_renderer_o i3_renderer_o;

struct i3_renderer_i
{
    i3_renderer_o* self;

    i3_render_graph_builder_i* (*create_graph_builder)(i3_renderer_o* self);
    void (*set_render_graph)(i3_renderer_o* self, i3_render_graph_i* graph);

    void (*render)(i3_renderer_o* self, i3_game_time_t* game_time);

    void (*destroy)(i3_renderer_o* self);
};

i3_renderer_i* i3_renderer_create(i3_render_backend_i* backend, i3_render_window_i* window);
