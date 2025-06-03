#pragma once

#include "render_graph.h"

typedef struct i3_renderer_o i3_renderer_o;

typedef struct i3_renderer_i
{
    i3_renderer_o* self;

    i3_render_graph_builder_i* (*create_graph_builder)(i3_renderer_o* self);

    void (*destroy)(i3_renderer_o* self);
} i3_renderer_i;

i3_renderer_i* i3_renderer_create(i3_render_backend_i* backend);
