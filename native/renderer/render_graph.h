#pragma once

#include "render_context.h"

typedef struct i3_render_pass_desc_t
{
    const char* name;
    void* user_data;

    void (*init)(i3_render_pass_i* self);
    void (*destroy)(i3_render_pass_i* self);

    void (*resolution_change)(i3_render_pass_i* self);
    void (*update)(i3_render_pass_i* self);
    void (*render)(i3_render_pass_i* self);
} i3_render_pass_desc_t;

struct i3_render_pass_i
{
    i3_render_pass_o* self;

    const i3_render_pass_desc_t* (*get_desc)(i3_render_pass_o* self);

    i3_render_backend_i* (*get_backend)(i3_render_pass_o* self);
    i3_render_window_i* (*get_window)(i3_render_pass_o* self);
    i3_renderer_i* (*get_renderer)(i3_render_pass_o* self);
    void (*get_render_size)(i3_render_pass_o* self, uint32_t* width, uint32_t* height);
    i3_game_time_t* (*get_game_time)(i3_render_pass_o* self);

    void* (*get_user_data)(i3_render_pass_o* self);
    void (*set_user_data)(i3_render_pass_o* self, void* user_data);
    void (*destroy)(i3_render_pass_o* self);
};

struct i3_render_graph_i
{
    i3_render_graph_o* self;

    void (*set_render_context)(i3_render_graph_o* self, i3_render_context_t* context);

    void (*resolution_change)(i3_render_graph_o* self);
    void (*update)(i3_render_graph_o* self);
    void (*render)(i3_render_graph_o* self);
    void (*destroy)(i3_render_graph_o* self);
};

struct i3_render_graph_builder_i
{
    i3_render_graph_builder_o* self;

    void (*add_pass)(i3_render_graph_builder_o* self, i3_render_pass_desc_t* desc);
    void (*begin_pass)(i3_render_graph_builder_o* self, i3_render_pass_desc_t* desc);
    void (*end_pass)(i3_render_graph_builder_o* self);

    i3_render_graph_i* (*build)(i3_render_graph_builder_o* self);

    void (*destroy)(i3_render_graph_builder_o* self);
};

i3_render_graph_builder_i* i3_render_graph_builder_create(i3_render_backend_i* backend);