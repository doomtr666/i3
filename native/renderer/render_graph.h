#pragma once

#include "native/core/time.h"
#include "native/render_backend/render_backend.h"

typedef struct i3_render_pass_o i3_render_pass_o;
typedef struct i3_render_pass_i i3_render_pass_i;

typedef struct i3_render_graph_o i3_render_graph_o;
typedef struct i3_render_graph_i i3_render_graph_i;

typedef struct i3_render_graph_builder_o i3_render_graph_builder_o;
typedef struct i3_render_graph_builder_i i3_render_graph_builder_i;

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

    const i3_render_pass_desc_t* (*get_desc)(i3_render_pass_i* self);
    void* (*get_user_data)(i3_render_pass_i* self);
    void (*set_user_data)(i3_render_pass_i* self, void* user_data);

    void (*destroy)(i3_render_pass_i* self);
};

struct i3_render_graph_i
{
    i3_render_graph_o* self;

    void (*resolution_change)(i3_render_graph_i* self);
    void (*update)(i3_render_graph_i* self);
    void (*render)(i3_render_graph_i* self);
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