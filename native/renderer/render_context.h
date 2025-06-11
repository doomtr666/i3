#pragma once

#include "native/core/blackboard.h"
#include "native/core/time.h"
#include "native/render_backend/render_backend.h"

typedef struct i3_render_pass_o i3_render_pass_o;
typedef struct i3_render_pass_i i3_render_pass_i;

typedef struct i3_render_graph_o i3_render_graph_o;
typedef struct i3_render_graph_i i3_render_graph_i;

typedef struct i3_render_graph_builder_o i3_render_graph_builder_o;
typedef struct i3_render_graph_builder_i i3_render_graph_builder_i;

typedef struct i3_renderer_o i3_renderer_o;
typedef struct i3_renderer_i i3_renderer_i;

typedef struct i3_render_context_t
{
    i3_render_backend_i* backend;
    i3_render_window_i* window;
    i3_rbk_device_i* device;
    i3_rbk_swapchain_i* swapchain;
    i3_renderer_i* renderer;
    i3_render_graph_i* render_graph;
    uint32_t render_width, render_height;
    i3_game_time_t time;
} i3_render_context_t;