#pragma once

#include "render_context.h"

typedef struct i3_render_pass_desc_t
{
    const char* name;
    void* user_data;

    void (*init)(i3_render_pass_i* pass);
    void (*destroy)(i3_render_pass_i* pass);

    void (*resolution_change)(i3_render_pass_i* pass);
    void (*update)(i3_render_pass_i* pass);
    void (*render)(i3_render_pass_i* pass);
} i3_render_pass_desc_t;

struct i3_render_pass_i
{
    i3_render_pass_o* self;

    const i3_render_pass_desc_t* (*get_desc)(i3_render_pass_o* self);
    i3_render_backend_i* (*get_backend)(i3_render_pass_o* self);
    i3_render_window_i* (*get_window)(i3_render_pass_o* self);
    i3_renderer_i* (*get_renderer)(i3_render_pass_o* self);
    i3_rbk_device_i* (*get_device)(i3_render_pass_o* self);
    i3_content_store_i* (*get_content_store)(i3_render_pass_o* self);
    void (*get_render_size)(i3_render_pass_o* self, uint32_t* width, uint32_t* height);
    i3_game_time_t* (*get_game_time)(i3_render_pass_o* self);

    // cmd buffers
    i3_rbk_cmd_buffer_i* (*get_cmd_buffer)(i3_render_pass_o* self);
    void (*submit_cmd_buffers)(i3_render_pass_o* self, uint32_t cmd_buffer_count, i3_rbk_cmd_buffer_i** cmd_buffers);

    void* (*get_user_data)(i3_render_pass_o* self);
    void (*set_user_data)(i3_render_pass_o* self, void* user_data);

    // blackboard
    bool (*put)(i3_render_pass_o* self, const char* key, void* data, uint32_t size);
    bool (*get)(i3_render_pass_o* self, const char* key, void* data);

    void (*destroy)(i3_render_pass_o* self);
};

struct i3_render_graph_i
{
    i3_render_graph_o* self;

    void (*resolution_change)(i3_render_graph_o* self);
    void (*update)(i3_render_graph_o* self);
    void (*render)(i3_render_graph_o* self);

    // blackboard
    bool (*put)(i3_render_graph_o* self, const char* key, void* data, uint32_t size);
    bool (*get)(i3_render_graph_o* self, const char* key, void* data);

    void (*destroy)(i3_render_graph_o* self);
};

struct i3_render_graph_builder_i
{
    i3_render_graph_builder_o* self;

    void (*add_pass)(i3_render_graph_builder_o* self, const char* parent_name, i3_render_pass_desc_t* desc);
    void (*begin_pass)(i3_render_graph_builder_o* self, const char* parent_name, i3_render_pass_desc_t* desc);
    void (*end_pass)(i3_render_graph_builder_o* self);

    i3_render_graph_i* (*build)(i3_render_graph_builder_o* self);

    void (*destroy)(i3_render_graph_builder_o* self);
};

i3_render_graph_builder_i* i3_render_graph_builder_create(i3_render_context_t* context);