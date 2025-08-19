#pragma once

#include "model.h"

// model instance
typedef struct i3_model_instance_o i3_model_instance_o;

typedef struct i3_model_instance_i
{
    i3_model_instance_o* self;

    void (*destroy)(i3_model_instance_o* self);

} i3_model_instance_i;

// scene
typedef struct i3_scene_o i3_scene_o;

typedef struct i3_scene_i
{
    i3_scene_o* self;

    i3_model_i* (*add_model)(i3_scene_o* self, i3_content_i* model_content);
    i3_model_instance_i* (*add_instance)(i3_scene_o* self, i3_model_i* model, i3_mat4_t transform);

    void (*update)(i3_scene_o* self, i3_rbk_cmd_buffer_i* cmd_buffer, i3_game_time_t* game_time);
    void (*destroy)(i3_scene_o* self);
} i3_scene_i;

i3_scene_i* i3_scene_create(i3_render_context_t* context);