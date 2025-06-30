#pragma once

#include "render_context.h"

// model
typedef struct i3_model_o i3_model_o;

typedef struct i3_model_i
{
    i3_model_o* self;

    void (*destroy)(i3_model_o* self);
} i3_model_i;

i3_model_i* i3_model_create(i3_render_context_t* context, i3_rbk_cmd_buffer_i* cmb_buffer, i3_content_i* model_content);

// model instance
typedef struct i3_model_instance_o i3_model_instance_o;

typedef struct i3_model_instance_i
{
    i3_model_instance_o* self;
} i3_model_instance_i;

// scene
typedef struct i3_scene_o i3_scene_o;

typedef struct i3_scene_i
{
    i3_scene_o* self;

    i3_model_instance_i* (*add_instance)(i3_scene_o* self, i3_model_i* model, i3_mat4_t transform);

} i3_scene_i;

i3_scene_i* i3_scene_create(i3_render_context_t* context);