#pragma once

#include "render_context.h"

typedef struct i3_model_o i3_model_o;
typedef struct i3_model_i i3_model_i;

typedef struct i3_mesh_t
{
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t index_count;
    uint32_t material_index;
} i3_mesh_t;

typedef struct i3_node_t
{
    i3_mat4_t transform;       // initial transform
    uint32_t mesh_offset;      // offset in the meshes array
    uint32_t mesh_count;       // number of meshes in this node
    uint32_t children_offset;  // offset in the children array
    uint32_t children_count;   // number of children in this node
} i3_node_t;

// model
struct i3_model_i
{
    i3_model_o* self;

    bool (*is_loaded)(i3_model_o* self);
    void (*upload)(i3_model_o* self, i3_rbk_cmd_buffer_i* cmd_buffer);

    i3_node_t* (*get_nodes)(i3_model_o* self);
    uint32_t (*get_node_count)(i3_model_o* self);

    i3_mesh_t* (*get_meshes)(i3_model_o* self);
    uint32_t (*get_mesh_count)(i3_model_o* self);

    uint32_t* (*get_node_children)(i3_model_o* self);
    uint32_t* (*get_node_meshes)(i3_model_o* self);

    void (*bind_buffers)(i3_model_o* self, i3_rbk_cmd_buffer_i* cmd_buffer);

    void (*destroy)(i3_model_o* self);
};

i3_model_i* i3_model_create(i3_render_context_t* context, i3_content_i* model_content);
