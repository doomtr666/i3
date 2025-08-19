#pragma once

#include "render_context.h"

typedef struct i3_model_o i3_model_o;
typedef struct i3_model_i i3_model_i;

struct i3_mesh_t
{
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t index_count;
    uint32_t material_index;
};

struct i3_node_t
{
    uint32_t mesh_offset;      // offset in the meshes array
    uint32_t mesh_count;       // number of meshes in this node
    uint32_t children_offset;  // offset in the children array
    uint32_t children_count;   // number of children in this node
};

// model
struct i3_model_i
{
    i3_model_o* self;

    bool (*is_loaded)(i3_model_o* self);
    void (*upload)(i3_model_o* self, i3_rbk_cmd_buffer_i* cmd_buffer);

    void (*destroy)(i3_model_o* self);
};

struct i3_model_o
{
    i3_model_i iface;  // interface for the model

    i3_render_context_t* context;  // render context
    i3_content_i* content;         // model content

    i3_rbk_buffer_i* positions;   // position buffer
    i3_rbk_buffer_i* normals;     // normal buffer
    i3_rbk_buffer_i* tangents;    // tangent buffer
    i3_rbk_buffer_i* binormals;   // binormal buffer
    i3_rbk_buffer_i* tex_coords;  // texture coordinate buffer
    i3_rbk_buffer_i* indices;     // index buffer

    i3_array_t meshes;          // array of meshes in the model
    i3_array_t nodes;           // array of nodes in the model
    i3_array_t node_tranforms;  // array of node transforms
    i3_array_t node_children;   // array of node children indices
    i3_array_t node_meshes;     // array of node meshes indices
};

i3_model_i* i3_model_create(i3_render_context_t* context, i3_content_i* model_content);
