#pragma once

#include "scene.h"

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

struct i3_model_o
{
    i3_model_i iface;  // interface for the model

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

i3_model_o* i3_model_allocate();
