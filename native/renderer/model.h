#pragma once

#include "scene.h"

struct i3_model_o
{
    i3_model_i iface;  // interface for the model

    i3_rbk_buffer_i* positions;   // position buffer
    i3_rbk_buffer_i* normals;     // normal buffer
    i3_rbk_buffer_i* tangents;    // tangent buffer
    i3_rbk_buffer_i* binormals;   // binormal buffer
    i3_rbk_buffer_i* tex_coords;  // texture coordinate buffer
    i3_rbk_buffer_i* indices;     // index buffer
};

i3_model_o* i3_model_allocate();
