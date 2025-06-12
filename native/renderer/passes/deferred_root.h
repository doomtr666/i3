#pragma once

#include "../render_graph.h"

#include "native/math/mat.h"

#define I3_RENDERER_DEFERRED_ROOT_PASS_NAME "deferred_root_pass"

// this pass is the root of the deferred rendering pipeline
// it maintains up to date main uniform buffer
// pass parameters:
// i3_cam_t "cam", the camera to used for rendering

i3_render_pass_desc_t* i3_renderer_get_deferred_root_pass_desc();
