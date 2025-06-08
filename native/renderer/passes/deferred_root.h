#pragma once

#include "../render_graph.h"

#define I3_RENDERER_DEFERRED_ROOT_PASS_NAME "deferred_root_pass"

i3_render_pass_desc_t* i3_renderer_get_deferred_root_pass_desc();
