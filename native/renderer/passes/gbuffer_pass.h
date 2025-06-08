#pragma once

#include "../render_graph.h"

#define I3_RENDERER_GBUFFER_PASS_NAME "gbuffer_pass"

i3_render_pass_desc_t* i3_renderer_get_gbuffer_pass_desc();