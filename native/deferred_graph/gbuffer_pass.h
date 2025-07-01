#pragma once

#include "native/renderer/renderer.h"

#define I3_RENDERER_GBUFFER_PASS_NAME "gbuffer_pass"

i3_render_pass_desc_t* i3_get_gbuffer_pass_desc();