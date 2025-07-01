#pragma once

#include "deferred_root.h"
#include "gbuffer_pass.h"
#include "light_pass.h"

// setup default render passes
void i3_setup_deferred_graph(i3_render_graph_builder_i* graph_builder);