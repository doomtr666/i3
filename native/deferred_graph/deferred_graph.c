#include "deferred_graph.h"

// setup default render passes
void i3_setup_deferred_graph(i3_render_graph_builder_i* graph_builder)
{
    assert(graph_builder != NULL);

    // create an extensible graph, based on the default passes
    graph_builder->begin_pass(graph_builder->self, NULL, i3_get_deferred_root_pass_desc());
    graph_builder->add_pass(graph_builder->self, NULL, i3_get_gbuffer_pass_desc());
    graph_builder->add_pass(graph_builder->self, NULL, i3_get_opaque_pass_desc());
    graph_builder->add_pass(graph_builder->self, NULL, i3_get_light_pass_desc());
    graph_builder->end_pass(graph_builder->self);
}