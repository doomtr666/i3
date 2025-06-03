#include "renderer.h"

struct i3_renderer_o
{
    i3_renderer_i iface;
    i3_render_backend_i* backend;
};

static i3_render_graph_builder_i* i3_render_create_graph_builder(i3_renderer_o* self)
{
    assert(self != NULL);

    return i3_render_graph_builder_create(self->backend);
}

static void i3_renderer_destroy(i3_renderer_o* self)
{
    assert(self != NULL);
    i3_free(self);
}

static i3_renderer_o i3_renderer_iface_ =
{
    .iface =
    {
        .self = NULL,
        .create_graph_builder = i3_render_create_graph_builder,
        .destroy = i3_renderer_destroy,
    },
};

i3_renderer_i* i3_renderer_create(i3_render_backend_i* backend)
{
    assert(backend != NULL);

    i3_renderer_o* renderer = i3_alloc(sizeof(i3_renderer_o));
    assert(renderer != NULL);

    *renderer = i3_renderer_iface_;
    renderer->iface.self = renderer;
    renderer->backend = backend;

    return &renderer->iface;
}