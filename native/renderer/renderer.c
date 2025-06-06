#include "renderer.h"

struct i3_renderer_o
{
    i3_renderer_i iface;
    i3_render_context_t context;
};

static i3_render_graph_builder_i* i3_render_create_graph_builder(i3_renderer_o* self)
{
    assert(self != NULL);

    return i3_render_graph_builder_create(self->context.backend);
}

static void i3_renderer_set_render_graph(i3_renderer_o* self, i3_render_graph_i* graph)
{
    assert(self != NULL);
    assert(graph != NULL);

    self->context.render_graph = graph;

    // set the render context for the graph
    self->context.render_graph->set_render_context(self->context.render_graph->self, &self->context);

    // reset the window size to 0, to trigger a resolution change next render
    self->context.render_width = 0;
    self->context.render_height = 0;
}

static void i3_renderer_render(i3_renderer_o* self, i3_game_time_t* game_time)
{
    assert(self != NULL);
    assert(game_time != NULL);

    // update the game time in the render context
    self->context.time = *game_time;

    if (self->context.render_graph != NULL)
    {
        // check if the window size has changed
        uint32_t render_width, render_height;
        self->context.window->get_render_size(self->context.window->self, &render_width, &render_height);

        if (self->context.render_width != render_width || self->context.render_height != render_height)
        {
            self->context.render_width = render_width;
            self->context.render_height = render_height;

            // if the render size is zero, skip rendering
            if (render_width == 0 || render_height == 0)
                return;

            // trigger resolution change in the render graph
            self->context.render_graph->resolution_change(self->context.render_graph->self);
        }

        // update the render graph
        self->context.render_graph->update(self->context.render_graph->self);

        // render the graph
        self->context.render_graph->render(self->context.render_graph->self);
    }
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
        .create_graph_builder = i3_render_create_graph_builder,
        .set_render_graph = i3_renderer_set_render_graph,
        .render = i3_renderer_render,
        .destroy = i3_renderer_destroy,
    },
};

i3_renderer_i* i3_renderer_create(i3_render_backend_i* backend, i3_render_window_i* window)
{
    assert(backend != NULL);

    i3_renderer_o* renderer = i3_alloc(sizeof(i3_renderer_o));
    assert(renderer != NULL);

    *renderer = i3_renderer_iface_;
    renderer->iface.self = renderer;
    renderer->context = (i3_render_context_t){
        .backend = backend,
        .window = window,
        .renderer = &renderer->iface,
    };

    return &renderer->iface;
}