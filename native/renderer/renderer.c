#include "renderer.h"

struct i3_renderer_o
{
    i3_renderer_i iface;
    i3_render_backend_i* backend;
    i3_render_window_i* window;
    uint32_t win_width, win_height;
    i3_render_graph_i* render_graph;
};

static i3_render_graph_builder_i* i3_render_create_graph_builder(i3_renderer_o* self)
{
    assert(self != NULL);

    return i3_render_graph_builder_create(self->backend);
}

static void i3_renderer_set_render_graph(i3_renderer_o* self, i3_render_graph_i* graph)
{
    assert(self != NULL);
    assert(graph != NULL);

    self->render_graph = graph;
    // reset the window size to 0, to trigger a resolution change next render
    self->win_width = 0;
    self->win_height = 0;
}

static void i3_renderer_render(i3_renderer_o* self, i3_game_time_t* game_time)
{
    assert(self != NULL);
    assert(game_time != NULL);

    if (self->render_graph != NULL)
    {
        // check if the window size has changed
        uint32_t win_width, win_height;
        self->window->get_render_size(self->window->self, &win_width, &self->win_height);

        if (self->win_width != win_width || self->win_height != win_height)
        {
            self->win_width = win_width;
            self->win_height = win_height;

            if (win_width == 0 || win_height == 0)
            {
                // if the window size is zero, skip rendering
                return;
            }

            // trigger resolution change in the render graph
            self->render_graph->resolution_change(self->render_graph);
        }

        // update the render graph
        self->render_graph->update(self->render_graph);
        // render the graph
        self->render_graph->render(self->render_graph);
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
    renderer->backend = backend;
    renderer->window = window;

    return &renderer->iface;
}