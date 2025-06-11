#include "renderer.h"

#include "passes/deferred_root.h"
#include "passes/gbuffer_pass.h"
#include "passes/light_pass.h"

struct i3_renderer_o
{
    i3_renderer_i iface;
    i3_render_context_t context;
};

static i3_render_graph_builder_i* i3_renderer_create_graph_builder(i3_renderer_o* self)
{
    assert(self != NULL);

    return i3_render_graph_builder_create(&self->context);
}

static void i3_renderer_set_render_graph(i3_renderer_o* self, i3_render_graph_i* graph)
{
    assert(self != NULL);
    assert(graph != NULL);

    self->context.render_graph = graph;

    // reset the window size to 0, to trigger a resolution change next render
    self->context.render_width = 0;
    self->context.render_height = 0;
}

// setup default render passes
static void i3_renderer_setup_default_passes(i3_renderer_o* self, i3_render_graph_builder_i* graph_builder)
{
    assert(self != NULL);
    assert(graph_builder != NULL);

    // create an extensible graph, based on the default passes
    graph_builder->begin_pass(graph_builder->self, NULL, i3_renderer_get_deferred_root_pass_desc());
    graph_builder->add_pass(graph_builder->self, NULL, i3_renderer_get_gbuffer_pass_desc());
    graph_builder->add_pass(graph_builder->self, NULL, i3_renderer_get_light_pass_desc());
    graph_builder->end_pass(graph_builder->self);
}

static void i3_renderer_create_render_target(i3_renderer_o* self,
                                             i3_render_target_t* target,
                                             i3_rbk_image_desc_t* image_desc,
                                             i3_rbk_image_view_desc_t* view_desc)
{
    assert(self != NULL);
    assert(target != NULL);
    assert(image_desc != NULL);
    assert(view_desc != NULL);

    // release image view if it exists
    if (target->image_view != NULL)
        target->image_view->destroy(target->image_view->self);
    // release image if it exists
    if (target->image != NULL)
        target->image->destroy(target->image->self);

    // create the image
    target->image = self->context.device->create_image(self->context.device->self, image_desc);

    // create the image view
    target->image_view = self->context.device->create_image_view(self->context.device->self, target->image, view_desc);
}

static void i3_renderer_destroy_render_target(i3_renderer_o* self, i3_render_target_t* target)
{
    assert(self != NULL);
    assert(target != NULL);

    // release image view if it exists
    if (target->image_view != NULL)
    {
        target->image_view->destroy(target->image_view->self);
        target->image_view = NULL;
    }

    // release image if it exists
    if (target->image != NULL)
    {
        target->image->destroy(target->image->self);
        target->image = NULL;
    }
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

        // get the output image from the graph
        i3_render_target_t output;
        if (self->context.render_graph->get(self->context.render_graph->self, "output", &output))
        {
            // present the output image
            self->context.device->present(self->context.device->self, self->context.swapchain, output.image_view);
        }
        else
        {
            assert(false && "Render graph did not provide an output image");
        }

        // device end frame
        self->context.device->end_frame(self->context.device->self);
    }
}

static void i3_renderer_destroy(i3_renderer_o* self)
{
    assert(self != NULL);

    // wait for last frame to complete
    self->context.device->wait_idle(self->context.device->self);

    // destroy the swapchain
    self->context.swapchain->destroy(self->context.swapchain->self);

    // destroy the device
    self->context.device->destroy(self->context.device->self);

    i3_free(self);
}

static i3_renderer_o i3_renderer_iface_ =
{
    .iface =
    {
        .create_graph_builder = i3_renderer_create_graph_builder,
        .set_render_graph = i3_renderer_set_render_graph,
        .setup_default_passes = i3_renderer_setup_default_passes,
        .create_render_target = i3_renderer_create_render_target,
        .destroy_render_target = i3_renderer_destroy_render_target,
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

    // create the backend device
    i3_rbk_device_i* device = backend->create_device(backend->self, 0);

    // create the swapchain
    i3_rbk_swapchain_desc_t swapchain_desc = {
        .requested_image_count = 2,
        .srgb = false,
        .vsync = true,
    };
    i3_rbk_swapchain_i* swapchain = device->create_swapchain(device->self, window, &swapchain_desc);

    renderer->context = (i3_render_context_t){
        .backend = backend,
        .window = window,
        .device = device,
        .swapchain = swapchain,
        .renderer = &renderer->iface,
    };

    return &renderer->iface;
}