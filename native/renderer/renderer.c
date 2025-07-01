#include "renderer.h"

struct i3_renderer_o
{
    i3_renderer_i iface;
    i3_render_context_t context;
    i3_render_graph_i* graph;
};

static i3_model_i* i3_renderer_create_model(i3_renderer_o* self,
                                            i3_rbk_cmd_buffer_i* cmb_buffer,
                                            i3_content_i* model_content)
{
    assert(self != NULL);
    assert(cmb_buffer != NULL);
    assert(model_content != NULL);

    return i3_model_create(&self->context, cmb_buffer, model_content);
}

static i3_scene_i* i3_renderer_create_scene(i3_renderer_o* self)
{
    assert(self != NULL);

    return i3_scene_create(&self->context);
}

static void i3_renderer_set_scene(i3_renderer_o* self, i3_scene_i* scene)
{
    assert(self != NULL);
    assert(scene != NULL);

    // TODO
}

static i3_render_graph_builder_i* i3_renderer_create_graph_builder(i3_renderer_o* self)
{
    assert(self != NULL);

    return i3_render_graph_builder_create(&self->context);
}

static void i3_renderer_set_render_graph(i3_renderer_o* self, i3_render_graph_i* graph)
{
    assert(self != NULL);
    assert(graph != NULL);

    self->graph = graph;

    // reset the window size to 0, to trigger a resolution change next render
    self->context.render_width = 0;
    self->context.render_height = 0;
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

    if (self->graph != NULL)
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
            self->graph->resolution_change(self->graph->self);
        }

        // update the render graph
        self->graph->update(self->graph->self);

        // render the graph
        self->graph->render(self->graph->self);

        // get the output image from the graph
        i3_render_target_t output;
        if (self->graph->get(self->graph->self, "output", &output))
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
        .create_model = i3_renderer_create_model,
        .create_scene = i3_renderer_create_scene,
        .set_scene = i3_renderer_set_scene,
        .create_graph_builder = i3_renderer_create_graph_builder,
        .set_render_graph = i3_renderer_set_render_graph,
        .create_render_target = i3_renderer_create_render_target,
        .destroy_render_target = i3_renderer_destroy_render_target,
        .render = i3_renderer_render,
        .destroy = i3_renderer_destroy,
    },
};

i3_renderer_i* i3_renderer_create(i3_render_backend_i* backend,
                                  i3_render_window_i* window,
                                  i3_content_store_i* content_store)
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
        .log = i3_get_logger(I3_RENDERER_LOGGER_NAME),
        .backend = backend,
        .window = window,
        .device = device,
        .swapchain = swapchain,
        .renderer = &renderer->iface,
        .content_store = content_store,
    };

    return &renderer->iface;
}