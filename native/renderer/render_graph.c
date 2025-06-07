#include "native/core/array.h"

#include "render_graph.h"

// render graph structure
struct i3_render_graph_o
{
    i3_render_graph_i iface;
    i3_array_t passes;
    i3_array_t resolution_changes;  // array of resolution change handlers
    i3_array_t updates;             // array of update handlers
    i3_array_t renders;             // array of render handlers

    // blackboard to comunicate between render passes
    i3_blackboard_t blackboard;
};

// pass structure
struct i3_render_pass_o
{
    i3_render_pass_i iface;
    i3_render_pass_desc_t desc;

    i3_render_context_t* context;
    i3_render_pass_o* parent;
    i3_array_t children;  // array of child passes
};

// builder structure
struct i3_render_graph_builder_o
{
    i3_render_graph_builder_i iface;
    i3_render_backend_i* backend;

    i3_render_pass_o* root;
    i3_array_t pass_stack;
    i3_hashtable_t pass_table;
};

// pass implementation

static const i3_render_pass_desc_t* i3_render_pass_get_desc(i3_render_pass_o* self)
{
    assert(self != NULL);
    return &self->desc;
}

static i3_render_backend_i* i3_render_pass_get_backend(i3_render_pass_o* self)
{
    assert(self != NULL);
    assert(self->context != NULL);

    return self->context->backend;
}

static i3_render_window_i* i3_render_pass_get_window(i3_render_pass_o* self)
{
    assert(self != NULL);
    assert(self->context != NULL);

    return self->context->window;
}

static i3_renderer_i* i3_render_pass_get_renderer(i3_render_pass_o* self)
{
    assert(self != NULL);
    assert(self->context != NULL);

    return self->context->renderer;
}

static i3_rbk_device_i* i3_render_pass_get_device(i3_render_pass_o* self)
{
    assert(self != NULL);
    assert(self->context != NULL);

    return self->context->device;
}

static void i3_render_pass_get_render_size(i3_render_pass_o* self, uint32_t* width, uint32_t* height)
{
    assert(self != NULL);
    assert(self->context != NULL);
    assert(width != NULL);
    assert(height != NULL);

    *width = self->context->render_width;
    *height = self->context->render_height;
}

static i3_game_time_t* i3_render_pass_get_game_time(i3_render_pass_o* self)
{
    assert(self != NULL);
    assert(self->context != NULL);

    return &self->context->time;
}

static void* i3_render_pass_get_user_data(i3_render_pass_o* self)
{
    assert(self != NULL);
    return self->desc.user_data;
}

static void i3_render_pass_set_user_data(i3_render_pass_o* self, void* user_data)
{
    assert(self != NULL);
    self->desc.user_data = user_data;
}

static bool i3_render_pass_put(i3_render_pass_o* self, const char* key, void* data, uint32_t size)
{
    assert(self != NULL);
    assert(self->context != NULL);
    assert(self->context->render_graph != NULL);

    i3_render_graph_o* graph = (i3_render_graph_o*)self->context->render_graph;

    return i3_blackboard_put(&graph->blackboard, key, data, size);
}

static bool i3_render_pass_get(i3_render_pass_o* self, const char* key, void* data)
{
    assert(self != NULL);
    assert(self->context != NULL);
    assert(self->context->render_graph != NULL);

    i3_render_graph_o* graph = (i3_render_graph_o*)self->context->render_graph;

    return i3_blackboard_get(&graph->blackboard, key, data);
}

static void i3_render_pass_destroy(i3_render_pass_o* self)
{
    assert(self != NULL);

    // Call the custom destroy function if provided
    if (self->desc.destroy != NULL)
        self->desc.destroy(&self->iface);

    // Clear the children array
    i3_array_destroy(&self->children);

    // Free the pass itself
    i3_free(self);
}

static i3_render_pass_o i3_render_pass_iface_ =
{
    .iface =
    {
        .get_desc = i3_render_pass_get_desc,
        .get_backend = i3_render_pass_get_backend,
        .get_window = i3_render_pass_get_window,
        .get_renderer = i3_render_pass_get_renderer,
        .get_device = i3_render_pass_get_device,
        .get_render_size = i3_render_pass_get_render_size,
        .get_game_time = i3_render_pass_get_game_time,
        .get_user_data = i3_render_pass_get_user_data,
        .set_user_data = i3_render_pass_set_user_data,
        .put = i3_render_pass_put,
        .get = i3_render_pass_get,
        .destroy = i3_render_pass_destroy,
    },
};

static i3_render_pass_o* i3_render_pass_create(i3_render_pass_desc_t* desc)
{
    assert(desc != NULL);

    i3_render_pass_o* pass = i3_alloc(sizeof(i3_render_pass_o));
    assert(pass != NULL);

    *pass = i3_render_pass_iface_;
    pass->iface.self = pass;
    pass->desc = *desc;  // copy the pass description

    i3_array_init(&pass->children, sizeof(i3_render_pass_o*));

    return pass;
}

// graph implementation

static void i3_render_graph_set_render_context(i3_render_graph_o* self, i3_render_context_t* context)
{
    assert(self != NULL);
    assert(context != NULL);

    // Set the render context for all passes
    for (uint32_t i = 0; i < i3_array_count(&self->passes); ++i)
    {
        i3_render_pass_o* pass = *(i3_render_pass_o**)i3_array_at(&self->passes, i);
        assert(pass != NULL);
        pass->context = context;
    }
}

static void i3_render_graph_resolution_change(i3_render_graph_o* self)
{
    assert(self != NULL);

    // call all resolution change handlers
    for (uint32_t i = 0; i < i3_array_count(&self->resolution_changes); ++i)
    {
        i3_render_pass_o* pass = *(i3_render_pass_o**)i3_array_at(&self->resolution_changes, i);
        assert(pass != NULL);
        if (pass->desc.resolution_change != NULL)
            pass->desc.resolution_change(&pass->iface);
    }
}

static void i3_render_graph_update(i3_render_graph_o* self)
{
    assert(self != NULL);

    // call all update handlers
    for (uint32_t i = 0; i < i3_array_count(&self->updates); ++i)
    {
        i3_render_pass_o* pass = *(i3_render_pass_o**)i3_array_at(&self->updates, i);
        assert(pass != NULL);
        if (pass->desc.update != NULL)
            pass->desc.update(&pass->iface);
    }
}

static void i3_render_graph_render(i3_render_graph_o* self)
{
    assert(self != NULL);

    // call all render handlers
    for (uint32_t i = 0; i < i3_array_count(&self->renders); ++i)
    {
        i3_render_pass_o* pass = *(i3_render_pass_o**)i3_array_at(&self->renders, i);
        assert(pass != NULL);
        if (pass->desc.render != NULL)
            pass->desc.render(&pass->iface);
    }
}

static void i3_render_graph_destroy(i3_render_graph_o* self)
{
    assert(self != NULL);

    // call all passes' destroy methods
    for (uint32_t i = 0; i < i3_array_count(&self->passes); ++i)
    {
        i3_render_pass_o* pass = *(i3_render_pass_o**)i3_array_at(&self->passes, i);
        assert(pass != NULL);
        pass->iface.destroy(pass);
    }

    // clear the passes arrays
    i3_array_destroy(&self->passes);
    i3_array_destroy(&self->resolution_changes);
    i3_array_destroy(&self->updates);
    i3_array_destroy(&self->renders);

    // destroy the blackboard
    i3_blackboard_destroy(&self->blackboard);

    // free the graph itself
    i3_free(self);
}

static bool i3_render_graph_put(i3_render_pass_o* self, const char* key, void* data, uint32_t size)
{
    assert(self != NULL);

    i3_render_graph_o* graph = (i3_render_graph_o*)self->context->render_graph;
    return i3_blackboard_put(&graph->blackboard, key, data, size);
}

static bool i3_render_graph_get(i3_render_pass_o* self, const char* key, void* data)
{
    assert(self != NULL);

    i3_render_graph_o* graph = (i3_render_graph_o*)self->context->render_graph;
    return i3_blackboard_get(&graph->blackboard, key, data);
}

static i3_render_graph_o i3_render_graph_iface_ =
{
    .iface =
    {
        .set_render_context = i3_render_graph_set_render_context,
        .resolution_change = i3_render_graph_resolution_change,
        .update = i3_render_graph_update,
        .render = i3_render_graph_render,
        .put = i3_render_graph_put,
        .get = i3_render_graph_get,
        .destroy = i3_render_graph_destroy,
    },
};

// graph builder implementation

static void i3_render_graph_builder_begin_pass(i3_render_graph_builder_o* self,
                                               const char* parent_name,
                                               i3_render_pass_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    // Create a new render pass
    i3_render_pass_o* pass = i3_render_pass_create(desc);
    assert(pass != NULL);

    i3_render_pass_o* parent = NULL;
    if (parent_name != NULL)
    {
        // Check if the parent pass exists in the pass table
        parent = (i3_render_pass_o*)i3_hashtable_find(&self->pass_table, parent_name, strlen(parent_name));
        assert(false && "Parent pass not found in the pass table.");
    }
    else if (i3_array_count(&self->pass_stack) > 0)
    {
        // If no parent name is provided, use the last pass in the stack as the parent
        parent = *((i3_render_pass_o**)i3_array_back(&self->pass_stack));
    }

    if (parent != NULL)
    {
        // Set the parent pass
        pass->parent = parent;
        i3_array_push(&pass->parent->children, pass);
    }
    else
    {
        if (self->root == NULL)
            self->root = pass;
        else
            assert(false && "Root pass already exists, cannot add another root pass.");
    }

    // Push the new pass onto the stack
    i3_array_push(&self->pass_stack, pass);

    // Add the pass to the pass table
    i3_hashtable_insert(&self->pass_table, desc->name, strlen(desc->name), pass);
}

static void i3_render_graph_builder_end_pass(i3_render_graph_builder_o* self)
{
    assert(self != NULL);
    i3_array_pop(&self->pass_stack);
}

static void i3_render_graph_builder_add_pass(i3_render_graph_builder_o* self,
                                             const char* parent_name,
                                             i3_render_pass_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    i3_render_graph_builder_begin_pass(self, parent_name, desc);
    i3_render_graph_builder_end_pass(self);
}

static void i3_render_graph_builder_build_r(i3_render_graph_o* graph, i3_render_pass_o* pass)
{
    assert(graph != NULL);

    if (pass == NULL)
        return;

    // initialize the pass
    if (pass->desc.init != NULL)
        pass->desc.init(&pass->iface);

    // Add the pass to the graph
    i3_array_push(&graph->passes, &pass);

    // Check if the pass has a resolution change handler
    if (pass->desc.resolution_change != NULL)
        i3_array_push(&graph->resolution_changes, pass);

    // Check if the pass has an update handler
    if (pass->desc.update != NULL)
        i3_array_push(&graph->updates, pass);

    // Check if the pass has a render handler
    if (pass->desc.render != NULL)
        i3_array_push(&graph->renders, pass);

    // Recursively build the children passes
    for (uint32_t i = 0; i < i3_array_count(&pass->children); ++i)
    {
        i3_render_pass_o* child_pass = *(i3_render_pass_o**)i3_array_at(&pass->children, i);
        assert(child_pass != NULL);
        i3_render_graph_builder_build_r(graph, child_pass);
    }
}

static i3_render_graph_i* i3_render_graph_builder_build(i3_render_graph_builder_o* self)
{
    assert(self != NULL);

    // pass stack should be empty at this point
    assert(i3_array_count(&self->pass_stack) == 0);

    // Create a new render graph instance
    i3_render_graph_o* graph = i3_alloc(sizeof(i3_render_graph_o));
    assert(graph != NULL);
    *graph = i3_render_graph_iface_;
    graph->iface.self = graph;

    i3_array_init(&graph->passes, sizeof(i3_render_pass_o*));
    i3_array_init(&graph->resolution_changes, sizeof(i3_render_pass_o*));
    i3_array_init(&graph->updates, sizeof(i3_render_pass_o*));
    i3_array_init(&graph->renders, sizeof(i3_render_pass_o*));

    // initialize the blackboard
    i3_blackboard_init(&graph->blackboard);

    // build the graph recursively
    i3_render_graph_builder_build_r(graph, self->root);

    return &graph->iface;
}

static void i3_render_graph_builder_destroy(i3_render_graph_builder_o* self)
{
    assert(self != NULL);

    i3_array_destroy(&self->pass_stack);
    i3_hashtable_destroy(&self->pass_table);
    i3_free(self);
}

static i3_render_graph_builder_o i3_render_graph_builder_iface_ =
{
    .iface =
    {
        .add_pass = i3_render_graph_builder_add_pass, 
        .begin_pass = i3_render_graph_builder_begin_pass,
        .end_pass = i3_render_graph_builder_end_pass, 
        .build = i3_render_graph_builder_build, 
        .destroy = i3_render_graph_builder_destroy, 
    },
};

i3_render_graph_builder_i* i3_render_graph_builder_create(i3_render_backend_i* backend)
{
    assert(backend != NULL);

    i3_render_graph_builder_o* builder = i3_alloc(sizeof(i3_render_graph_builder_o));
    assert(builder != NULL);

    *builder = i3_render_graph_builder_iface_;
    builder->iface.self = builder;
    builder->backend = backend;

    i3_array_init(&builder->pass_stack, sizeof(i3_render_pass_o*));
    i3_hashtable_init(&builder->pass_table);

    return &builder->iface;
}