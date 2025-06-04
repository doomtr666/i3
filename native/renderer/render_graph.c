#include "native/core/array.h"

#include "render_graph.h"

// pass implementation
struct i3_render_pass_o
{
    i3_render_pass_i iface;
    i3_render_pass_desc_t desc;

    i3_render_pass_o* parent;
    i3_array_t children;  // array of child passes
};

static const i3_render_pass_desc_t* i3_render_pass_get_desc(i3_render_pass_i* self)
{
    assert(self != NULL);
    i3_render_pass_o* pass = (i3_render_pass_o*)self->self;
    return &pass->desc;
}

static void* i3_render_pass_get_user_data(i3_render_pass_i* self)
{
    assert(self != NULL);
    i3_render_pass_o* pass = (i3_render_pass_o*)self->self;
    return pass->desc.user_data;
}

static void i3_render_pass_set_user_data(i3_render_pass_i* self, void* user_data)
{
    assert(self != NULL);
    i3_render_pass_o* pass = (i3_render_pass_o*)self->self;
    pass->desc.user_data = user_data;
}

static void i3_render_pass_destroy(i3_render_pass_i* self)
{
    assert(self != NULL);
    i3_render_pass_o* pass = (i3_render_pass_o*)self->self;

    // Call the custom destroy function if provided
    if (pass->desc.destroy != NULL)
        pass->desc.destroy(self);

    // Clear the children array
    i3_array_clear(&pass->children);

    // Free the pass itself
    i3_free(pass);
}

static i3_render_pass_o i3_render_pass_iface_ =
{
    .iface =
    {
        .get_desc = i3_render_pass_get_desc, 
        .get_user_data = i3_render_pass_get_user_data,
        .set_user_data = i3_render_pass_set_user_data,
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

    i3_array_init(&pass->children, sizeof(i3_render_pass_o*));

    return pass;
}

// render graph implementation
struct i3_render_graph_o
{
    i3_render_graph_i iface;
    i3_array_t passes;
    i3_array_t resolution_changes;  // array of resolution change handlers
    i3_array_t updates;             // array of update handlers
    i3_array_t renders;             // array of render handlers
};

static void i3_render_graph_resolution_change(i3_render_graph_i* self)
{
    assert(self != NULL);
    i3_render_graph_o* graph = (i3_render_graph_o*)self->self;

    // call all resolution change handlers
    for (uint32_t i = 0; i < i3_array_count(&graph->resolution_changes); ++i)
    {
        i3_render_pass_o* pass = *(i3_render_pass_o**)i3_array_at(&graph->resolution_changes, i);
        assert(pass != NULL);
        if (pass->desc.resolution_change != NULL)
            pass->desc.resolution_change(&pass->iface);
    }
}

static void i3_render_graph_update(i3_render_graph_i* self)
{
    assert(self != NULL);
    i3_render_graph_o* graph = (i3_render_graph_o*)self->self;

    // call all update handlers
    for (uint32_t i = 0; i < i3_array_count(&graph->updates); ++i)
    {
        i3_render_pass_o* pass = *(i3_render_pass_o**)i3_array_at(&graph->updates, i);
        assert(pass != NULL);
        if (pass->desc.update != NULL)
            pass->desc.update(&pass->iface);
    }
}

static void i3_render_graph_render(i3_render_graph_i* self)
{
    assert(self != NULL);
    i3_render_graph_o* graph = (i3_render_graph_o*)self->self;

    // call all render handlers
    for (uint32_t i = 0; i < i3_array_count(&graph->renders); ++i)
    {
        i3_render_pass_o* pass = *(i3_render_pass_o**)i3_array_at(&graph->renders, i);
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
        pass->iface.destroy(&pass->iface);
    }

    // clear the passes arrays
    i3_array_destroy(&self->passes);
    i3_array_destroy(&self->resolution_changes);
    i3_array_destroy(&self->updates);
    i3_array_destroy(&self->renders);

    // free the graph itself
    i3_free(self);
}

static i3_render_graph_o i3_render_graph_iface_ =
{
    .iface =
    {
        .resolution_change = i3_render_graph_resolution_change,
        .update = i3_render_graph_update,
        .render = i3_render_graph_render,
        .destroy = i3_render_graph_destroy, // to be implemented later
    },
};

// graph builder implementation

struct i3_render_graph_builder_o
{
    i3_render_graph_builder_i iface;
    i3_render_backend_i* backend;

    i3_render_pass_o* root;
    i3_array_t pass_stack;
};

static void i3_render_graph_builder_add_pass(i3_render_graph_builder_o* self, i3_render_pass_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);
}

static void i3_render_graph_builder_begin_pass(i3_render_graph_builder_o* self, i3_render_pass_desc_t* desc)
{
    assert(self != NULL);
    assert(desc != NULL);

    // Create a new render pass
    i3_render_pass_o* pass = i3_render_pass_create(desc);
    assert(pass != NULL);

    // init parent
    pass->parent = i3_array_back(&self->pass_stack);

    // Add the new pass to the parent's children
    if (pass->parent != NULL)
        i3_array_push(&pass->parent->children, pass);
    else
    {
        if (self->root == NULL)
            self->root = pass;
        else
            assert(false && "Root pass already exists, cannot add another root pass.");
    }

    // Push the new pass onto the stack
    i3_array_push(&self->pass_stack, pass);
}

static void i3_render_graph_builder_end_pass(i3_render_graph_builder_o* self)
{
    assert(self != NULL);
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

    // Create a new render graph instance
    i3_render_graph_o* graph = i3_alloc(sizeof(i3_render_graph_o));
    assert(graph != NULL);
    *graph = i3_render_graph_iface_;
    graph->iface.self = graph;

    i3_array_init(&graph->passes, sizeof(i3_render_pass_o*));
    i3_array_init(&graph->resolution_changes, sizeof(i3_render_pass_o*));
    i3_array_init(&graph->updates, sizeof(i3_render_pass_o*));
    i3_array_init(&graph->renders, sizeof(i3_render_pass_o*));

    // build the graph recursively
    i3_render_graph_builder_build_r(graph, self->root);

    return &graph->iface;
}

static void i3_render_graph_builder_destroy(i3_render_graph_builder_o* self)
{
    assert(self != NULL);

    i3_array_destroy(&self->pass_stack);
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

    return &builder->iface;
}