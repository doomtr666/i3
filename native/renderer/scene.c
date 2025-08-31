#include "scene.h"

// model instance

struct i3_model_instance_o
{
    i3_model_instance_i iface;  // interface for the model instance

    i3_model_i* model;    // model
    i3_mat4_t transform;  // the main transform of the instance
    bool transforms_dirty;

    // list of node transforms
    i3_mat4_t node_transforms[];
};

static i3_model_i* i3_model_instance_get_model(i3_model_instance_o* self)
{
    assert(self != NULL);

    return self->model;
}

static i3_mat4_t* i3_model_instance_get_transforms(i3_model_instance_o* self)
{
    assert(self != NULL);

    return self->node_transforms;
}

static void i3_model_instance_set_transform(i3_model_instance_o* self, i3_mat4_t transform)
{
    assert(self != NULL);

    self->transform = transform;
    self->transforms_dirty = true;
}

// recursively update transforms
static void i3_model_instance_transform_update_r(i3_model_instance_o* self, i3_mat4_t transform, uint32_t index)
{
    assert(self != NULL);

    i3_node_t* nodes = self->model->get_nodes(self->model->self);
    uint32_t* node_children = self->model->get_node_children(self->model->self);

    // get the node at index
    i3_node_t* node = &nodes[index];
    i3_mat4_t world_transform = i3_mat4_mult(transform, node->transform);

    self->node_transforms[index] = world_transform;

    // recurse children
    for (uint32_t i = 0; i < node->children_count; ++i)
    {
        uint32_t child_index = node_children[node->children_offset + i];
        i3_model_instance_transform_update_r(self, world_transform, child_index);
    }
}

static void i3_model_instance_update(i3_model_instance_o* self)
{
    assert(self != NULL);

    // update instance transforms
    if (self->transforms_dirty)
    {
        i3_model_instance_transform_update_r(self, self->transform, 0);
        self->transforms_dirty = false;
    }
}

static void i3_model_instance_destroy(i3_model_instance_o* self)
{
    assert(self != NULL);

    i3_free(self);
}

static i3_model_instance_o i3_model_instance_iface_ = {
    .iface = {
        .self = NULL,
        .get_model = i3_model_instance_get_model,
        .get_transforms = i3_model_instance_get_transforms,
        .set_transform = i3_model_instance_set_transform,
        .update = i3_model_instance_update,
        .destroy = i3_model_instance_destroy,
    },
};

i3_model_instance_i* i3_model_instance_create(i3_model_i* model, i3_mat4_t transform)
{
    assert(model != NULL);

    // get model node_transforms count
    uint32_t num_transforms = model->get_node_count(model->self);

    i3_model_instance_o* instance
        = (i3_model_instance_o*)i3_alloc(sizeof(i3_model_instance_o) + sizeof(i3_mat4_t) * num_transforms);
    *instance = i3_model_instance_iface_;
    instance->iface.self = instance;
    instance->model = model;
    instance->transform = transform;
    instance->transforms_dirty = true;

    return &instance->iface;
}

struct i3_scene_o
{
    i3_scene_i iface;              // interface for the scene
    i3_render_context_t* context;  // render context

    i3_array_t models;               // array of models in the scene
    i3_array_t instances;            // array of model instances in the scene
    i3_array_t instance_transforms;  // array of model instance transforms
};

static i3_model_i* i3_scene_add_model(i3_scene_o* self, i3_content_i* model_content)
{
    assert(self != NULL);
    assert(model_content != NULL);

    i3_model_i* model = i3_model_create(self->context, model_content);
    i3_array_push(&self->models, &model);

    return model;
}

static i3_model_instance_i* i3_scene_add_instance(i3_scene_o* self, i3_model_i* model, i3_mat4_t transform)
{
    assert(self != NULL);
    assert(model != NULL);

    i3_model_instance_i* instance = i3_model_instance_create(model, transform);
    i3_array_push(&self->instances, &instance);

    return instance;
}

static void i3_scene_update(i3_scene_o* self, i3_rbk_cmd_buffer_i* cmd_buffer, i3_game_time_t* game_time)
{
    assert(self != NULL);
    assert(game_time != NULL);

    // load all models
    for (uint32_t i = 0; i < self->models.count; ++i)
    {
        i3_model_i* model = *(i3_model_i**)i3_array_at(&self->models, i);
        if (!model->is_loaded(model->self))
            model->upload(model->self, cmd_buffer);
    }

    // update all model instances
    for (uint32_t i = 0; i < self->instances.count; ++i)
    {
        i3_model_instance_i* instance = *(i3_model_instance_i**)i3_array_at(&self->instances, i);
        instance->update(instance->self);
    }
}

static void i3_scene_render(i3_scene_o* self, i3_rbk_cmd_buffer_i* cmd_buffer, void* ctx, i3_scene_visitor_t visitor)
{
    assert(self != NULL);
    assert(cmd_buffer != NULL);
    assert(visitor != NULL);

    // render all model instances
    for (uint32_t i = 0; i < i3_array_count(&self->instances); ++i)
    {
        i3_model_instance_i* instance = *(i3_model_instance_i**)i3_array_at(&self->instances, i);
        visitor(ctx, cmd_buffer, instance);
    }
}

static void i3_scene_destroy(i3_scene_o* self)
{
    assert(self != NULL);

    // destroy all model instances
    for (uint32_t i = 0; i < self->instances.count; ++i)
    {
        i3_model_instance_i* instance = *(i3_model_instance_i**)i3_array_at(&self->instances, i);
        instance->destroy(instance->self);
    }
    i3_array_destroy(&self->instances);

    // destroy all models
    for (uint32_t i = 0; i < self->models.count; ++i)
    {
        i3_model_i* model = *(i3_model_i**)i3_array_at(&self->models, i);
        model->destroy(model->self);
    }

    i3_array_destroy(&self->models);

    i3_array_destroy(&self->instance_transforms);

    // destroy the scene
    i3_free(self);
}

static i3_scene_o i3_scene_iface_ = {
    .iface = {
        .self = NULL,
        .add_model = i3_scene_add_model,
        .add_instance = i3_scene_add_instance,
        .update = i3_scene_update,
        .render = i3_scene_render,
        .destroy = i3_scene_destroy,
    },
};

i3_scene_i* i3_scene_create(i3_render_context_t* context)
{
    assert(context != NULL);

    i3_scene_o* scene = (i3_scene_o*)i3_alloc(sizeof(i3_scene_o));

    *scene = i3_scene_iface_;   // initialize the interface
    scene->iface.self = scene;  // set the self pointer
    scene->context = context;   // set the render context

    // initialize arrays
    i3_array_init(&scene->models, sizeof(i3_model_i*));
    i3_array_init(&scene->instances, sizeof(i3_model_instance_i*));
    i3_array_init(&scene->instance_transforms, sizeof(i3_mat4_t));

    return &scene->iface;
}