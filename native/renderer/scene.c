#include "scene.h"

// model instance

struct i3_model_instance_o
{
    i3_model_instance_i iface;  // interface for the model instance

    i3_model_i* model;    // model
    i3_mat4_t transform;  // the transform of the instance
};

static void i3_model_instance_destroy(i3_model_instance_o* self)
{
    assert(self != NULL);

    i3_free(self);
}

static i3_model_instance_o i3_model_instance_iface_ = {
    .iface.destroy = i3_model_instance_destroy,
};

i3_model_instance_i* i3_model_instance_create(i3_model_i* model, i3_mat4_t transform)
{
    i3_model_instance_o* instance = (i3_model_instance_o*)i3_alloc(sizeof(i3_model_instance_o));
    *instance = i3_model_instance_iface_;
    instance->iface.self = instance;
    instance->model = model;
    instance->transform = transform;

    return &instance->iface;
}

struct i3_scene_o
{
    i3_scene_i iface;              // interface for the scene
    i3_render_context_t* context;  // render context

    i3_array_t models;     // array of models in the scene
    i3_array_t instances;  // array of model instances in the scene
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

    // destroy the scene
    i3_free(self);
}

static i3_scene_o i3_scene_iface_ = {
    .iface = {
        .self = NULL,
        .add_model = i3_scene_add_model,
        .add_instance = i3_scene_add_instance,
        .update = i3_scene_update,
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

    return &scene->iface;
}