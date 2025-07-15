#include "scene.h"

// model instance

struct i3_model_instance_o
{
    i3_model_instance_i iface;  // interface for the model instance

    i3_content_i* content;  // model content
    i3_model_i* model;      // GPU loaded model
    i3_mat4_t transform;    // the transform of the instance
};

static void i3_model_instance_destroy(i3_model_instance_o* self)
{
    assert(self != NULL);

    i3_free(self);
}

static i3_model_instance_o i3_model_instance_iface_ = {
    .iface.destroy = (void (*)(i3_model_instance_o*))i3_model_instance_destroy,
};

i3_model_instance_i* i3_model_instance_create(i3_mat4_t transform)
{
    i3_model_instance_o* instance = (i3_model_instance_o*)i3_alloc(sizeof(i3_model_instance_o));
    *instance = i3_model_instance_iface_;  // set the interface
    instance->iface.self = instance;       // set the self pointer
    instance->transform = transform;       // set the transform

    return &instance->iface;
}

struct i3_scene_o
{
    i3_scene_i iface;              // interface for the scene
    i3_render_context_t* context;  // render context

    i3_hashtable_t model_table;    // key is i3_content_i*, value is i3_model_i*
    i3_array_t models;             // array of models in the scene
    i3_array_t pending_instances;  // array of pending model instances to be added
    i3_array_t instances;          // array of model instances in the scene
};

static i3_model_instance_i* i3_scene_add_instance(i3_scene_o* self, i3_content_i* model_content, i3_mat4_t transform)
{
    assert(self != NULL);
    assert(model_content != NULL);

    model_content->add_ref(model_content->self);  // increment reference count

    i3_model_i* model = i3_hashtable_find(&self->model_table, model_content, sizeof(i3_content_i*));

    i3_model_instance_i* instance = i3_model_instance_create(transform);

    instance->self->content = model_content;  // set the content of the instance
    instance->self->model = model;            // set the model of the instance

    if (model == NULL)
        i3_array_push(&self->pending_instances, &instance);
    else
        i3_array_push(&self->instances, &instance);  // add the instance to the

    return instance;
}

static void i3_scene_update(i3_scene_o* self, i3_rbk_cmd_buffer_i* cmd_buffer, i3_game_time_t* game_time)
{
    assert(self != NULL);
    assert(game_time != NULL);

    // load pending instances
    for (uint32_t i = 0; i < 16; ++i)
    {
        // get last pending instance
        if (i3_array_count(&self->pending_instances) == 0)
            break;

        i3_model_instance_i* instance = *((i3_model_instance_i**)i3_array_back(&self->pending_instances));
        i3_array_pop(&self->pending_instances);  // remove the last pending instance

        // check if the model is already loaded
        i3_model_i* model = i3_hashtable_find(&self->model_table, &instance->self->content, sizeof(i3_content_i*));

        if (model == NULL)
        {
            // create the model
            model = i3_model_create(self->context, cmd_buffer, instance->self->content);
            if (model == NULL)
            {
                i3_log_err(self->context->log, "Failed to create model from content");
                continue;
            }

            // add the model to the hashtable
            i3_hashtable_insert(&self->model_table, &instance->self->content, sizeof(i3_content_i*), model);
            i3_array_push(&self->models, model);  // add the model to the models array
        }

        // set the model of the instance
        instance->self->model = model;

        // add the instance to the scene
        i3_array_push(&self->instances, instance);
    }
}

static void i3_scene_destroy(i3_scene_o* self)
{
    assert(self != NULL);

    // destroy all pending instances
    for (uint32_t i = 0; i < i3_array_count(&self->pending_instances); ++i)
    {
        i3_model_instance_i* instance = *((i3_model_instance_i**)i3_array_at(&self->pending_instances, i));
        instance->self->content->release(instance->self->content->self);  // release the content reference
        instance->destroy(instance->self);
    }

    // destroy all model instances in the scene
    for (uint32_t i = 0; i < i3_array_count(&self->instances); ++i)
    {
        i3_model_instance_i* instance = *((i3_model_instance_i**)i3_array_at(&self->instances, i));
        instance->self->content->release(instance->self->content->self);  // release the content reference
        instance->destroy(instance->self);
    }

    // destroy the model instances array
    for (uint32_t i = 0; i < i3_array_count(&self->models); ++i)
    {
        i3_model_i* model = *((i3_model_i**)i3_array_at(&self->models, i));
        model->destroy(model->self);  // destroy the model
    }

    // destroy data structures
    i3_array_destroy(&self->instances);
    i3_array_destroy(&self->pending_instances);  // destroy the pending instances array
    i3_array_destroy(&self->models);
    i3_hashtable_destroy(&self->model_table);

    // destroy the scene
    i3_free(self);
}

static i3_scene_o i3_scene_iface_ = {
    .iface = {
        .self = NULL,
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

    scene->context = context;  // set the render context

    i3_array_init(&scene->instances, sizeof(i3_model_instance_i));          // initialize the instances array
    i3_array_init(&scene->pending_instances, sizeof(i3_model_instance_i));  // initialize the pending instances array
    i3_array_init(&scene->models, sizeof(i3_model_i*));                     // initialize the models array
    i3_hashtable_init(&scene->model_table);                                 // initialize the models hashtable

    return &scene->iface;
}