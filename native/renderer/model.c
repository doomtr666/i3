#include "model.h"

static void i3_model_destroy(i3_model_o* self)
{
    assert(self != NULL);
    if (self->positions)
        self->positions->destroy(self->positions->self);
    if (self->normals)
        self->normals->destroy(self->normals->self);
    if (self->tangents)
        self->tangents->destroy(self->tangents->self);
    if (self->binormals)
        self->binormals->destroy(self->binormals->self);
    if (self->tex_coords)
        self->tex_coords->destroy(self->tex_coords->self);
    if (self->indices)
        self->indices->destroy(self->indices->self);

    i3_free(self);
}

static i3_model_o i3_model_iface_  = {
    .iface = {
        .self = NULL,
        .destroy = i3_model_destroy,
    },
};

i3_model_o* i3_model_allocate()
{
    i3_model_o* model = (i3_model_o*)i3_alloc(sizeof(i3_model_o));
    *model = i3_model_iface_;   // initialize the interface
    model->iface.self = model;  // set the self pointer
    return model;
}