#include "model.h"

static void i3_model_destroy(i3_model_o* self)
{
    assert(self != NULL);

    // destroy the buffers
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

    // destroy the meshes array
    i3_array_destroy(&self->meshes);
    // destroy the nodes array
    i3_array_destroy(&self->nodes);
    // destroy the node transforms array
    i3_array_destroy(&self->node_tranforms);
    // destroy the node children array
    i3_array_destroy(&self->node_children);
    // destroy the node meshes array
    i3_array_destroy(&self->node_meshes);

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