
extern "C"
{
#include "scene.h"
}

#include "fbs/model_generated.h"

struct i3_model_o
{
    i3_model_i iface;  // interface for the model

    i3_rbk_buffer_i* positions;   // position buffer
    i3_rbk_buffer_i* normals;     // normal buffer
    i3_rbk_buffer_i* tangents;    // tangent buffer
    i3_rbk_buffer_i* binormals;   // binormal buffer
    i3_rbk_buffer_i* tex_coords;  // texture coordinate buffer
    i3_rbk_buffer_i* indices;     // index buffer
};

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

extern "C" i3_model_i* i3_model_create(i3_render_context_t* context,
                                       i3_rbk_cmd_buffer_i* cmd_buffer,
                                       i3_content_i* model_content)
{
    assert(context != NULL);
    assert(model_content != NULL);

    auto data = model_content->get_data(model_content->self);
    assert(data != NULL);
    auto size = model_content->get_size(model_content->self);
    assert(size > 0);

    // vertify the model content
    if (!content::VerifyModelBuffer(flatbuffers::Verifier((const uint8_t*)data, size)))
    {
        i3_log_err(context->log, "Invalid model content");
        return NULL;
    }

    // create the model object
    i3_model_o* model = (i3_model_o*)i3_alloc(sizeof(i3_model_o));
    memset(model, 0, sizeof(i3_model_o));

    // initialize the model interface
    model->iface.self = model;
    model->iface.destroy = i3_model_destroy;  // set the destroy function

    // parse the model content
    auto model_data = content::GetModel(data);

    // create and upload position buffer
    if (model_data->positions() != nullptr & model_data->positions()->size() > 0)
    {
        i3_rbk_buffer_desc_t position_desc = {
            I3_RBK_BUFFER_FLAG_VERTEX_BUFFER,
            model_data->positions()->size() * sizeof(content::Vec3),
        };

        model->positions = context->device->create_buffer(context->device->self, &position_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, model->positions, 0, position_desc.size,
                                 model_data->positions()->Data());
    }

    // create and upload normal buffer
    if (model_data->normals() != nullptr && model_data->normals()->size() > 0)
    {
        i3_rbk_buffer_desc_t normal_desc = {
            I3_RBK_BUFFER_FLAG_VERTEX_BUFFER,
            model_data->normals()->size() * sizeof(content::Vec3),
        };

        model->normals = context->device->create_buffer(context->device->self, &normal_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, model->normals, 0, normal_desc.size, model_data->normals()->Data());
    }

    // create and upload tangent buffer
    if (model_data->tangents() != nullptr && model_data->tangents()->size() > 0)
    {
        i3_rbk_buffer_desc_t tangent_desc = {
            I3_RBK_BUFFER_FLAG_VERTEX_BUFFER,
            model_data->tangents()->size() * sizeof(content::Vec3),
        };

        model->tangents = context->device->create_buffer(context->device->self, &tangent_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, model->tangents, 0, tangent_desc.size,
                                 model_data->tangents()->Data());
    }

    // create and upload binormal buffer
    if (model_data->binormals() != nullptr && model_data->binormals()->size() > 0)
    {
        i3_rbk_buffer_desc_t binormal_desc = {
            I3_RBK_BUFFER_FLAG_VERTEX_BUFFER,
            model_data->binormals()->size() * sizeof(content::Vec3),
        };

        model->binormals = context->device->create_buffer(context->device->self, &binormal_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, model->binormals, 0, binormal_desc.size,
                                 model_data->binormals()->Data());
    }

    // create and upload texture coordinate buffer
    if (model_data->tex_coords() != nullptr && model_data->tex_coords()->size() > 0)
    {
        i3_rbk_buffer_desc_t tex_coord_desc = {
            I3_RBK_BUFFER_FLAG_VERTEX_BUFFER,
            model_data->tex_coords()->size() * sizeof(content::Vec2),
        };

        model->tex_coords = context->device->create_buffer(context->device->self, &tex_coord_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, model->tex_coords, 0, tex_coord_desc.size,
                                 model_data->tex_coords()->Data());
    }

    // create and upload index buffer
    if (model_data->indices() != nullptr && model_data->indices()->size() > 0)
    {
        i3_rbk_buffer_desc_t index_desc = {
            I3_RBK_BUFFER_FLAG_INDEX_BUFFER,
            model_data->indices()->size() * sizeof(uint32_t),
        };

        model->indices = context->device->create_buffer(context->device->self, &index_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, model->indices, 0, index_desc.size, model_data->indices()->Data());
    }

    return &model->iface;
}