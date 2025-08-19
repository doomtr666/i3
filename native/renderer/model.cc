
extern "C"
{
#include "model.h"
}

#include "fbs/model_generated.h"

static void i3_model_upload(i3_model_o* self, i3_rbk_cmd_buffer_i* cmd_buffer)
{
    assert(self != NULL);
    assert(cmd_buffer != NULL);
}

static bool i3_model_is_loaded(i3_model_o* self)
{
    assert(self != NULL);

    return self->positions != NULL;
}

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

    // release the content reference
    if (self->content)
        self->content->release(self->content->self);

    i3_free(self);
}

i3_model_i* i3_model_create(i3_render_context_t* context, i3_content_i* model_content)
{
    assert(context != NULL);
    assert(model_content != NULL);

    auto data = model_content->get_data(model_content->self);
    assert(data != NULL);
    auto size = model_content->get_size(model_content->self);
    assert(size > 0);

    // verify the model content
    auto verifier = flatbuffers::Verifier((const uint8_t*)data, size);
    if (!content::VerifyModelBuffer(verifier))
    {
        i3_log_err(context->log, "Invalid model content");
        return NULL;
    }

    // keep a reference to the content
    model_content->add_ref(model_content->self);  // add a reference to the content

    // create the model object
    i3_model_o* model = (i3_model_o*)i3_alloc(sizeof(i3_model_o));
    memset(model, 0, sizeof(i3_model_o));  // zero-initialize the model
    model->iface.self = model;             // set the self pointer
    model->iface.is_loaded = i3_model_is_loaded;
    model->iface.upload = i3_model_upload;
    model->iface.destroy = i3_model_destroy;
    model->context = context;        // set the render context
    model->content = model_content;  // set the model content

    return &model->iface;
}

#if 0

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
    auto verifier = flatbuffers::Verifier((const uint8_t*)data, size);
    if (!content::VerifyModelBuffer(verifier))
    {
        i3_log_err(context->log, "Invalid model content");
        return NULL;
    }

    // create the model object
    i3_model_o* model = (i3_model_o*)i3_alloc(sizeof(i3_model_o));
    *model = i3_model_iface_;   // initialize the interface
    model->iface.self = model;  // set the self pointer

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

    // meshes
    i3_array_init_capacity(&model->meshes, sizeof(content::Mesh), model_data->meshes()->size());
    for (const auto& mesh : *model_data->meshes())
    {
        i3_mesh_t mesh_data = {
            mesh->vertex_offset(),
            mesh->index_offset(),
            mesh->index_count(),
            mesh->material_index(),
        };

        i3_array_push(&model->meshes, &mesh_data);
    }

    // TODO: materials

    // nodes
    i3_array_init_capacity(&model->nodes, sizeof(content::Node), model_data->nodes()->size());
    for (const auto& node : *model_data->nodes())
    {
        i3_node_t node_data = {
            node->mesh_offset(),
            node->mesh_count(),
            node->children_offset(),
            node->children_count(),
        };

        i3_array_push(&model->nodes, &node_data);
    }

    // TODO: node names

    // node transforms
    i3_array_init_capacity(&model->node_tranforms, sizeof(content::Mat4), model_data->node_transforms()->size());
    for (const auto& transform : *model_data->node_transforms())
    {
        i3_mat4_t transform_data;
        for (size_t i = 0; i < 16; ++i)
            transform_data.m[i] = transform->m()->Get(i);
        i3_array_push(&model->node_tranforms, &transform_data);
    }

    // node meshes
    i3_array_init_capacity(&model->node_meshes, sizeof(uint32_t), model_data->node_meshes()->size());
    for (const auto& mesh_index : *model_data->node_meshes())
    {
        uint32_t index = mesh_index;
        i3_array_push(&model->node_meshes, &index);
    }

    // node children
    i3_array_init_capacity(&model->node_children, sizeof(uint32_t), model_data->node_children()->size());
    for (const auto& child_index : *model_data->node_children())
    {
        uint32_t index = child_index;
        i3_array_push(&model->node_children, &index);
    }

    return &model->iface;
}

#endif