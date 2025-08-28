
extern "C"
{
#include "model.h"
}

#include "fbs/model_generated.h"

struct i3_model_o
{
    i3_model_i iface;  // interface for the model

    i3_render_context_t* context;  // render context
    i3_content_i* content;         // model content

    i3_rbk_buffer_i* positions;   // position buffer
    i3_rbk_buffer_i* normals;     // normal buffer
    i3_rbk_buffer_i* tangents;    // tangent buffer
    i3_rbk_buffer_i* binormals;   // binormal buffer
    i3_rbk_buffer_i* tex_coords;  // texture coordinate buffer
    i3_rbk_buffer_i* indices;     // index buffer

    i3_array_t meshes;         // array of meshes in the model
    i3_array_t nodes;          // array of nodes in the model
    i3_array_t node_children;  // array of node children indices
    i3_array_t node_meshes;    // array of node meshes indices
};

static void i3_model_upload(i3_model_o* self, i3_rbk_cmd_buffer_i* cmd_buffer)
{
    assert(self != NULL);
    assert(cmd_buffer != NULL);

    if (self->positions != NULL)
        return;  // already uploaded

    i3_rbk_device_i* device = self->context->device;

    // parse the model content
    auto model_data = content::GetModel(self->content->get_data(self->content->self));

    // create and upload position buffer
    if (model_data->positions() != nullptr & model_data->positions()->size() > 0)
    {
        i3_rbk_buffer_desc_t position_desc = {
            I3_RBK_BUFFER_FLAG_VERTEX_BUFFER,
            model_data->positions()->size() * sizeof(content::Vec3),
        };

        self->positions = device->create_buffer(device->self, &position_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, self->positions, 0, position_desc.size,
                                 model_data->positions()->Data());
    }

    // create and upload normal buffer
    if (model_data->normals() != nullptr && model_data->normals()->size() > 0)
    {
        i3_rbk_buffer_desc_t normal_desc = {
            I3_RBK_BUFFER_FLAG_VERTEX_BUFFER,
            model_data->normals()->size() * sizeof(content::Vec3),
        };

        self->normals = device->create_buffer(device->self, &normal_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, self->normals, 0, normal_desc.size, model_data->normals()->Data());
    }

    // create and upload tangent buffer
    if (model_data->tangents() != nullptr && model_data->tangents()->size() > 0)
    {
        i3_rbk_buffer_desc_t tangent_desc = {
            I3_RBK_BUFFER_FLAG_VERTEX_BUFFER,
            model_data->tangents()->size() * sizeof(content::Vec3),
        };

        self->tangents = device->create_buffer(device->self, &tangent_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, self->tangents, 0, tangent_desc.size,
                                 model_data->tangents()->Data());
    }

    // create and upload binormal buffer
    if (model_data->binormals() != nullptr && model_data->binormals()->size() > 0)
    {
        i3_rbk_buffer_desc_t binormal_desc = {
            I3_RBK_BUFFER_FLAG_VERTEX_BUFFER,
            model_data->binormals()->size() * sizeof(content::Vec3),
        };

        self->binormals = device->create_buffer(device->self, &binormal_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, self->binormals, 0, binormal_desc.size,
                                 model_data->binormals()->Data());
    }

    // create and upload texture coordinate buffer
    if (model_data->tex_coords() != nullptr && model_data->tex_coords()->size() > 0)
    {
        i3_rbk_buffer_desc_t tex_coord_desc = {
            I3_RBK_BUFFER_FLAG_VERTEX_BUFFER,
            model_data->tex_coords()->size() * sizeof(content::Vec2),
        };

        self->tex_coords = device->create_buffer(device->self, &tex_coord_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, self->tex_coords, 0, tex_coord_desc.size,
                                 model_data->tex_coords()->Data());
    }

    // create and upload index buffer
    if (model_data->indices() != nullptr && model_data->indices()->size() > 0)
    {
        i3_rbk_buffer_desc_t index_desc = {
            I3_RBK_BUFFER_FLAG_INDEX_BUFFER,
            model_data->indices()->size() * sizeof(uint32_t),
        };

        self->indices = device->create_buffer(device->self, &index_desc);
        cmd_buffer->write_buffer(cmd_buffer->self, self->indices, 0, index_desc.size, model_data->indices()->Data());
    }
}

static bool i3_model_is_loaded(i3_model_o* self)
{
    assert(self != NULL);

    return self->positions != NULL;
}

static void i3_model_bind_buffers(i3_model_o* self, i3_rbk_cmd_buffer_i* cmd_buffer)
{
    assert(self != NULL);
    assert(cmd_buffer != NULL);

    i3_rbk_buffer_i* buffers[] = {
        self->positions, self->normals, self->tangents, self->binormals, self->tex_coords,
    };

    cmd_buffer->bind_vertex_buffers(cmd_buffer->self, 0, 5, buffers, NULL);
    cmd_buffer->bind_index_buffer(cmd_buffer->self, self->indices, 0, I3_RBK_INDEX_TYPE_UINT32);
}

static i3_node_t* i3_model_get_nodes(i3_model_o* self)
{
    assert(self != NULL);

    return (i3_node_t*)i3_array_data(&self->nodes);
}

static uint32_t i3_model_get_node_count(i3_model_o* self)
{
    assert(self != NULL);

    return i3_array_count(&self->nodes);
}

static i3_mesh_t* i3_model_get_meshes(i3_model_o* self)
{
    assert(self != NULL);

    return (i3_mesh_t*)i3_array_data(&self->meshes);
}

static uint32_t i3_model_get_mesh_count(i3_model_o* self)
{
    assert(self != NULL);

    return i3_array_count(&self->meshes);
}

static uint32_t* i3_model_get_node_children(i3_model_o* self)
{
    assert(self != NULL);

    return (uint32_t*)i3_array_data(&self->node_children);
}

static uint32_t* i3_model_get_node_meshes(i3_model_o* self)
{
    assert(self != NULL);

    return (uint32_t*)i3_array_data(&self->node_meshes);
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
    model->iface.bind_buffers = i3_model_bind_buffers;
    model->iface.get_nodes = i3_model_get_nodes;
    model->iface.get_node_count = i3_model_get_node_count;
    model->iface.get_meshes = i3_model_get_meshes;
    model->iface.get_mesh_count = i3_model_get_mesh_count;
    model->iface.get_node_children = i3_model_get_node_children;
    model->iface.get_node_meshes = i3_model_get_node_meshes;
    model->iface.destroy = i3_model_destroy;
    model->context = context;        // set the render context
    model->content = model_content;  // set the model content

    // parse the model content
    auto model_data = content::GetModel(model->content->get_data(model->content->self));

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
        i3_mat4_t transform;
        for (size_t i = 0; i < 16; ++i)
            transform.m[i] = node->transform().m()->Get(i);

        i3_node_t node_data = {
            transform, node->mesh_offset(), node->mesh_count(), node->children_offset(), node->children_count(),
        };

        i3_array_push(&model->nodes, &node_data);
    }

    // TODO: node names

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
