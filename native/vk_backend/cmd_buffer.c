#include "cmd_buffer.h"

// resource interface

static void i3_vk_cmd_buffer_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    cmd_buffer->use_count++;
}

static void i3_vk_cmd_buffer_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    if (--cmd_buffer->use_count == 0)
    {
        // destroy the use list
        i3_vk_use_list_destroy(&cmd_buffer->use_list);

        // destroy the command list
        i3_vk_cmd_list_destroy(&cmd_buffer->cmd_list);

        // free the command buffer
        i3_memory_pool_free(&cmd_buffer->device->cmd_buffer_pool, cmd_buffer);
    }
}

static int32_t i3_vk_cmd_buffer_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    return cmd_buffer->use_count;
}

static void i3_vk_cmd_buffer_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // i3_vk_set_debug_name(cmd_buffer->device->log, cmd_buffer->device->handle, VK_OBJECT_TYPE_COMMAND_BUFFER,
    // cmd_buffer->base.handle, name);
}

// cmd buffer interface

static i3_rbk_resource_i* i3_vk_cmd_buffer_get_resource(i3_rbk_cmd_buffer_o* self)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    return &cmd_buffer->base;
}

// copy buffer
static void i3_vk_cmd_buffer_copy_buffer(i3_rbk_cmd_buffer_o* self,
                                         i3_rbk_buffer_i* src_buffer,
                                         i3_rbk_buffer_i* dst_buffer,
                                         uint32_t src_offset,
                                         uint32_t dst_offset,
                                         uint32_t size)
{
    assert(self != NULL);
    assert(src_buffer != NULL);
    assert(dst_buffer != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // add resources to the use list
    i3_vk_use_list_add(&cmd_buffer->use_list, src_buffer);
    i3_vk_use_list_add(&cmd_buffer->use_list, dst_buffer);

    // emit command
    i3_vk_cmd_copy_buffer_t* cmd = i3_vk_cmd_write_copy_buffer(&cmd_buffer->cmd_list);
    cmd->src_buffer = src_buffer;
    cmd->dst_buffer = dst_buffer;
    cmd->src_offset = src_offset;
    cmd->dst_offset = dst_offset;
    cmd->size = size;
}

// write buffer
static void i3_vk_cmd_buffer_write_buffer(i3_rbk_cmd_buffer_o* self,
                                          i3_rbk_buffer_i* buffer,
                                          uint32_t dst_offset,
                                          uint32_t size,
                                          const void* data)
{
    assert(self != NULL);
    assert(buffer != NULL);
    assert(data != NULL);
    assert(size > 0);

    // TODO: use vkUpdateBuffer for small data transfers ?

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;
    i3_rbk_device_i* device = &cmd_buffer->device->iface;

    // create a temporary buffer of size
    i3_rbk_buffer_desc_t temp_buffer_desc = {
        .flags = I3_RBK_BUFFER_FLAG_STAGING,
        .size = size,
    };
    i3_rbk_buffer_i* staging_buffer = device->create_buffer(device->self, &temp_buffer_desc);

    // copy data to the temporary buffer
    void* dst = staging_buffer->map(staging_buffer->self);
    memcpy(dst, data, size);
    staging_buffer->unmap(staging_buffer->self);

    // copy the temporary buffer to the destination buffer
    i3_vk_cmd_buffer_copy_buffer((i3_rbk_cmd_buffer_o*)cmd_buffer, staging_buffer, buffer, 0, dst_offset, size);

    // destroy the temporary buffer
    staging_buffer->destroy(staging_buffer->self);
}

static void i3_vk_cmd_buffer_destroy(i3_rbk_cmd_buffer_o* self)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    cmd_buffer->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_cmd_buffer_o i3_vk_cmd_buffer_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_cmd_buffer_add_ref,
        .release = i3_vk_cmd_buffer_release,
        .get_use_count = i3_vk_cmd_buffer_get_use_count,
        .set_debug_name = i3_vk_cmd_buffer_set_debug_name,
    },
    .iface =
    {
        .get_resource = i3_vk_cmd_buffer_get_resource,
        .write_buffer = i3_vk_cmd_buffer_write_buffer,
        .copy_buffer = i3_vk_cmd_buffer_copy_buffer,
        .destroy = i3_vk_cmd_buffer_destroy,
    },
};

// create cmd buffer

i3_rbk_cmd_buffer_i* i3_vk_device_create_cmd_buffer(i3_rbk_device_o* self)
{
    assert(self != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;
    i3_vk_cmd_buffer_o* cmd_buffer = i3_memory_pool_alloc(&device->cmd_buffer_pool);

    *cmd_buffer = i3_vk_cmd_buffer_iface_;
    cmd_buffer->base.self = (i3_rbk_resource_o*)cmd_buffer;
    cmd_buffer->iface.self = (i3_rbk_cmd_buffer_o*)cmd_buffer;
    cmd_buffer->device = device;
    cmd_buffer->use_count = 1;

    i3_vk_use_list_init(&cmd_buffer->use_list, device);
    i3_vk_cmd_list_init(&cmd_buffer->cmd_list, device);

    return &cmd_buffer->iface;
}