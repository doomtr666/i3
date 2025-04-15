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
        // TODO:
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

    //i3_vk_set_debug_name(cmd_buffer->device->log, cmd_buffer->device->handle, VK_OBJECT_TYPE_COMMAND_BUFFER, cmd_buffer->base.handle, name);
}

// cmd buffer interface

static i3_rbk_resource_i* i3_vk_cmd_buffer_get_resource_i(i3_rbk_cmd_buffer_o* self)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    return &cmd_buffer->base;
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
        .get_resource_i = i3_vk_cmd_buffer_get_resource_i,
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

    return &cmd_buffer->iface;
}