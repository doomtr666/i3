#pragma once

#include "native/core/list.h"

#include "barrier.h"
#include "cmd_list.h"
#include "device.h"
#include "use_list.h"

typedef struct i3_vk_cmd_buffer_o
{
    i3_rbk_resource_i base;
    i3_rbk_cmd_buffer_i iface;
    i3_vk_device_o* device;
    uint32_t use_count;

    // list of resources used by this command buffer
    i3_vk_use_list_t use_list;

    // command list
    i3_vk_cmd_list_t cmd_list;

    // list of barriers
    i3_dlist(i3_vk_barrier_t) barriers;
} i3_vk_cmd_buffer_o;

i3_rbk_cmd_buffer_i* i3_vk_device_create_cmd_buffer(i3_rbk_device_o* self);