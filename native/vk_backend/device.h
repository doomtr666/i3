#pragma once

#include "native/core/memory_pool.h"

#include "vk_mem_alloc.h"

#include "device_desc.h"
#include "device_ext.h"
#include "backend.h"


#define I3_RESOURCE_BLOCK_CAPACITY 1024

typedef struct i3_vk_device_o
{
    i3_rbk_device_i iface;
    i3_logger_i* log;
    i3_vk_backend_o* backend;
    i3_vk_device_desc desc;
    VkDevice handle;
    i3_vkbk_device_ext_t ext;
    VmaAllocator vma;

    // resource pools
    i3_memory_pool_t sampler_pool;
    i3_memory_pool_t buffer_pool;
    i3_memory_pool_t image_pool;
    i3_memory_pool_t image_view_pool;
    i3_memory_pool_t shader_module_pool;
    i3_memory_pool_t pipeline_pool;

} i3_vk_device_o;

i3_rbk_device_i* i3_vk_device_create(i3_vk_backend_o* backend, i3_vk_device_desc* device_desc);