#pragma once

#include "device.h"

#define I3_USE_LIST_BLOCK_CAPACITY 1024
#define I3_USE_LIST_BLOCK_RESOURCE_CAPACITY 64

typedef struct i3_vk_use_list_block_t
{
    i3_rbk_resource_i* resources[I3_USE_LIST_BLOCK_RESOURCE_CAPACITY];
    uint32_t resource_count;
    struct i3_vk_use_list_block_t* next;
} i3_vk_use_list_block_t;

typedef struct i3_vk_use_list_t
{
    i3_vk_device_o* device;
    i3_vk_use_list_block_t* current;
} i3_vk_use_list_t;

static inline void i3_vk_use_list_init(i3_vk_use_list_t* list, i3_vk_device_o* device);
static inline void i3_vk_use_list_destroy(i3_vk_use_list_t* list);
static inline void i3_vk_use_list_add_resource(i3_vk_use_list_t* list, i3_rbk_resource_i* resource);

#define i3_vk_use_list_add(list, resource)                                     \
    {                                                                          \
        i3_rbk_resource_i* res__ = (resource)->get_resource((resource)->self); \
        i3_vk_use_list_add_resource(list, res__);                              \
    }                                                                          \
    while (0)

// implementation

static inline void i3_vk_use_list_init(i3_vk_use_list_t* list, i3_vk_device_o* device)
{
    assert(list != NULL);
    assert(device != NULL);

    *list = (i3_vk_use_list_t){.device = device};
}

static inline void i3_vk_use_list_destroy(i3_vk_use_list_t* list)
{
    assert(list != NULL);

    // release all resources in the use list
    // and free the blocks
    i3_vk_use_list_block_t* block = list->current;
    while (block != NULL)
    {
        i3_vk_use_list_block_t* next = block->next;

        for (uint32_t i = 0; i < block->resource_count; ++i)
        {
            i3_rbk_resource_i* resource = block->resources[i];
            resource->release(resource->self);
        }

        i3_memory_pool_free(&list->device->use_list_block_pool, block);
        block = next;
    }
}

static inline void i3_vk_use_list_add_resource(i3_vk_use_list_t* list, i3_rbk_resource_i* resource)
{
    assert(list != NULL);
    assert(resource != NULL);

    if (list->current == NULL || list->current->resource_count >= I3_USE_LIST_BLOCK_RESOURCE_CAPACITY)
    {
        i3_vk_use_list_block_t* new_block = i3_memory_pool_alloc(&list->device->use_list_block_pool);
        new_block->resource_count = 0;
        new_block->next = list->current;
        list->current = new_block;
    }

    list->current->resources[list->current->resource_count++] = resource;
    resource->add_ref(resource->self);
}

#define i3_vk_use_list_add(list, resource)                                     \
    {                                                                          \
        i3_rbk_resource_i* res__ = (resource)->get_resource((resource)->self); \
        i3_vk_use_list_add_resource(list, res__);                              \
    }                                                                          \
    while (0)