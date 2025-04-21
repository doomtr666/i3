#pragma once

#include "device.h"

#define I3_VK_CMD_LIST_BLOCK_CAPACITY 1024
#define I3_VK_CMD_LIST_BLOCK_SIZE (64 * I3_KB)

typedef struct i3_vk_cmd_list_block_t
{
    uint8_t commands[I3_VK_CMD_LIST_BLOCK_SIZE];
    uint8_t* current;
    uint8_t* end;
    struct i3_vk_cmd_list_block_t* next;
} i3_vk_cmd_list_block_t;

typedef struct i3_vk_cmd_list_t
{
    i3_vk_device_o* device;
    i3_vk_cmd_list_block_t* first;
    i3_vk_cmd_list_block_t* current;
} i3_vk_cmd_list_t;

static inline void i3_vk_cmd_list_init(i3_vk_cmd_list_t* list, i3_vk_device_o* device);
static inline void i3_vk_cmd_list_destroy(i3_vk_cmd_list_t* list);
static inline void* i3_vk_cmd_list_write(i3_vk_cmd_list_t* list, uint32_t size);
static inline void i3_vk_cmd_list_write_id(i3_vk_cmd_list_t* list, uint32_t id);
static inline void* i3_vk_cmd_list_read(i3_vk_cmd_list_t* list, uint32_t size);
static inline uint32_t i3_vk_cmd_list_read_id(i3_vk_cmd_list_t* list);

// command list

#define I3_VK_CMDS()                              \
    I3_VK_CMD_BEGIN(copy_buffer)                  \
    I3_VK_CMD_FIELD(i3_rbk_buffer_i*, src_buffer) \
    I3_VK_CMD_FIELD(i3_rbk_buffer_i*, dst_buffer) \
    I3_VK_CMD_FIELD(uint32_t, src_offset)         \
    I3_VK_CMD_FIELD(uint32_t, dst_offset)         \
    I3_VK_CMD_FIELD(uint32_t, size)               \
    I3_VK_CMD_END(copy_buffer)

// command ids
#define I3_VK_CMD_BEGIN(name) I3_VK_CMD_##name,
#define I3_VK_CMD_FIELD(type, name)
#define I3_VK_CMD_END(name)

typedef enum
{
    I3_VK_CMDS() I3_VK_CMD_END,
} i3_vk_cmd_id_t;

#undef I3_VK_CMD_BEGIN
#undef I3_VK_CMD_FIELD
#undef I3_VK_CMD_END

// command structures parameters
#define I3_VK_CMD_BEGIN(name) \
    typedef struct            \
    {
#define I3_VK_CMD_FIELD(type, name) type name;
#define I3_VK_CMD_END(name) \
    }                       \
    i3_vk_cmd_##name##_t;

I3_VK_CMDS()

#undef I3_VK_CMD_BEGIN
#undef I3_VK_CMD_FIELD
#undef I3_VK_CMD_END

// write functions
#define I3_VK_CMD_BEGIN(name)                                                                                        \
    static inline i3_vk_cmd_##name##_t* i3_vk_cmd_write_##name(i3_vk_cmd_list_t* list)                               \
    {                                                                                                                \
        assert(list != NULL);                                                                                        \
        i3_vk_cmd_list_write_id(list, I3_VK_CMD_##name);                                                             \
        i3_vk_cmd_##name##_t* cmd = (i3_vk_cmd_##name##_t*)i3_vk_cmd_list_write(list, sizeof(i3_vk_cmd_##name##_t)); \
        return cmd;                                                                                                  \
    }
#define I3_VK_CMD_FIELD(type, name)
#define I3_VK_CMD_END(name)

I3_VK_CMDS()

#undef I3_VK_CMD_BEGIN
#undef I3_VK_CMD_FIELD
#undef I3_VK_CMD_END

// forward declarations of decode functions
#define I3_VK_CMD_BEGIN(name) void i3_vk_cmd_decode_##name(void* ctx, i3_vk_cmd_##name##_t* cmd);
#define I3_VK_CMD_FIELD(type, name)
#define I3_VK_CMD_END(name)

I3_VK_CMDS()

#undef I3_VK_CMD_BEGIN
#undef I3_VK_CMD_FIELD
#undef I3_VK_CMD_END

// decode functions
#define I3_VK_CMD_BEGIN(name)                                                                   \
    case I3_VK_CMD_##name:                                                                      \
    {                                                                                           \
        i3_vk_cmd_##name##_t* cmd =                                                             \
            (i3_vk_cmd_##name##_t*)i3_vk_cmd_list_read(cmd_list, sizeof(i3_vk_cmd_##name##_t)); \
        i3_vk_cmd_decode_##name(ctx, cmd);                                                      \
        break;                                                                                  \
    }
#define I3_VK_CMD_FIELD(type, name)
#define I3_VK_CMD_END(name)

static inline void i3_vk_cmd_decode(void* ctx, i3_vk_cmd_list_t* cmd_list)
{
    assert(cmd_list != NULL);

    // reset command list for reading
    cmd_list->current = cmd_list->first;
    cmd_list->current->current = cmd_list->current->commands;

    while (true)
    {
        uint32_t cmd_id = i3_vk_cmd_list_read_id(cmd_list);
        if (cmd_id == I3_VK_CMD_END)
            break;

        switch (cmd_id)
        {
            I3_VK_CMDS()

            default:
                assert(false && "Unknown command id");
                break;
        }
    }
}

#undef I3_VK_CMD_BEGIN
#undef I3_VK_CMD_FIELD
#undef I3_VK_CMD_END

// implementation

static inline void i3_vk_cmd_list_init(i3_vk_cmd_list_t* list, i3_vk_device_o* device)
{
    assert(list != NULL);
    assert(device != NULL);

    *list = (i3_vk_cmd_list_t){.device = device};
}

static inline void i3_vk_cmd_list_destroy(i3_vk_cmd_list_t* list)
{
    assert(list != NULL);

    // free all blocks in the command list
    i3_vk_cmd_list_block_t* block = list->first;
    while (block != NULL)
    {
        i3_vk_cmd_list_block_t* next = block->next;
        i3_memory_pool_free(&list->device->cmd_list_block_pool, block);
        block = next;
    }
}

static inline void* i3_vk_cmd_list_write(i3_vk_cmd_list_t* list, uint32_t size)
{
    assert(list != NULL);
    assert(size > 0);

    // allocate a new block if needed
    if (list->current == NULL || (list->current->end + size) > (list->current->commands + I3_VK_CMD_LIST_BLOCK_SIZE))
    {
        i3_vk_cmd_list_block_t* block = i3_memory_pool_alloc(&list->device->cmd_list_block_pool);
        assert(block != NULL);

        block->end = block->commands;
        block->next = NULL;

        if (list->first == NULL)
            list->first = block;
        else
            list->current->next = block;

        list->current = block;
    }

    void* ptr = list->current->end;
    list->current->end += size;

    return ptr;
}

static inline void i3_vk_cmd_list_write_id(i3_vk_cmd_list_t* list, uint32_t id)
{
    assert(list != NULL);

    // write the command id to the command list
    uint32_t* cmd_id = (uint32_t*)i3_vk_cmd_list_write(list, sizeof(uint32_t));
    *cmd_id = id;
}

static inline void* i3_vk_cmd_list_read(i3_vk_cmd_list_t* list, uint32_t size)
{
    assert(list != NULL);
    assert(size > 0);

    if (list->current->current + size > list->current->end)
    {
        list->current = list->current->next;
        if (list->current == NULL)
            return NULL;
        list->current->current = list->current->commands;
    }

    void* ptr = list->current->current;
    list->current->current += size;
    return ptr;
}

static inline uint32_t i3_vk_cmd_list_read_id(i3_vk_cmd_list_t* list)
{
    assert(list != NULL);
    // read the command id from the command list
    uint32_t* cmd_id = (uint32_t*)i3_vk_cmd_list_read(list, sizeof(uint32_t));
    // if the command id is NULL, it means that the command list is terminated
    if (cmd_id == NULL)
        return I3_VK_CMD_END;
    return *cmd_id;
}