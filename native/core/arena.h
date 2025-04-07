#pragma once

#include "array.h"

// TODO handle alignment
typedef struct i3_arena_t i3_arena_t;

static inline void i3_arena_init(i3_arena_t* arena, uint32_t block_size);
static inline void i3_arena_free(i3_arena_t* arena);
static inline void* i3_arena_alloc(i3_arena_t* arena, uint32_t size);
static inline uint32_t i3_arena_block_size(i3_arena_t* arena);
static inline uint32_t i3_arena_allocated(i3_arena_t* arena);
static inline uint32_t i3_arena_allocation_count(i3_arena_t* arena);

// implementation

struct i3_arena_t
{
    i3_array_t blocks;
    uint32_t block_size;
    uint32_t allocated;
    uint8_t* block_start;
    uint8_t* block_end;
};

static inline void i3_arena_init(i3_arena_t* arena, uint32_t block_size)
{
    assert(arena != NULL);

    i3_array_init(&arena->blocks, sizeof(void*));
    arena->block_size = block_size;
    arena->allocated = 0;
    arena->block_start = NULL;
    arena->block_end = NULL;
}

static inline void i3_arena_free(i3_arena_t* arena)
{
    assert(arena != NULL);

    // free all blocks
    for (uint32_t i = 0; i < i3_array_count(&arena->blocks); ++i)
    {
        void* block = *(void**)i3_array_at(&arena->blocks, i);
        i3_free(block);
    }

    // free array
    i3_array_free(&arena->blocks);
}

static inline void* i3_arena_alloc(i3_arena_t* arena, uint32_t size)
{
    assert(arena != NULL);

    if (size > arena->block_size / 2)
    {
        // allocate new block to avoid fragmentation
        uint8_t* block = (uint8_t*)i3_alloc(size);
        assert(block != NULL);
        i3_array_push(&arena->blocks, &block);
        arena->allocated += size;
        return block;
    }

    if (arena->block_start == NULL || (uint32_t)(arena->block_end - arena->block_start) < size)
    {
        // allocate new block
        uint8_t* block = (uint8_t*)i3_alloc(arena->block_size);
        assert(block != NULL);

        i3_array_push(&arena->blocks, &block);
        arena->block_start = block;
        arena->block_end = block + arena->block_size;
        arena->allocated += arena->block_size;
    }

    uint8_t* ptr = arena->block_start;
    arena->block_start += size;
    return ptr;
}

static inline uint32_t i3_arena_block_size(i3_arena_t* arena)
{
    assert(arena != NULL);

    return arena->block_size;
}

static inline uint32_t i3_arena_allocated(i3_arena_t* arena)
{
    assert(arena != NULL);

    return arena->allocated;
}

static inline uint32_t i3_arena_allocation_count(i3_arena_t* arena)
{
    assert(arena != NULL);

    return i3_array_count(&arena->blocks);
}