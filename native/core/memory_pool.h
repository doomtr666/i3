#pragma once

#include "common.h"
#include "list.h"

typedef struct i3_memory_pool_t i3_memory_pool_t;

static inline void i3_memory_pool_init(i3_memory_pool_t* pool, uint32_t elem_align, uint32_t elem_size, uint32_t block_capacity);
static inline void i3_memory_pool_destroy(i3_memory_pool_t* pool);
static inline void* i3_memory_pool_alloc(i3_memory_pool_t* pool);
static inline void i3_memory_pool_free(i3_memory_pool_t* pool, void* ptr);
static inline uint32_t i3_memory_pool_total_capacity(i3_memory_pool_t* pool);
static inline uint32_t i3_memory_pool_allocated(i3_memory_pool_t* pool);

// implementation
typedef struct i3_memory_pool_block_t
{
    struct i3_memory_pool_block_t* next;
    struct i3_memory_pool_block_t* prev;
} i3_memory_pool_block_t;

struct i3_memory_pool_t
{
    uint32_t elem_align;
    uint32_t elem_size;
    uint32_t elem_aligned_size;
    uint32_t block_capacity;
    uint32_t block_size;

    i3_dlist(i3_memory_pool_block_t) blocks;

    uint32_t current_block_used;
    uint8_t* current_block_start;
    void* next_free;

    // stats
    uint32_t total_capacity;
    uint32_t allocated;
};

static inline void i3_memory_pool_ensure_free_(i3_memory_pool_t* pool)
{
    assert(pool != NULL);

    // check if a free elem exists
    if (pool->next_free != NULL || pool->current_block_used < pool->block_capacity)
        return;

    // allocate a new block
    i3_memory_pool_block_t* block = (i3_memory_pool_block_t*)i3_alloc(pool->block_size);
    i3_dlist_append(&pool->blocks, block);

    // stats
    pool->total_capacity += pool->block_capacity;

    // reset used
    pool->current_block_start = (uint8_t*)i3_align_p(block + 1, pool->elem_align);
    pool->current_block_used = 0;
}

static inline void i3_memory_pool_init(i3_memory_pool_t* pool,
    uint32_t elem_align,
    uint32_t elem_size,
    uint32_t block_capacity)
{
    assert(pool != NULL);
    assert(elem_align == 0 || i3_check_align(elem_align));
    assert(block_capacity > 1);

    // make sure elem align is at least sizeof(void*) for free ptr management
    if (elem_align < I3_DEFAULT_ALIGN)
        elem_align = I3_DEFAULT_ALIGN;

    uint32_t elem_aligned_size = i3_align_v(elem_size, elem_align);
    uint32_t block_size = block_capacity * elem_aligned_size + sizeof(i3_memory_pool_block_t) + elem_align - 1;

    memset(pool, 0, sizeof(i3_memory_pool_t));

    pool->elem_align = elem_align;
    pool->elem_size = elem_size;
    pool->elem_aligned_size = elem_aligned_size;
    pool->block_capacity = block_capacity;
    pool->block_size = block_size;
    pool->current_block_used = pool->block_capacity;
}

static inline void i3_memory_pool_destroy(i3_memory_pool_t* pool)
{
    assert(pool != NULL);

    // free blocks
    while (!i3_dlist_empty(&pool->blocks))
    {
        i3_memory_pool_block_t* victim = i3_dlist_first(&pool->blocks);
        i3_dlist_remove(&pool->blocks, victim);
        i3_free(victim);
    }
}

static inline void* i3_memory_pool_alloc(i3_memory_pool_t* pool)
{
    assert(pool != NULL);

    i3_memory_pool_ensure_free_(pool);

    void* ptr;
    if (pool->next_free)
    {
        ptr = pool->next_free;
        pool->next_free = *((void**)ptr);
    }
    else
    {
        ptr = pool->current_block_start + pool->current_block_used * pool->elem_aligned_size;
        ++pool->current_block_used;
    }

    ++pool->allocated;

    return ptr;
}

static inline void i3_memory_pool_free(i3_memory_pool_t* pool, void* ptr)
{
    assert(pool != NULL);
    assert(ptr != NULL);

    *((void**)ptr) = pool->next_free;
    pool->next_free = ptr;

    --pool->allocated;
}

static inline uint32_t i3_memory_pool_total_capacity(i3_memory_pool_t* pool)
{
    assert(pool != NULL);

    return pool->total_capacity;
}

static inline uint32_t i3_memory_pool_allocated(i3_memory_pool_t* pool)
{
    assert(pool != NULL);

    return pool->allocated;
}