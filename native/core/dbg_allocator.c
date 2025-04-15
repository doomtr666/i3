#include <inttypes.h>
#include <stdio.h>

#include "common.h"
#include "list.h"

typedef struct i3_allocation_info_t
{
    struct i3_allocation_info_t* next;
    struct i3_allocation_info_t* prev;
    uint32_t index;
    void* ptr;
    size_t size;
    const char* file;
    int line;
} i3_allocation_info_t;

static i3_dlist(i3_allocation_info_t) i3_allocations_;
static uint32_t i3_allocation_counter_;
static uint32_t i3_break_allocation_index_ = UINT32_MAX;
static bool i3_dbg_alloc_initialized_ = false;

static void i3_dbg_dump_memory_leaks()
{
    if (!i3_dlist_empty(&i3_allocations_))
    {
        fprintf(stderr, "non freed memory allocations:\n");
        i3_allocation_info_t* it;
        i3_dlist_foreach(&i3_allocations_, it)
            fprintf(stderr, "{%d} %s(%d): %p (%" PRIu64 ")\n", it->index, it->file, it->line, it->ptr, it->size);

        fflush(stderr);
    }
}

static void i3_dbg_destroy_allocations(void)
{
    i3_dbg_dump_memory_leaks();
}

static void i3_dbg_allocator_initialize()
{
    if (!i3_dbg_alloc_initialized_)
    {
        atexit(i3_dbg_destroy_allocations);
        i3_dbg_alloc_initialized_ = true;
    }
}

static void i3_add_memory_tracker(void* ptr, size_t size, const char* file, int line)
{
    i3_dbg_allocator_initialize();

    i3_allocation_info_t* alloc = malloc(sizeof(i3_allocation_info_t));
    assert(alloc != NULL);
    *alloc = (i3_allocation_info_t){
        .index = i3_allocation_counter_++,
        .ptr = ptr,
        .size = size,
        .file = file,
        .line = line,
    };

    i3_dlist_append(&i3_allocations_, alloc);

    if (alloc->index == i3_break_allocation_index_)
        i3_break();
}

static void i3_remove_memory_tracker(void* ptr, const char* file, int line)
{
    i3_dbg_allocator_initialize();

    // lookup allocation
    i3_allocation_info_t* victim = NULL;
    i3_allocation_info_t* it;
    i3_dlist_foreach(&i3_allocations_, it)
    {
        if (it->ptr == ptr)
        {
            victim = it;
            break;
        }
    }

    // cleanup
    if (victim)
    {
        i3_dlist_remove(&i3_allocations_, victim);
        free(victim);
    }
}

// dbg allocator

void* i3_dbg_alloc_(size_t size, const char* file, int line)
{
    void* ptr = malloc(size);
    i3_add_memory_tracker(ptr, size, file, line);
    return ptr;
}

void* i3_dbg_calloc_(size_t n, size_t size, const char* file, int line)
{
    void* ptr = calloc(n, size);
    i3_add_memory_tracker(ptr, size, file, line);
    return ptr;
}

void* i3_dbg_realloc_(void* ptr, size_t size, const char* file, int line)
{
    void* new_ptr = realloc(ptr, size);
    i3_remove_memory_tracker(ptr, file, line);
    i3_add_memory_tracker(new_ptr, size, file, line);
    return new_ptr;
}

void i3_dbg_free_(void* ptr, const char* file, int line)
{
    i3_remove_memory_tracker(ptr, file, line);
    free(ptr);
}

void i3_dbg_break_on_alloc(uint32_t index)
{
    i3_break_allocation_index_ = index;
}
