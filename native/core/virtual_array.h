#pragma once

#include "virtual_alloc.h"

typedef struct i3_virtual_array_t i3_virtual_array_t;

static inline void i3_virtual_array_init(i3_virtual_array_t* array, uint32_t element_size, uint32_t max_capacity);
static inline void i3_virtual_array_init_capacity(i3_virtual_array_t* array, uint32_t element_size, uint32_t capacity, uint32_t max_capacity);
static inline void i3_virtual_array_init_addr(i3_virtual_array_t* array, void* addr, uint32_t element_size, uint32_t capacity, uint32_t max_capacity);
static inline void i3_virtual_array_free(i3_virtual_array_t* array);
static inline uint32_t i3_virtual_array_push(i3_virtual_array_t* array, void* element);
static inline void i3_virtual_array_pop(i3_virtual_array_t* array);
static inline void* i3_virtual_array_addn(i3_virtual_array_t* array, uint32_t count);
static inline void* i3_virtual_array_at(i3_virtual_array_t* array, uint32_t index);
static inline void* i3_virtual_array_front(i3_virtual_array_t* array);
static inline void* i3_virtual_array_back(i3_virtual_array_t* array);
static inline void i3_virtual_array_clear(i3_virtual_array_t* array);
static inline void* i3_virtual_array_data(i3_virtual_array_t* array);
static inline uint32_t i3_virtual_array_count(i3_virtual_array_t* array);
static inline uint32_t i3_virtual_array_capacity(i3_virtual_array_t* array);
static inline uint32_t i3_virtual_array_max_capacity(i3_virtual_array_t* array);
static inline uint32_t i3_virtual_array_element_size(i3_virtual_array_t* array);

// implementation

struct i3_virtual_array_t
{
    void* data;
    uint32_t count;
    uint32_t capacity;
    uint32_t max_capacity;
    uint32_t element_size;
    bool owner;
};

static inline void i3_virtual_array_ensure__(i3_virtual_array_t* array, uint32_t requested)
{
    assert(array != NULL);
    if (array->capacity < requested)
    {
        uint32_t new_capacity = array->capacity;
        if (new_capacity == 0)
            new_capacity = 1;
        while (new_capacity < requested)
            new_capacity *= 2;

        size_t new_size = i3_align_v((size_t)new_capacity * (size_t)array->element_size, I3_PAGE_SIZE);
        size_t old_size = i3_align_v((size_t)array->capacity * (size_t)array->element_size, I3_PAGE_SIZE);
        if (old_size != new_size)
        {
            bool ret = i3_virtual_commit((uint8_t*)array->data + old_size, new_size - old_size);
            assert(ret == true);
            (void*)ret;
        }

        array->capacity = new_capacity;
    }
}

static inline void i3_virtual_array_init(i3_virtual_array_t* array, uint32_t element_size, uint32_t max_capacity)
{
    assert(array != NULL);
    assert(element_size > 0);
    assert(max_capacity > 0);

    // total size
    size_t total_size = i3_align_v((size_t)max_capacity * (size_t)element_size, I3_PAGE_SIZE);

    array->data = i3_virtual_alloc(total_size);
    assert(array->data != NULL);
    array->count = 0;
    array->capacity = 0;
    array->max_capacity = max_capacity;
    array->element_size = element_size;
    array->owner = true;
}

static inline void i3_virtual_array_init_capacity(i3_virtual_array_t* array, uint32_t element_size, uint32_t capacity, uint32_t max_capacity)
{
    assert(array != NULL);
    assert(element_size > 0);
    assert(max_capacity > 0);

    i3_virtual_array_init(array, element_size, max_capacity);
    i3_virtual_array_ensure__(array, capacity);
}

static inline void i3_virtual_array_init_addr(i3_virtual_array_t* array, void* addr, uint32_t element_size, uint32_t capacity, uint32_t max_capacity)
{
    assert(array != NULL);
    assert(addr != NULL);
    assert(element_size > 0);
    assert(max_capacity > 0);

    array->data = addr;
    array->count = 0;
    array->capacity = 0;
    array->max_capacity = max_capacity;
    array->element_size = element_size;
    array->owner = false;

    i3_virtual_array_ensure__(array, capacity);
}

static inline void i3_virtual_array_free(i3_virtual_array_t* array)
{
    assert(array != NULL);

    if (array->owner)
        i3_virtual_free(array->data);
    else
    {
        // otherwise decommit memory
        if (array->capacity > 0)
        {
            size_t size = i3_align_v((size_t)array->capacity * (size_t)array->element_size, I3_PAGE_SIZE);
            i3_virtual_decommit((uint8_t*)array->data, size);
        }
    }
}

static inline uint32_t i3_virtual_array_push(i3_virtual_array_t* array, void* element)
{
    assert(array != NULL);
    assert(element != NULL);

    i3_virtual_array_ensure__(array, array->count + 1);

    memcpy((uint8_t*)array->data + array->count * array->element_size, element, array->element_size);

    uint32_t index = array->count;
    array->count++;

    return index;
}

static inline void i3_virtual_array_pop(i3_virtual_array_t* array)
{
    assert(array != NULL);
    assert(array->count > 0);

    array->count--;
}

static inline void* i3_virtual_array_addn(i3_virtual_array_t* array, uint32_t count)
{
    assert(array != NULL);

    i3_virtual_array_ensure__(array, array->count + count);

    void* data = (uint8_t*)array->data + array->count * array->element_size;

    array->count += count;

    return data;
}

static inline void* i3_virtual_array_at(i3_virtual_array_t* array, uint32_t index)
{
    assert(array != NULL);
    assert(index < array->count);

    return (uint8_t*)array->data + index * array->element_size;
}

static inline void* i3_virtual_array_front(i3_virtual_array_t* array)
{
    assert(array != NULL);
    assert(array->count > 0);

    return array->data;
}

static inline void* i3_virtual_array_back(i3_virtual_array_t* array)
{
    assert(array != NULL);
    assert(array->count > 0);

    return (uint8_t*)array->data + (array->count - 1) * array->element_size;
}

static inline void i3_virtual_array_clear(i3_virtual_array_t* array)
{
    assert(array != NULL);

    array->count = 0;
}

static inline void* i3_virtual_array_data(i3_virtual_array_t* array)
{
    assert(array != NULL);

    return array->data;
}

static inline uint32_t i3_virtual_array_count(i3_virtual_array_t* array)
{
    assert(array != NULL);

    return array->count;
}

static inline uint32_t i3_virtual_array_capacity(i3_virtual_array_t* array)
{
    assert(array != NULL);

    return array->capacity;
}

static inline uint32_t i3_virtual_array_max_capacity(i3_virtual_array_t* array)
{
    assert(array != NULL);

    return array->max_capacity;
}

static inline uint32_t i3_virtual_array_element_size(i3_virtual_array_t* array)
{
    assert(array != NULL);

    return array->element_size;
}

static inline uint32_t i3_virtual_array_index_of(i3_virtual_array_t* array, void* element)
{
    assert(array != NULL);
    assert(element != NULL);

    uintptr_t index = ((uintptr_t)element - (uintptr_t)array->data) / array->element_size;

    if (index < array->count)
        return (uint32_t)index;

    return UINT32_MAX;
}
