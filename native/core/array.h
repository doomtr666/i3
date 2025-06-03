#pragma once

#include "common.h"

typedef struct i3_array_t i3_array_t;

static inline void i3_array_init(i3_array_t* array, uint32_t element_size);
static inline void i3_array_init_capacity(i3_array_t* array, uint32_t element_size, uint32_t capacity);
static inline void i3_array_init_count(i3_array_t* array, uint32_t element_size, uint32_t count);
static inline void i3_array_destroy(i3_array_t* array);
static inline void i3_array_push(i3_array_t* array, void* element);
static inline void i3_array_pop(i3_array_t* array);
static inline void* i3_array_addn(i3_array_t* array, uint32_t count);
static inline void i3_array_resize(i3_array_t* array, uint32_t new_count);
static inline void* i3_array_at(i3_array_t* array, uint32_t index);
static inline void* i3_array_front(i3_array_t* array);
static inline void* i3_array_back(i3_array_t* array);
static inline void i3_array_clear(i3_array_t* array);
static inline void* i3_array_data(i3_array_t* array);
static inline uint32_t i3_array_count(i3_array_t* array);
static inline uint32_t i3_array_capacity(i3_array_t* array);
static inline uint32_t i3_array_element_size(i3_array_t* array);
static inline uint32_t i3_array_index_of(i3_array_t* array, void* element);

// implementation

struct i3_array_t
{
    void* data;
    uint32_t count;
    uint32_t capacity;
    uint32_t element_size;
};

static inline void i3_array_ensure_(i3_array_t* array, uint32_t requested)
{
    assert(array != NULL);

    if (array->capacity < requested)
    {
        if (array->capacity == 0)
            array->capacity = 1;
        while (array->capacity < requested)
            array->capacity *= 2;

        void* new_data = i3_realloc(array->data, array->capacity * array->element_size);
        assert(new_data != NULL);
        array->data = new_data;
    }
}

static inline void i3_array_init(i3_array_t* array, uint32_t element_size)
{
    assert(array != NULL);

    array->count = 0;
    array->capacity = 0;
    array->data = NULL;
    array->element_size = element_size;
}

static inline void i3_array_init_capacity(i3_array_t* array, uint32_t element_size, uint32_t capacity)
{
    assert(array != NULL);

    i3_array_init(array, element_size);
    i3_array_ensure_(array, capacity);
}

static inline void i3_array_init_count(i3_array_t* array, uint32_t element_size, uint32_t count)
{
    assert(array != NULL);

    i3_array_init(array, element_size);
    i3_array_resize(array, count);
}

static inline void i3_array_destroy(i3_array_t* array)
{
    assert(array != NULL);

    i3_free(array->data);
}

static inline void i3_array_push(i3_array_t* array, void* element)
{
    assert(array != NULL);
    assert(element != NULL);

    i3_array_ensure_(array, array->count + 1);
    array->count++;
    memcpy(i3_array_at(array, array->count - 1), element, array->element_size);
}

static inline void i3_array_pop(i3_array_t* array)
{
    assert(array != NULL);
    if (array->count > 0)
        array->count--;
}

static inline void* i3_array_addn(i3_array_t* array, uint32_t count)
{
    assert(array != NULL);

    i3_array_ensure_(array, array->count + count);
    array->count += count;
    return i3_array_at(array, array->count - count);
}

static inline void i3_array_resize(i3_array_t* array, uint32_t new_count)
{
    assert(array != NULL);

    i3_array_ensure_(array, new_count);
    array->count = new_count;
}

static inline void* i3_array_at(i3_array_t* array, uint32_t index)
{
    assert(array != NULL);
    assert(index < array->count);

    return (uint8_t*)array->data + index * array->element_size;
}

static inline void* i3_array_front(i3_array_t* array)
{
    assert(array != NULL);
    assert(array->count > 0);

    return array->data;
}

static inline void* i3_array_back(i3_array_t* array)
{
    assert(array != NULL);

    if (array->count == 0)
        return NULL;

    return i3_array_at(array, array->count - 1);
}

static inline void i3_array_clear(i3_array_t* array)
{
    assert(array != NULL);

    array->count = 0;
}

static inline void* i3_array_data(i3_array_t* array)
{
    assert(array != NULL);

    return array->data;
}

static inline uint32_t i3_array_count(i3_array_t* array)
{
    assert(array != NULL);

    return array->count;
}

static inline uint32_t i3_array_capacity(i3_array_t* array)
{
    assert(array != NULL);

    return array->capacity;
}

static inline uint32_t i3_array_element_size(i3_array_t* array)
{
    assert(array != NULL);

    return array->element_size;
}

static inline uint32_t i3_array_index_of(i3_array_t* array, void* element)
{
    assert(array != NULL);
    assert(element != NULL);

    uintptr_t index = ((uintptr_t)element - (uintptr_t)array->data) / array->element_size;

    if (index < array->count)
        return (uint32_t)index;

    return UINT32_MAX;
}
