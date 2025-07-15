#pragma once

#include <stdint.h>

#define I3_CONTENT_STORE_LOGGER_NAME "content_store"

typedef struct i3_content_o i3_content_o;

typedef struct i3_content_i
{
    i3_content_o* self;

    const void* (*get_data)(i3_content_o* self);
    uint32_t (*get_size)(i3_content_o* self);

    void (*add_ref)(i3_content_o* self);
    void (*release)(i3_content_o* self);

} i3_content_i;

typedef struct i3_content_store_o i3_content_store_o;

typedef struct i3_content_store_i
{
    i3_content_store_o* self;

    i3_content_i* (*load)(i3_content_store_o* self, const char* path);

    void (*destroy)(i3_content_store_o* self);

} i3_content_store_i;

i3_content_store_i* i3_content_store_create();
