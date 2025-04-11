#pragma once

#include "native/core/common.h"

#define I3_RENDER_WINDOW_LOGGER_NAME "render_window"

typedef struct i3_render_window_i i3_render_window_i;
typedef struct i3_render_window_o i3_render_window_o;

struct i3_render_window_i
{
    i3_render_window_o* self;

    // vulkan surface
    void* (*get_vk_surface)(i3_render_window_o* self);

    void* (*get_native_handle)(i3_render_window_o* self);
    bool (*should_close)(i3_render_window_o* self);
    void (*destroy)(i3_render_window_o* self);
};

I3_EXPORT i3_render_window_i* i3_render_window_create_vulkan(void* vk_instance, const char* title, uint32_t width, uint32_t height);
I3_EXPORT void i3_render_window_poll_events();

// vulkan specific
I3_EXPORT const char** i3_render_window_get_required_vk_instance_extensions(uint32_t* count);
