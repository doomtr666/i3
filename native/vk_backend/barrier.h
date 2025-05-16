#pragma once

#include "native/core/array.h"

#include "device.h"

#define I3_VK_BARRIER_MAX_RESOURCE_COUNT 16
#define I3_VK_BARRIER_MAX_SUBIMAGE_COUNT 256

// buffer usage

typedef struct i3_vk_buffer_state_t
{
    uint32_t queue_family_index;
    VkPipelineStageFlags stage_mask;
    VkAccessFlags access_mask;
    VkDeviceSize offset;
    VkDeviceSize size;
} i3_vk_buffer_state_t;

static inline void i3_vk_init_buffer_state(i3_vk_buffer_state_t* state, VkDeviceSize size);
static inline void i3_vk_destroy_buffer_state(i3_vk_buffer_state_t* state);

// image usage

typedef struct i3_vk_subimage_state_t
{
    uint32_t queue_family_index;
    VkPipelineStageFlags stage_mask;
    VkAccessFlags access_mask;
    VkImageLayout layout;
} i3_vk_subimage_state_t;

typedef struct i3_vk_image_state_t
{
    uint32_t layers;
    uint32_t levels;
    i3_array_t sub_image_barrier_infos;
} i3_vk_image_state_t;

static inline void i3_vk_init_image_state(i3_vk_image_state_t* state, uint32_t layers, uint32_t levels);
static inline void i3_vk_destroy_image_state(i3_vk_image_state_t* state);
static inline i3_vk_subimage_state_t* i3_vk_get_subimage_state(i3_vk_image_state_t* state,
                                                               uint32_t layer,
                                                               uint32_t level);
// barrier

typedef struct i3_vk_buffer_usage_t
{
    uint32_t queue_family_index;
    VkAccessFlags access_mask;
    i3_rbk_buffer_i* buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
} i3_vk_buffer_usage_t;

typedef struct i3_vk_image_usage_t
{
    uint32_t queue_family_index;
    VkAccessFlags access_mask;
    i3_rbk_image_view_i* image_view;
    VkImageLayout layout;
} i3_vk_image_usage_t;

typedef struct i3_vk_buffer_barrier_t
{
    VkPipelineStageFlags src_stage_mask;
    VkBufferMemoryBarrier vk_buffer_barrier;
} i3_vk_buffer_barrier_t;

typedef struct i3_vk_subimage_barrier_t
{
    VkPipelineStageFlags src_stage_mask;
    VkImageMemoryBarrier vk_image_barrier;
} i3_vk_subimage_barrier_t;

typedef struct i3_vk_barrier_t
{
    // chain
    struct i3_vk_barrier_t* next;
    struct i3_vk_barrier_t* prev;

    // stage mask
    VkPipelineStageFlags dst_stage_mask;

    // buffer barriers
    uint32_t buffer_usage_count;
    i3_vk_buffer_usage_t buffer_usages[I3_VK_BARRIER_MAX_RESOURCE_COUNT];

    // image barriers
    uint32_t image_usage_count;
    i3_vk_image_usage_t image_usages[I3_VK_BARRIER_MAX_RESOURCE_COUNT];

    // vk barriers
    uint32_t vk_buffer_barrier_count;
    i3_vk_buffer_barrier_t vk_buffer_barriers[I3_VK_BARRIER_MAX_RESOURCE_COUNT];
    uint32_t vk_subimage_barrier_count;
    i3_vk_subimage_barrier_t vk_subimage_barriers[I3_VK_BARRIER_MAX_SUBIMAGE_COUNT];

} i3_vk_barrier_t;

static inline void i3_vk_barrier_init(i3_vk_barrier_t* barrier, VkPipelineStageFlags stage_mask);
static inline i3_vk_buffer_usage_t* i3_vk_add_buffer_barrier(i3_vk_barrier_t* barrier);
static inline i3_vk_image_usage_t* i3_vk_add_image_barrier(i3_vk_barrier_t* barrier);

// implementation

static inline void i3_vk_init_buffer_state(i3_vk_buffer_state_t* state, VkDeviceSize size)
{
    assert(state != NULL);

    *state = (i3_vk_buffer_state_t){
        .queue_family_index = VK_QUEUE_FAMILY_IGNORED,
        .stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        .size = size,
    };
}

static inline void i3_vk_destroy_buffer_state(i3_vk_buffer_state_t* state)
{
    assert(state != NULL);
}

static inline void i3_vk_init_image_state(i3_vk_image_state_t* state, uint32_t layers, uint32_t levels)
{
    assert(state != NULL);
    assert(layers > 0);
    assert(levels > 0);

    uint32_t count = layers * levels;
    state->layers = layers;
    state->levels = levels;

    i3_array_init_count(&state->sub_image_barrier_infos, sizeof(i3_vk_subimage_state_t), count);
    i3_vk_subimage_state_t* sub_img_infos = (i3_vk_subimage_state_t*)i3_array_data(&state->sub_image_barrier_infos);

    for (uint32_t i = 0; i < count; ++i)
    {
        sub_img_infos[i] = (i3_vk_subimage_state_t){
            .queue_family_index = VK_QUEUE_FAMILY_IGNORED,
            .stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .layout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
    }
}

static inline void i3_vk_destroy_image_state(i3_vk_image_state_t* state)
{
    assert(state != NULL);

    i3_array_destroy(&state->sub_image_barrier_infos);
}

static inline i3_vk_subimage_state_t* i3_vk_get_subimage_state(i3_vk_image_state_t* state,
                                                               uint32_t layer,
                                                               uint32_t level)
{
    assert(state != NULL);
    assert(layer < state->layers);
    assert(level < state->levels);

    uint32_t index = layer * state->levels + level;
    return (i3_vk_subimage_state_t*)i3_array_at(&state->sub_image_barrier_infos, index);
}

static inline void i3_vk_barrier_init(i3_vk_barrier_t* barrier, VkPipelineStageFlags stage_mask)
{
    assert(barrier != NULL);

    barrier->next = NULL;
    barrier->prev = NULL;
    barrier->dst_stage_mask = stage_mask;
    barrier->buffer_usage_count = 0;
    barrier->image_usage_count = 0;
    barrier->vk_buffer_barrier_count = 0;
    barrier->vk_subimage_barrier_count = 0;
}

static inline i3_vk_buffer_usage_t* i3_vk_add_buffer_barrier(i3_vk_barrier_t* barrier)
{
    assert(barrier != NULL);

    assert(barrier->buffer_usage_count < I3_VK_BARRIER_MAX_RESOURCE_COUNT);
    i3_vk_buffer_usage_t* buffer_barrier = &barrier->buffer_usages[barrier->buffer_usage_count++];
    return buffer_barrier;
}

static inline i3_vk_image_usage_t* i3_vk_add_image_barrier(i3_vk_barrier_t* barrier)
{
    assert(barrier != NULL);

    assert(barrier->image_usage_count < I3_VK_BARRIER_MAX_RESOURCE_COUNT);
    i3_vk_image_usage_t* image_barrier = &barrier->image_usages[barrier->image_usage_count++];
    return image_barrier;
}
