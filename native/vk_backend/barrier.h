#pragma once

#include "native/core/array.h"

#include "device.h"

#define I3_VK_BARRIER_MAX_RESOURCE_COUNT 16

// buffer usage

typedef struct i3_vk_buffer_usage_t
{
    uint32_t queue_family_index;
    VkPipelineStageFlags stage_mask;
    VkAccessFlags access_mask;
    VkDeviceSize offset;
    VkDeviceSize size;
} i3_vk_buffer_usage_t;

static inline void i3_vk_init_buffer_usage(i3_vk_buffer_usage_t* usage, VkDeviceSize size);
static inline void i3_vk_destroy_buffer_usage(i3_vk_buffer_usage_t* usage);

// image usage

typedef struct i3_vk_subimage_usage_t
{
    uint32_t queue_family_index;
    VkPipelineStageFlags stage_mask;
    VkAccessFlags access_mask;
    VkImageLayout layout;
} i3_vk_subimage_usage_t;

typedef struct i3_vk_image_usage_t
{
    i3_array_t sub_image_barrier_infos;
} i3_vk_image_usage_t;

static inline void i3_vk_init_image_usage(i3_vk_image_usage_t* info, uint32_t count);
static inline void i3_vk_destroy_image_usage(i3_vk_image_usage_t* info);

// barrier

typedef struct i3_vk_buffer_barrier_t
{
    uint32_t queue_family_index;
    VkAccessFlags access_mask;
    i3_rbk_buffer_i* buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
} i3_vk_buffer_barrier_t;

typedef struct i3_vk_image_barrier_t
{
    uint32_t queue_family_index;
    VkAccessFlags access_mask;
    i3_rbk_image_i* image;
    VkImageAspectFlags aspect_mask;
    uint32_t base_mip_level;
    uint32_t level_count;
    uint32_t base_array_layer;
    uint32_t layer_count;
    VkImageLayout layout;
} i3_vk_image_barrier_t;

typedef struct i3_vk_barrier_t
{
    // chain
    struct i3_vk_barrier_t* next;
    struct i3_vk_barrier_t* prev;

    // stage mask
    VkPipelineStageFlags stage_mask;

    // buffer barriers
    uint32_t buffer_barrier_count;
    i3_vk_buffer_barrier_t buffer_barriers[I3_VK_BARRIER_MAX_RESOURCE_COUNT];

    // image barriers
    uint32_t image_barrier_count;
    i3_vk_image_barrier_t image_barriers[I3_VK_BARRIER_MAX_RESOURCE_COUNT];

    // vk barriers
    uint32_t vk_buffer_barrier_count;
    VkBufferMemoryBarrier vk_buffer_barriers[I3_VK_BARRIER_MAX_RESOURCE_COUNT];
    uint32_t vk_image_barrier_count;
    VkImageMemoryBarrier vk_image_barriers[I3_VK_BARRIER_MAX_RESOURCE_COUNT];

} i3_vk_barrier_t;

static inline void i3_vk_barrier_init(i3_vk_barrier_t* barrier, VkPipelineStageFlags stage_mask);
static inline i3_vk_buffer_barrier_t* i3_vk_add_buffer_barrier(i3_vk_barrier_t* barrier);
static inline i3_vk_image_barrier_t* i3_vk_add_image_barrier(i3_vk_barrier_t* barrier);

// implementation

static inline void i3_vk_init_buffer_usage(i3_vk_buffer_usage_t* usage, VkDeviceSize size)
{
    assert(usage != NULL);

    *usage = (i3_vk_buffer_usage_t){
        .queue_family_index = VK_QUEUE_FAMILY_IGNORED,
        .stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        .size = size,
    };
}

static inline void i3_vk_destroy_buffer_usage(i3_vk_buffer_usage_t* usage)
{
    assert(usage != NULL);
}

static inline void i3_vk_init_image_usage(i3_vk_image_usage_t* info, uint32_t count)
{
    assert(info != NULL);

    i3_array_init_count(&info->sub_image_barrier_infos, sizeof(i3_vk_subimage_usage_t), count);
    i3_vk_subimage_usage_t* sub_img_infos = (i3_vk_subimage_usage_t*)i3_array_data(&info->sub_image_barrier_infos);

    for (uint32_t i = 0; i < count; ++i)
    {
        sub_img_infos[i] = (i3_vk_subimage_usage_t){
            .queue_family_index = VK_QUEUE_FAMILY_IGNORED,
            .stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .layout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
    }
}

static inline void i3_vk_destroy_image_usage(i3_vk_image_usage_t* info)
{
    assert(info != NULL);

    i3_array_destroy(&info->sub_image_barrier_infos);
}

static inline void i3_vk_barrier_init(i3_vk_barrier_t* barrier, VkPipelineStageFlags stage_mask)
{
    assert(barrier != NULL);

    barrier->next = NULL;
    barrier->prev = NULL;
    barrier->stage_mask = stage_mask;
    barrier->buffer_barrier_count = 0;
    barrier->image_barrier_count = 0;
    barrier->vk_buffer_barrier_count = 0;
    barrier->vk_image_barrier_count = 0;
}

static inline i3_vk_buffer_barrier_t* i3_vk_add_buffer_barrier(i3_vk_barrier_t* barrier)
{
    assert(barrier != NULL);

    assert(barrier->buffer_barrier_count < I3_VK_BARRIER_MAX_RESOURCE_COUNT);
    i3_vk_buffer_barrier_t* buffer_barrier = &barrier->buffer_barriers[barrier->buffer_barrier_count++];
    return buffer_barrier;
}

static inline i3_vk_image_barrier_t* i3_vk_add_image_barrier(i3_vk_barrier_t* barrier)
{
    assert(barrier != NULL);

    assert(barrier->image_barrier_count < I3_VK_BARRIER_MAX_RESOURCE_COUNT);
    i3_vk_image_barrier_t* image_barrier = &barrier->image_barriers[barrier->image_barrier_count++];
    return image_barrier;
}
