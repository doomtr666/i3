#pragma once

#include "native/core/array.h"

#include "device.h"

// buffer barrier info

typedef struct i3_vk_buffer_barrier_info_t
{
    uint32_t queue_family_index;
    VkPipelineStageFlags stage_mask;
    VkAccessFlags access_mask;
    VkDeviceSize offset;
    VkDeviceSize size;
} i3_vk_buffer_barrier_info_t;

static inline void i3_vk_init_buffer_barrier_info(i3_vk_buffer_barrier_info_t* info, VkDeviceSize size)
{
    assert(info != NULL);

    *info = (i3_vk_buffer_barrier_info_t){
        .queue_family_index = VK_QUEUE_FAMILY_IGNORED,
        .stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        .size = size,
    };
}

static inline void i3_vk_destroy_buffer_barrier_info(i3_vk_buffer_barrier_info_t* info)
{
    assert(info != NULL);
}

// image barrier info

typedef struct i3_vk_sub_image_barrier_info_t
{
    uint32_t queue_family_index;
    VkPipelineStageFlags stage_mask;
    VkAccessFlags access_mask;
    VkImageLayout layout;
} i3_vk_sub_image_barrier_info_t;

typedef struct i3_vk_image_barrier_info_t
{
    i3_array_t sub_image_barrier_infos;
} i3_vk_image_barrier_info_t;

static inline void i3_vk_init_image_barrier_info(i3_vk_image_barrier_info_t* info, uint32_t count)
{
    assert(info != NULL);

    i3_array_init_count(&info->sub_image_barrier_infos, sizeof(i3_vk_sub_image_barrier_info_t), count);
    i3_vk_sub_image_barrier_info_t* sub_img_infos =
        (i3_vk_sub_image_barrier_info_t*)i3_array_data(&info->sub_image_barrier_infos);

    for (uint32_t i = 0; i < count; ++i)
    {
        sub_img_infos[i] = (i3_vk_sub_image_barrier_info_t){
            .queue_family_index = VK_QUEUE_FAMILY_IGNORED,
            .stage_mask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .layout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
    }
}

static inline void i3_vk_destroy_image_barrier_info(i3_vk_image_barrier_info_t* info)
{
    assert(info != NULL);

    i3_array_destroy(&info->sub_image_barrier_infos);
}