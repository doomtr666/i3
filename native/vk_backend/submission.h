#pragma

#include "device.h"

#define I3_VK_SUBMISSION_CAPACITY 16

typedef struct i3_vk_submission_t
{
    // retained command buffers
    i3_rbk_cmd_buffer_i* cmd_buffers[I3_VK_SUBMISSION_CAPACITY];

    // generated command buffers
    VkCommandBuffer command_buffers[I3_VK_SUBMISSION_CAPACITY];

    // command buffer count
    uint32_t cmd_buffer_count;

    VkFence completion_fence;
} i3_vk_submission_t;

void i3_vk_device_submit_cmd_buffers(i3_rbk_device_o* self,
                                     i3_rbk_cmd_buffer_i** cmd_buffers,
                                     uint32_t cmd_buffer_count);

void i3_vk_device_end_frame(i3_rbk_device_o* self);
