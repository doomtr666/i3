#pragma

#include "device.h"
#include "use_list.h"

#define I3_VK_SUBMISSION_CAPACITY 256

typedef struct i3_vk_submission_t
{
    // retained resource
    i3_vk_use_list_t use_list;

    // generated command buffers
    VkCommandBuffer command_buffers[I3_VK_SUBMISSION_CAPACITY];
    uint32_t cmd_buffer_count;

    // fence for submission completion
    VkFence completion_fence;
} i3_vk_submission_t;

void i3_vk_device_submit_cmd_buffers(i3_rbk_device_o* self,
                                     i3_rbk_cmd_buffer_i** cmd_buffers,
                                     uint32_t cmd_buffer_count);

void i3_vk_device_end_frame(i3_rbk_device_o* self);

void i3_vk_device_present(i3_rbk_device_o* self, i3_rbk_swapchain_i* swapchain, i3_rbk_image_view_i* image_view);
