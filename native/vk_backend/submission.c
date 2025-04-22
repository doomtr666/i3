#include "submission.h"
#include "buffer.h"
#include "cmd_buffer.h"
#include "swapchain.h"

#include <stdio.h>

typedef struct i3_vk_cmd_ctx_t
{
    i3_logger_i* logger;
    VkDevice device;
    VkCommandBuffer cmd_buffer;
} i3_vk_cmd_ctx_t;

void i3_vk_cmd_decode_copy_buffer(void* ctx, i3_vk_cmd_copy_buffer_t* cmd)
{
    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;
    assert(cmd_ctx != NULL);

    VkBufferCopy copy_region = {
        .srcOffset = cmd->src_offset,
        .dstOffset = cmd->dst_offset,
        .size = cmd->size,
    };

    i3_vk_buffer_o* src_buffer = (i3_vk_buffer_o*)cmd->src_buffer->self;
    i3_vk_buffer_o* dst_buffer = (i3_vk_buffer_o*)cmd->dst_buffer->self;

    vkCmdCopyBuffer(cmd_ctx->cmd_buffer, src_buffer->handle, dst_buffer->handle, 1, &copy_region);
}

i3_vk_submission_t* i3_vk_alloc_submission(i3_vk_device_o* device)
{
    i3_vk_submission_t* submission = i3_memory_pool_alloc(&device->submission_pool);
    assert(submission != NULL);

    // command buffer count
    submission->cmd_buffer_count = 0;

    // initialize use list
    i3_vk_use_list_init(&submission->use_list, device);

    // create a fence for the submission
    VkFenceCreateInfo fence_ci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    i3_vk_check(vkCreateFence(device->handle, &fence_ci, NULL, &submission->completion_fence));

    return submission;
}

void i3_vk_free_submission(i3_vk_device_o* device, i3_vk_submission_t* submission)
{
    assert(device != NULL);
    assert(submission != NULL);

    // release retained resources
    i3_vk_use_list_destroy(&submission->use_list);

    // free the command buffer
    vkFreeCommandBuffers(device->handle, device->cmd_pool, submission->cmd_buffer_count, submission->command_buffers);

    // destroy the fence
    vkDestroyFence(device->handle, submission->completion_fence, NULL);

    // free the submission
    i3_memory_pool_free(&device->submission_pool, submission);
}

void i3_vk_device_submit_cmd_buffers(i3_rbk_device_o* self,
                                     i3_rbk_cmd_buffer_i** cmd_buffers,
                                     uint32_t cmd_buffer_count)
{
    assert(self != NULL);
    assert(cmd_buffers != NULL);
    assert(cmd_buffer_count > 0);
    assert(cmd_buffer_count <= I3_VK_SUBMISSION_CAPACITY);

    i3_vk_device_o* device = (i3_vk_device_o*)self;

    // allocate a submission
    i3_vk_submission_t* submission = i3_vk_alloc_submission(device);

    // allocate command buffers
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = device->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = cmd_buffer_count,
    };
    i3_vk_check(vkAllocateCommandBuffers(device->handle, &alloc_info, submission->command_buffers));
    submission->cmd_buffer_count = cmd_buffer_count;

    // decode command buffers for the submission
    for (uint32_t j = 0; j < cmd_buffer_count; ++j)
    {
        i3_rbk_cmd_buffer_i* cmd_buffer = cmd_buffers[j];

        // retain the command buffer
        i3_vk_use_list_add(&submission->use_list, cmd_buffer);

        // decode the command buffer
        i3_vk_cmd_buffer_o* vk_cmd_buffer = (i3_vk_cmd_buffer_o*)cmd_buffer->self;
        i3_vk_cmd_list_t* cmd_list = &vk_cmd_buffer->cmd_list;

        i3_vk_cmd_ctx_t ctx = {
            .logger = device->log,
            .device = device->handle,
            .cmd_buffer = submission->command_buffers[j],
        };

        VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        };
        vkBeginCommandBuffer(ctx.cmd_buffer, &begin_info);
        i3_vk_cmd_decode(&ctx, cmd_list);
        vkEndCommandBuffer(ctx.cmd_buffer);
    }

    // submit the command buffers
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = submission->cmd_buffer_count,
        .pCommandBuffers = submission->command_buffers,
    };

    vkQueueSubmit(device->graphics_queue, 1, &submit_info, submission->completion_fence);

    // add the submission to the array
    i3_array_push(&device->submissions, &submission);
}

void i3_vk_device_end_frame(i3_rbk_device_o* self)
{
    assert(self != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;

    uint32_t i = 0;
    while (i < i3_array_count(&device->submissions))
    {
        i3_vk_submission_t* submission = *(i3_vk_submission_t**)i3_array_at(&device->submissions, i);
        assert(submission != NULL);

        // i3_log_dbg(device->log, "Checking submission %p", submission);

        if (vkGetFenceStatus(device->handle, submission->completion_fence) == VK_SUCCESS)
        {
            // i3_log_dbg(device->log, "Submission %p completed", submission);

            // swap last with the current one
            i3_vk_submission_t** submissions = i3_array_data(&device->submissions);
            submissions[i] = submissions[i3_array_count(&device->submissions) - 1];

            // remove last
            i3_array_pop(&device->submissions);

            // free the submission
            i3_vk_free_submission(device, submission);
        }
        else
        {
            ++i;
        }
    }
}

void i3_vk_device_present(i3_rbk_device_o* self, i3_rbk_swapchain_i* swapchain, i3_rbk_image_view_i* image_view)
{
    assert(self != NULL);
    assert(swapchain != NULL);
    assert(image_view != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;
    i3_vk_swapchain_o* vk_swapchain = (i3_vk_swapchain_o*)swapchain->self;

    // acquire swapchain image
    uint32_t image_index = i3_vk_swapchain_acquire_image(vk_swapchain);

    // present swapchain image
    i3_vk_swapchain_present(vk_swapchain, image_index);
}
