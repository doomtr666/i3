#include "submission.h"
#include "buffer.h"
#include "cmd_buffer.h"

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

    // create a fence for the submission
    VkFenceCreateInfo fence_ci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    i3_vk_check(vkCreateFence(device->handle, &fence_ci, NULL, &submission->completion_fence));

    return submission;
}

void i3_vk_free_submission(i3_vk_device_o* device, i3_vk_submission_t* submission)
{
    assert(device != NULL);
    assert(submission != NULL);

    // release command buffers
    for (uint32_t i = 0; i < submission->cmd_buffer_count; ++i)
    {
        i3_rbk_cmd_buffer_i* cmd_buffer = submission->cmd_buffers[i];
        assert(cmd_buffer != NULL);

        // free the command buffer
        vkFreeCommandBuffers(device->handle, device->cmd_pool, 1, &submission->command_buffers[i]);

        // release the command buffer
        i3_rbk_resource_release(cmd_buffer);
    }

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

    i3_vk_device_o* device = (i3_vk_device_o*)self;

    // create submissions grouping I3_VK_SUBMISSION_CAPACITY command buffers
    for (uint32_t i = 0; i < cmd_buffer_count; i += I3_VK_SUBMISSION_CAPACITY)
    {
        // allocate a submission
        i3_vk_submission_t* submission = i3_vk_alloc_submission(device);

        // create command buffers for the submission
        for (uint32_t j = 0; j < I3_VK_SUBMISSION_CAPACITY && (i + j) < cmd_buffer_count; ++j)
        {
            uint32_t index = i + j;
            i3_rbk_cmd_buffer_i* cmd_buffer = cmd_buffers[index];

            // retain the command buffer
            submission->cmd_buffers[j] = cmd_buffer;
            i3_rbk_resource_add_ref(cmd_buffer);
            submission->cmd_buffer_count = j + 1;

            // create the command buffer
            VkCommandBufferAllocateInfo alloc_info = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = device->cmd_pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            };
            vkAllocateCommandBuffers(device->handle, &alloc_info, &submission->command_buffers[j]);

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
