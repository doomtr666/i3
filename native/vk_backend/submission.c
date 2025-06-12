#include "submission.h"
#include "buffer.h"
#include "cmd_buffer.h"
#include "framebuffer.h"
#include "image.h"
#include "image_view.h"
#include "pipeline.h"
#include "swapchain.h"

typedef struct i3_vk_cmd_ctx_t
{
    i3_logger_i* logger;
    VkDevice device;
    VkCommandBuffer cmd_buffer;
} i3_vk_cmd_ctx_t;

// barrier
void i3_vk_cmd_decode_barrier(void* ctx, i3_vk_cmd_barrier_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    for (uint32_t i = 0; i < cmd->barrier.vk_subimage_barrier_count; i++)
    {
        vkCmdPipelineBarrier(cmd_ctx->cmd_buffer, cmd->barrier.vk_subimage_barriers[i].src_stage_mask,
                             cmd->barrier.dst_stage_mask, 0, 0, NULL, 0, NULL, 1,
                             &cmd->barrier.vk_subimage_barriers[i].vk_image_barrier);
    }
}

void i3_vk_cmd_decode_clear_color_image(void* ctx, i3_vk_cmd_clear_color_image_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdClearColorImage(cmd_ctx->cmd_buffer, cmd->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cmd->value, 1,
                         &cmd->subresource_range);
}

void i3_vk_cmd_decode_clear_depth_stencil_image(void* ctx, i3_vk_cmd_clear_depth_stencil_image_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdClearDepthStencilImage(cmd_ctx->cmd_buffer, cmd->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cmd->value, 1,
                                &cmd->subresource_range);
}

// copy buffer
void i3_vk_cmd_decode_copy_buffer(void* ctx, i3_vk_cmd_copy_buffer_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    VkBufferCopy copy_region = {
        .srcOffset = cmd->src_offset,
        .dstOffset = cmd->dst_offset,
        .size = cmd->size,
    };

    vkCmdCopyBuffer(cmd_ctx->cmd_buffer, cmd->src_buffer, cmd->dst_buffer, 1, &copy_region);
}

// bind vertex buffers
void i3_vk_cmd_decode_bind_vertex_buffers(void* ctx, i3_vk_cmd_bind_vertex_buffers_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdBindVertexBuffers(cmd_ctx->cmd_buffer, cmd->first_binding, cmd->binding_count, cmd->buffers, cmd->offsets);
}

// bind index buffer
void i3_vk_cmd_decode_bind_index_buffer(void* ctx, i3_vk_cmd_bind_index_buffer_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdBindIndexBuffer(cmd_ctx->cmd_buffer, cmd->buffer, cmd->offset, cmd->index_type);
}

// bind descriptor sets
void i3_vk_cmd_decode_bind_descriptor_sets(void* ctx, i3_vk_cmd_bind_descriptor_sets_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdBindDescriptorSets(cmd_ctx->cmd_buffer, cmd->bind_point, cmd->layout, cmd->first_set,
                            cmd->descriptor_set_count, cmd->descriptor_sets, 0, NULL);
}

// bind pipeline
void i3_vk_cmd_decode_bind_pipeline(void* ctx, i3_vk_cmd_bind_pipeline_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdBindPipeline(cmd_ctx->cmd_buffer, cmd->bind_point, cmd->pipeline);
}

// set viewports
void i3_vk_cmd_decode_set_viewports(void* ctx, i3_vk_cmd_set_viewports_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdSetViewport(cmd_ctx->cmd_buffer, cmd->first_viewport, cmd->viewport_count, cmd->viewports);
}

// set scissor
void i3_vk_cmd_decode_set_scissors(void* ctx, i3_vk_cmd_set_scissors_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdSetScissor(cmd_ctx->cmd_buffer, cmd->first_scissor, cmd->scissor_count, cmd->scissors);
}

// begin rendering
void i3_vk_cmd_decode_begin_rendering(void* ctx, i3_vk_cmd_begin_rendering_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    VkRenderPassBeginInfo render_pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = cmd->render_pass,
        .framebuffer = cmd->framebuffer,
        .renderArea = cmd->render_area,
    };

    vkCmdBeginRenderPass(cmd_ctx->cmd_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
}

// end rendering
void i3_vk_cmd_decode_end_rendering(void* ctx, i3_vk_cmd_end_rendering_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdEndRenderPass(cmd_ctx->cmd_buffer);
}

// push constants
void i3_vk_cmd_decode_push_constants(void* ctx, i3_vk_cmd_push_constants_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdPushConstants(cmd_ctx->cmd_buffer, cmd->layout, cmd->stage_flags, cmd->offset, cmd->size, cmd->data);
}

// draw
void i3_vk_cmd_decode_draw(void* ctx, i3_vk_cmd_draw_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdDraw(cmd_ctx->cmd_buffer, cmd->vertex_count, cmd->instance_count, cmd->first_vertex, cmd->first_instance);
}

// draw indexed
void i3_vk_cmd_decode_draw_indexed(void* ctx, i3_vk_cmd_draw_indexed_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdDrawIndexed(cmd_ctx->cmd_buffer, cmd->index_count, cmd->instance_count, cmd->first_index, cmd->vertex_offset,
                     cmd->first_instance);
}

// draw indirect
void i3_vk_cmd_decode_draw_indirect(void* ctx, i3_vk_cmd_draw_indirect_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdDrawIndirect(cmd_ctx->cmd_buffer, cmd->buffer, cmd->offset, cmd->draw_count, cmd->stride);
}

// draw indexed indirect
void i3_vk_cmd_decode_draw_indexed_indirect(void* ctx, i3_vk_cmd_draw_indexed_indirect_t* cmd)
{
    assert(ctx != NULL);
    assert(cmd != NULL);

    i3_vk_cmd_ctx_t* cmd_ctx = (i3_vk_cmd_ctx_t*)ctx;

    vkCmdDrawIndexedIndirect(cmd_ctx->cmd_buffer, cmd->buffer, cmd->offset, cmd->draw_count, cmd->stride);
}

i3_vk_submission_t* i3_vk_submission_alloc(i3_vk_device_o* device)
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

void i3_vk_submission_free(i3_vk_device_o* device, i3_vk_submission_t* submission)
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

void i3_vk_submission_resolve_barriers(i3_vk_submission_t* submission, i3_rbk_cmd_buffer_i* cmd_buffer)
{
    assert(submission != NULL);
    assert(cmd_buffer != NULL);

    i3_vk_cmd_buffer_o* buffer = (i3_vk_cmd_buffer_o*)cmd_buffer->self;
    i3_vk_barrier_t* barrier = NULL;

    i3_dlist_foreach(&buffer->barriers, barrier)
    {
        // resolve buffer barriers
        // TODO:

        // resolve image barriers
        for (uint32_t i = 0; i < barrier->image_usage_count; ++i)
        {
            i3_vk_image_usage_t* image_barrier = &barrier->image_usages[i];

            // get image view
            i3_vk_image_view_o* image_view = (i3_vk_image_view_o*)image_barrier->image_view->self;
            i3_vk_image_o* image = image_view->image;

            // check if barrier is required
            for (uint32_t j = image_view->desc.base_array_layer;
                 j < image_view->desc.base_array_layer + image_view->desc.layer_count; j++)
            {
                // foreach level
                for (uint32_t k = image_view->desc.base_mip_level;
                     k < image_view->desc.base_mip_level + image_view->desc.level_count; k++)
                {
                    // compare subimage state
                    i3_vk_subimage_state_t* subimage_state = i3_vk_get_subimage_state(&image->state, j, k);

                    if (subimage_state->queue_family_index != image_barrier->queue_family_index
                        || subimage_state->layout != image_barrier->layout
                        || subimage_state->access_mask != image_barrier->access_mask
                        || subimage_state->stage_mask != barrier->dst_stage_mask)
                    {
                        assert(barrier->vk_subimage_barrier_count < I3_VK_BARRIER_MAX_RESOURCE_COUNT);
                        i3_vk_subimage_barrier_t* resolved_barrier
                            = &barrier->vk_subimage_barriers[barrier->vk_subimage_barrier_count++];
                        *resolved_barrier = (i3_vk_subimage_barrier_t)
                        {
                            .src_stage_mask = subimage_state->stage_mask,
                            .vk_image_barrier = 
                            {
                                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                .srcAccessMask = subimage_state->access_mask,
                                .dstAccessMask = image_barrier->access_mask,
                                .oldLayout = subimage_state->layout,
                                .newLayout = image_barrier->layout,
                                .srcQueueFamilyIndex = subimage_state->queue_family_index,
                                .dstQueueFamilyIndex = image_barrier->queue_family_index,
                                .image = image->handle,
                                .subresourceRange = 
                                {
                                    .aspectMask = image_view->desc.aspect_mask,
                                    .baseMipLevel = image_view->desc.base_mip_level,
                                    .levelCount = image_view->desc.level_count,
                                    .baseArrayLayer = j,
                                    .layerCount = 1,
                                },
                            }
                        };

                        // update subimage state
                        subimage_state->queue_family_index = image_barrier->queue_family_index;
                        subimage_state->layout = image_barrier->layout;
                        subimage_state->access_mask = image_barrier->access_mask;
                        subimage_state->stage_mask = barrier->dst_stage_mask;
                    }
                }
            }
        }
    }
}

void i3_vk_device_submit_cmd_buffers(i3_rbk_device_o* self,
                                     uint32_t cmd_buffer_count,
                                     i3_rbk_cmd_buffer_i** cmd_buffers)
{
    assert(self != NULL);
    assert(cmd_buffers != NULL);
    assert(cmd_buffer_count > 0);
    assert(cmd_buffer_count <= I3_VK_SUBMISSION_CAPACITY);

    i3_vk_device_o* device = (i3_vk_device_o*)self;

    // allocate a submission
    i3_vk_submission_t* submission = i3_vk_submission_alloc(device);

    // resolve barriers: this loop has a dependency on the order of command buffers in the submission
    for (uint32_t j = 0; j < cmd_buffer_count; ++j)
    {
        i3_rbk_cmd_buffer_i* cmd_buffer = cmd_buffers[j];

        // retain the command buffer
        i3_vk_use_list_add(&submission->use_list, cmd_buffer);

        // resolve cmd_buffer barriers
        i3_vk_submission_resolve_barriers(submission, cmd_buffer);
    }

    // allocate vk command buffers
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = device->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = cmd_buffer_count,
    };
    i3_vk_check(vkAllocateCommandBuffers(device->handle, &alloc_info, submission->command_buffers));
    submission->cmd_buffer_count = cmd_buffer_count;

    // decode command buffers for the submission : this loop can be parallelized
    for (uint32_t j = 0; j < cmd_buffer_count; ++j)
    {
        i3_rbk_cmd_buffer_i* cmd_buffer = cmd_buffers[j];

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

        if (vkGetFenceStatus(device->handle, submission->completion_fence) == VK_SUCCESS)
        {
            // swap last with the current one
            i3_vk_submission_t** submissions = i3_array_data(&device->submissions);
            submissions[i] = submissions[i3_array_count(&device->submissions) - 1];

            // remove last
            i3_array_pop(&device->submissions);

            // free the submission
            i3_vk_submission_free(device, submission);
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
    if (image_index == UINT32_MAX)
        return;

    // allocate submission
    i3_vk_submission_t* submission = i3_vk_submission_alloc(device);

    // retain swapchain and image view
    i3_vk_use_list_add(&submission->use_list, swapchain);
    i3_vk_use_list_add(&submission->use_list, image_view);

    // allocate command buffer
    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = device->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    i3_vk_check(vkAllocateCommandBuffers(device->handle, &alloc_info, &submission->command_buffers[0]));
    submission->cmd_buffer_count = 1;
    VkCommandBuffer cmd_buffer = submission->command_buffers[0];

    // begin command buffer recording
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };

    vkBeginCommandBuffer(cmd_buffer, &begin_info);

    // get source image
    i3_vk_image_view_o* vk_image_view = (i3_vk_image_view_o*)image_view->self;
    i3_vk_image_o* src_image = vk_image_view->image;

    // get destination image
    VkImage dst_image = vk_swapchain->images[image_index];

    // transition src_image to TRANSFER_SRC
    const i3_rbk_image_view_desc_t* view_info = image_view->get_desc(image_view->self);
    i3_vk_subimage_state_t* src_subimage_info
        = i3_vk_get_subimage_state(&src_image->state, view_info->base_array_layer, view_info->base_mip_level);
    VkImageMemoryBarrier src_barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = src_subimage_info->access_mask,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = src_subimage_info->layout,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .image = src_image->handle,
        .subresourceRange.aspectMask = view_info->aspect_mask,
        .subresourceRange.baseArrayLayer = view_info->base_array_layer,
        .subresourceRange.layerCount = 1,
        .subresourceRange.baseMipLevel = view_info->base_mip_level,
        .subresourceRange.levelCount = 1,
    };

    vkCmdPipelineBarrier(cmd_buffer, src_subimage_info->stage_mask, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                         1, &src_barrier);

    // update image barrier info
    src_subimage_info->access_mask = VK_ACCESS_TRANSFER_READ_BIT;
    src_subimage_info->stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    src_subimage_info->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    // transition dst_image to TRANSFER_DST
    VkImageMemoryBarrier dst_barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .image = dst_image,
        .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .subresourceRange.levelCount = 1,
        .subresourceRange.layerCount = 1,
    };

    vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                         &dst_barrier);

    // blit image to swapchain image
    VkImageBlit region = {
        .srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .srcSubresource.layerCount = 1,
        .srcOffsets[1].x = src_image->desc.width,
        .srcOffsets[1].y = src_image->desc.height,
        .srcOffsets[1].z = 1,

        .dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .dstSubresource.layerCount = 1,
        .dstOffsets[1].x = vk_swapchain->extent.width,
        .dstOffsets[1].y = vk_swapchain->extent.height,
        .dstOffsets[1].z = 1,
    };

    vkCmdBlitImage(cmd_buffer, src_image->handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);

    // transition dst_image to PRESENT_SRC
    VkImageMemoryBarrier present_barrier = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                            .dstAccessMask = 0,
                                            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                            .image = dst_image,
                                            .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                            .subresourceRange.levelCount = 1,
                                            .subresourceRange.layerCount = 1};

    vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 0, NULL, 1,
                         &present_barrier);

    // end command buffer recording
    vkEndCommandBuffer(cmd_buffer);

    // submit the command buffer
    VkPipelineStageFlags wait_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSemaphore acquire_sem = i3_vk_swapchain_get_acquire_semaphore(vk_swapchain);
    VkSemaphore present_sem = i3_vk_swapchain_get_present_semaphore(vk_swapchain);
    VkSemaphore signal_sems[] = {i3_vk_swapchain_get_present_semaphore(vk_swapchain)};
    VkSubmitInfo submit_info = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .waitSemaphoreCount = 1,
                                .pWaitDstStageMask = &wait_mask,
                                .pWaitSemaphores = &acquire_sem,
                                .commandBufferCount = 1,
                                .pCommandBuffers = &cmd_buffer,
                                .signalSemaphoreCount = 1,
                                .pSignalSemaphores = &present_sem};

    vkQueueSubmit(device->graphics_queue, 1, &submit_info, submission->completion_fence);

    // present swapchain image
    i3_vk_swapchain_present(vk_swapchain, image_index);

    // add the submission to the array
    i3_array_push(&device->submissions, &submission);
}
