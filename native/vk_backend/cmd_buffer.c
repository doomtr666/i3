#include "cmd_buffer.h"

#include "buffer.h"
#include "convert.h"
#include "descriptor_set.h"
#include "framebuffer.h"
#include "image.h"
#include "image_view.h"
#include "pipeline.h"
#include "pipeline_layout.h"
#include "sampler.h"

// resource interface

static void i3_vk_cmd_buffer_add_ref(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    cmd_buffer->use_count++;
}

static void i3_vk_cmd_buffer_release(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    if (--cmd_buffer->use_count == 0)
    {
        // destroy the use list
        i3_vk_use_list_destroy(&cmd_buffer->use_list);

        // destroy the command list
        i3_vk_cmd_list_destroy(&cmd_buffer->cmd_list);

        // free the command buffer
        i3_memory_pool_free(&cmd_buffer->device->cmd_buffer_pool, cmd_buffer);
    }
}

static int32_t i3_vk_cmd_buffer_get_use_count(i3_rbk_resource_o* self)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    return cmd_buffer->use_count;
}

static void i3_vk_cmd_buffer_set_debug_name(i3_rbk_resource_o* self, const char* name)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // TODO:

    // i3_vk_set_debug_name(cmd_buffer->device->log, cmd_buffer->device->handle, VK_OBJECT_TYPE_COMMAND_BUFFER,
    // cmd_buffer->base.handle, name);
}

// cmd buffer interface

static i3_rbk_resource_i* i3_vk_cmd_buffer_get_resource(i3_rbk_cmd_buffer_o* self)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    return &cmd_buffer->base;
}

// barrier
static i3_vk_barrier_t* i3_vk_cmd_add_barriers(i3_vk_cmd_buffer_o* cmd_buffer, VkShaderStageFlagBits stage_mask)
{
    assert(cmd_buffer != NULL);

    i3_vk_cmd_barrier_t* cmd = i3_vk_cmd_write_barrier(&cmd_buffer->cmd_list);
    i3_vk_barrier_t* barrier = &cmd->barrier;
    i3_vk_barrier_init(barrier, stage_mask);
    i3_dlist_append(&cmd_buffer->barriers, barrier);

    return barrier;
}

// clear image
static void i3_vk_cmd_buffer_clear_image(i3_rbk_cmd_buffer_o* self,
                                         i3_rbk_image_view_i* image_view,
                                         const i3_rbk_clear_color_t* color)
{
    assert(self != NULL);
    assert(image_view != NULL);
    assert(color != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // retain the image view
    i3_vk_use_list_add(&cmd_buffer->use_list, image_view);

    // add barriers
    i3_vk_barrier_t* barriers = i3_vk_cmd_add_barriers(cmd_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT);
    i3_vk_image_usage_t* barrier = i3_vk_add_image_barrier(barriers);
    *barrier = (i3_vk_image_usage_t){
        .queue_family_index = VK_QUEUE_FAMILY_IGNORED,
        .access_mask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .image_view = image_view,
        .layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    };

    // emit command
    i3_rbk_image_i* image = image_view->get_image(image_view->self);
    const i3_rbk_image_view_desc_t* view_desc = image_view->get_desc(image_view->self);

    i3_vk_cmd_clear_image_t* cmd = i3_vk_cmd_write_clear_image(&cmd_buffer->cmd_list);
    cmd->image = ((i3_vk_image_o*)image->self)->handle;
    cmd->subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    cmd->subresource_range.baseArrayLayer = view_desc->base_array_layer;
    cmd->subresource_range.layerCount = view_desc->layer_count;
    cmd->subresource_range.baseMipLevel = view_desc->base_mip_level;
    cmd->subresource_range.levelCount = view_desc->level_count;
    cmd->color.uint32[0] = color->uint32[0];
    cmd->color.uint32[1] = color->uint32[1];
    cmd->color.uint32[2] = color->uint32[2];
    cmd->color.uint32[3] = color->uint32[3];
}

// copy buffer
static void i3_vk_cmd_buffer_copy_buffer(i3_rbk_cmd_buffer_o* self,
                                         i3_rbk_buffer_i* src_buffer,
                                         i3_rbk_buffer_i* dst_buffer,
                                         uint32_t src_offset,
                                         uint32_t dst_offset,
                                         uint32_t size)
{
    assert(self != NULL);
    assert(src_buffer != NULL);
    assert(dst_buffer != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // add resources to the use list
    i3_vk_use_list_add(&cmd_buffer->use_list, src_buffer);
    i3_vk_use_list_add(&cmd_buffer->use_list, dst_buffer);

    // emit command
    i3_vk_cmd_copy_buffer_t* cmd = i3_vk_cmd_write_copy_buffer(&cmd_buffer->cmd_list);
    cmd->src_buffer = ((i3_vk_buffer_o*)src_buffer->self)->handle;
    cmd->dst_buffer = ((i3_vk_buffer_o*)dst_buffer->self)->handle;
    cmd->src_offset = src_offset;
    cmd->dst_offset = dst_offset;
    cmd->size = size;
}

// write buffer
static void i3_vk_cmd_buffer_write_buffer(i3_rbk_cmd_buffer_o* self,
                                          i3_rbk_buffer_i* buffer,
                                          uint32_t dst_offset,
                                          uint32_t size,
                                          const void* data)
{
    assert(self != NULL);
    assert(buffer != NULL);
    assert(data != NULL);
    assert(size > 0);

    // TODO: use vkUpdateBuffer for small data transfers ?

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;
    i3_rbk_device_i* device = &cmd_buffer->device->iface;

    // create a temporary buffer of size
    i3_rbk_buffer_desc_t temp_buffer_desc = {
        .flags = I3_RBK_BUFFER_FLAG_STAGING,
        .size = size,
    };
    i3_rbk_buffer_i* staging_buffer = device->create_buffer(device->self, &temp_buffer_desc);

    // copy data to the temporary buffer
    void* dst = staging_buffer->map(staging_buffer->self);
    memcpy(dst, data, size);
    staging_buffer->unmap(staging_buffer->self);

    // copy the temporary buffer to the destination buffer
    i3_vk_cmd_buffer_copy_buffer((i3_rbk_cmd_buffer_o*)cmd_buffer, staging_buffer, buffer, 0, dst_offset, size);

    // destroy the temporary buffer
    staging_buffer->destroy(staging_buffer->self);
}

// bind vertex buffers
static void i3_vk_cmd_buffer_bind_vertex_buffers(i3_rbk_cmd_buffer_o* self,
                                                 uint32_t first_binding,
                                                 uint32_t binding_count,
                                                 const i3_rbk_buffer_i** buffers,
                                                 const uint32_t* offsets)
{
    assert(self != NULL);
    assert(buffers != NULL);

    assert(binding_count <= I3_VK_BIND_MAX_VERTEX_BUFFERS);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;
    i3_vk_cmd_bind_vertex_buffers_t* cmd = i3_vk_cmd_write_bind_vertex_buffers(&cmd_buffer->cmd_list);

    cmd->first_binding = first_binding;
    cmd->binding_count = binding_count;

    for (uint32_t i = 0; i < binding_count; i++)
    {
        // retain the buffer
        if (buffers[i] != NULL)
        {
            i3_vk_use_list_add(&cmd_buffer->use_list, buffers[i]);
            cmd->buffers[i] = ((i3_vk_buffer_o*)buffers[i]->self)->handle;
        }
        else
            cmd->buffers[i] = VK_NULL_HANDLE;

        if (offsets != NULL)
            cmd->offsets[i] = offsets[i];
        else
            cmd->offsets[i] = 0;
    }
}

// bind index buffer
static void i3_vk_cmd_buffer_bind_index_buffer(i3_rbk_cmd_buffer_o* self,
                                               i3_rbk_buffer_i* buffer,
                                               uint32_t offset,
                                               i3_rbk_index_type_t index_type)
{
    assert(self != NULL);
    assert(buffer != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // retain the buffer
    i3_vk_use_list_add(&cmd_buffer->use_list, buffer);

    i3_vk_cmd_bind_index_buffer_t* cmd = i3_vk_cmd_write_bind_index_buffer(&cmd_buffer->cmd_list);
    cmd->buffer = ((i3_vk_buffer_o*)buffer->self)->handle;
    cmd->offset = offset;
    cmd->index_type = i3_vk_convert_index_type(index_type);
}

// bind descriptor sets
static void i3_vk_cmd_buffer_bind_descriptor_sets(i3_rbk_cmd_buffer_o* self,
                                                  i3_rbk_pipeline_i* pipeline,
                                                  uint32_t first_set,
                                                  uint32_t descriptor_set_count,
                                                  const i3_rbk_descriptor_set_i** descriptor_sets)
{
    assert(self != NULL);
    assert(pipeline != NULL);
    assert(descriptor_sets != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // retain the pipeline
    i3_vk_use_list_add(&cmd_buffer->use_list, pipeline);

    i3_vk_pipeline_o* vk_pipeline = (i3_vk_pipeline_o*)pipeline->self;
    i3_vk_cmd_bind_descriptor_sets_t* cmd = i3_vk_cmd_write_bind_descriptor_sets(&cmd_buffer->cmd_list);
    cmd->bind_point = vk_pipeline->bind_point;
    cmd->layout = ((i3_vk_pipeline_layout_o*)vk_pipeline->layout->self)->handle;
    cmd->first_set = first_set;
    cmd->descriptor_set_count = descriptor_set_count;

    // copy & retain the descriptor sets
    for (uint32_t i = 0; i < descriptor_set_count; i++)
    {
        assert(descriptor_sets[i] != NULL);
        i3_vk_use_list_add(&cmd_buffer->use_list, descriptor_sets[i]);

        i3_vk_descriptor_set_o* vk_descriptor_set = (i3_vk_descriptor_set_o*)descriptor_sets[i]->self;
        cmd->descriptor_sets[i] = vk_descriptor_set->handle;

        // TODO: generate barriers
    }
}

// update descriptor sets
static void i3_vk_cmd_buffer_update_descriptor_sets(i3_rbk_cmd_buffer_o* self,
                                                    uint32_t write_count,
                                                    const i3_rbk_descriptor_set_write_t* writes)
{
    assert(self != NULL);
    assert(writes != NULL);
    assert(write_count > 0 && write_count <= I3_VK_MAX_DESCRIPTOR_SET_WRITES);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;
    i3_vk_cmd_update_descriptor_sets_t* cmd = i3_vk_cmd_write_update_descriptor_sets(&cmd_buffer->cmd_list);

    cmd->write_count = write_count;

    for (uint32_t i = 0; i < write_count; i++)
    {
        const i3_rbk_descriptor_set_write_t* write = &writes[i];
        VkWriteDescriptorSet* vk_write = &cmd->writes[i];
        *vk_write = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = NULL,
            .dstSet = ((i3_vk_descriptor_set_o*)write->descriptor_set->self)->handle,
            .dstBinding = write->binding,
            .dstArrayElement = write->array_element,
            .descriptorCount = 1,
            .descriptorType = i3_vk_convert_descriptor_type(write->descriptor_type),
        };

        i3_vk_use_list_add(&cmd_buffer->use_list, write->descriptor_set);

        // TODO: generate barriers
        // TODO: support count > 1
        // TODO: support dynamic offsets

        switch (write->descriptor_type)
        {
            case I3_RBK_DESCRIPTOR_TYPE_SAMPLER:
                i3_vk_use_list_add(&cmd_buffer->use_list, write->sampler);
                vk_write->pImageInfo = &cmd->image_infos[i];
                cmd->image_infos[i] = (VkDescriptorImageInfo){
                    .sampler = ((i3_vk_sampler_o*)write->sampler->self)->handle,
                };
                break;

            case I3_RBK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                i3_vk_use_list_add(&cmd_buffer->use_list, write->sampler);
                i3_vk_use_list_add(&cmd_buffer->use_list, write->image);
                vk_write->pImageInfo = &cmd->image_infos[i];
                cmd->image_infos[i] = (VkDescriptorImageInfo){
                    .sampler = ((i3_vk_sampler_o*)write->sampler->self)->handle,
                    .imageView = ((i3_vk_image_view_o*)write->image->self)->handle,
                    // TODO: .imageLayout = 0,
                };
                break;

            case I3_RBK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case I3_RBK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                i3_vk_use_list_add(&cmd_buffer->use_list, write->image);
                vk_write->pImageInfo = &cmd->image_infos[i];
                cmd->image_infos[i] = (VkDescriptorImageInfo){
                    .imageView = ((i3_vk_image_view_o*)write->image->self)->handle,
                    // TODO: .imageLayout = 0,
                };
                break;

            case I3_RBK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case I3_RBK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                i3_vk_use_list_add(&cmd_buffer->use_list, write->buffer);
                vk_write->pBufferInfo = &cmd->buffer_infos[i];
                cmd->buffer_infos[i] = (VkDescriptorBufferInfo){
                    .buffer = ((i3_vk_buffer_o*)write->buffer->self)->handle,
                    .range = ((i3_vk_buffer_o*)write->buffer->self)->desc.size,
                };
                break;
            default:
                assert(0);
        }
    }
}

// bind pipeline
static void i3_vk_cmd_buffer_bind_pipeline(i3_rbk_cmd_buffer_o* self, i3_rbk_pipeline_i* pipeline)
{
    assert(self != NULL);
    assert(pipeline != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // retain the pipeline
    i3_vk_use_list_add(&cmd_buffer->use_list, pipeline);

    i3_vk_cmd_bind_pipeline_t* cmd = i3_vk_cmd_write_bind_pipeline(&cmd_buffer->cmd_list);
    cmd->bind_point = ((i3_vk_pipeline_o*)pipeline->self)->bind_point;
    cmd->pipeline = ((i3_vk_pipeline_o*)pipeline->self)->handle;
}

// set viewports
static void i3_vk_cmd_buffer_set_viewports(i3_rbk_cmd_buffer_o* self,
                                           uint32_t first_viewport,
                                           uint32_t viewport_count,
                                           const i3_rbk_viewport_t* viewports)
{
    assert(self != NULL);
    assert(viewports != NULL);
    assert(viewport_count > 0 && viewport_count <= I3_VK_MAX_VIEWPORTS);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;
    i3_vk_cmd_set_viewports_t* cmd = i3_vk_cmd_write_set_viewports(&cmd_buffer->cmd_list);
    cmd->first_viewport = first_viewport;
    cmd->viewport_count = viewport_count;

    for (uint32_t i = 0; i < viewport_count; i++)
    {
        cmd->viewports[i].x = viewports[i].x;
        cmd->viewports[i].y = viewports[i].y;
        cmd->viewports[i].width = viewports[i].width;
        cmd->viewports[i].height = viewports[i].height;
        cmd->viewports[i].minDepth = viewports[i].min_depth;
        cmd->viewports[i].maxDepth = viewports[i].max_depth;
    }
}

// set scissors
static void i3_vk_cmd_buffer_set_scissors(i3_rbk_cmd_buffer_o* self,
                                          uint32_t first_scissor,
                                          uint32_t scissor_count,
                                          const i3_rbk_rect_t* scissors)
{
    assert(self != NULL);
    assert(scissors != NULL);
    assert(scissor_count > 0 && scissor_count <= I3_VK_MAX_SCISSORS);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;
    i3_vk_cmd_set_scissors_t* cmd = i3_vk_cmd_write_set_scissors(&cmd_buffer->cmd_list);
    cmd->first_scissor = first_scissor;
    cmd->scissor_count = scissor_count;

    for (uint32_t i = 0; i < scissor_count; i++)
    {
        cmd->scissors[i].offset.x = scissors[i].offset.x;
        cmd->scissors[i].offset.y = scissors[i].offset.y;
        cmd->scissors[i].extent.width = scissors[i].extent.width;
        cmd->scissors[i].extent.height = scissors[i].extent.height;
    }
}

// begin rendering
void i3_vk_cmd_buffer_begin_rendering(i3_rbk_cmd_buffer_o* self,
                                      i3_rbk_framebuffer_i* framebuffer,
                                      const i3_rbk_rect_t* render_area)
{
    assert(self != NULL);
    assert(framebuffer != NULL);
    assert(render_area != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // retain the framebuffer
    i3_vk_use_list_add(&cmd_buffer->use_list, framebuffer);

    // add barriers
    i3_vk_barrier_t* barriers = i3_vk_cmd_add_barriers(cmd_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    i3_vk_image_usage_t* barrier = i3_vk_add_image_barrier(barriers);
    i3_vk_framebuffer_o* fb = (i3_vk_framebuffer_o*)framebuffer->self;

    // color attachments
    for (uint32_t i = 0; i < fb->color_attachment_count; i++)
    {
        *barrier = (i3_vk_image_usage_t){
            .queue_family_index = VK_QUEUE_FAMILY_IGNORED,
            .access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .image_view = fb->color_attachments[i],
            .access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
    }

    // depth attachment
    if (fb->depth_attachment != NULL)
    {
        *barrier = (i3_vk_image_usage_t){
            .queue_family_index = VK_QUEUE_FAMILY_IGNORED,
            .access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .image_view = fb->depth_attachment,
            .access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
    }

    i3_vk_cmd_begin_rendering_t* cmd = i3_vk_cmd_write_begin_rendering(&cmd_buffer->cmd_list);
    cmd->framebuffer = ((i3_vk_framebuffer_o*)framebuffer->self)->handle;
    cmd->render_pass = ((i3_vk_framebuffer_o*)framebuffer->self)->render_pass;
    cmd->render_area.offset.x = render_area->offset.x;
    cmd->render_area.offset.y = render_area->offset.y;
    cmd->render_area.extent.width = render_area->extent.width;
    cmd->render_area.extent.height = render_area->extent.height;
}

// end rendering
static void i3_vk_cmd_buffer_end_rendering(i3_rbk_cmd_buffer_o* self)
{
    assert(self != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;
    i3_vk_cmd_end_rendering_t* cmd = i3_vk_cmd_write_end_rendering(&cmd_buffer->cmd_list);
}

// push constants
static void i3_vk_cmd_buffer_push_constants(i3_rbk_cmd_buffer_o* self,
                                            i3_rbk_pipeline_layout_i* layout,
                                            i3_rbk_shader_stage_flags_t stage_flags,
                                            uint32_t offset,
                                            uint32_t size,
                                            const void* data)
{
    assert(self != NULL);
    assert(layout != NULL);
    assert(data != NULL);
    assert(size > 0);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // retain the laout
    i3_vk_use_list_add(&cmd_buffer->use_list, layout);

    i3_vk_cmd_push_constants_t* cmd = i3_vk_cmd_write_push_constants(&cmd_buffer->cmd_list);
    cmd->layout = ((i3_vk_pipeline_layout_o*)layout->self)->handle;
    cmd->stage_flags = i3_vk_convert_shader_stage_flags(stage_flags);
    cmd->offset = offset;
    cmd->size = size;
    memcpy(cmd->data, data, size);
}

// draw
static void i3_vk_cmd_buffer_draw(i3_rbk_cmd_buffer_o* self,
                                  uint32_t vertex_count,
                                  uint32_t instance_count,
                                  uint32_t first_vertex,
                                  uint32_t first_instance)
{
    assert(self != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;
    i3_vk_cmd_draw_t* cmd = i3_vk_cmd_write_draw(&cmd_buffer->cmd_list);
    cmd->vertex_count = vertex_count;
    cmd->instance_count = instance_count;
    cmd->first_vertex = first_vertex;
    cmd->first_instance = first_instance;
}

// draw indexed
static void i3_vk_cmd_buffer_draw_indexed(i3_rbk_cmd_buffer_o* self,
                                          uint32_t index_count,
                                          uint32_t instance_count,
                                          uint32_t first_index,
                                          int32_t vertex_offset,
                                          uint32_t first_instance)
{
    assert(self != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;
    i3_vk_cmd_draw_indexed_t* cmd = i3_vk_cmd_write_draw_indexed(&cmd_buffer->cmd_list);
    cmd->index_count = index_count;
    cmd->instance_count = instance_count;
    cmd->first_index = first_index;
    cmd->vertex_offset = vertex_offset;
    cmd->first_instance = first_instance;
}

// draw indirect
static void i3_vk_cmd_buffer_draw_indirect(i3_rbk_cmd_buffer_o* self,
                                           i3_rbk_buffer_i* buffer,
                                           uint32_t offset,
                                           uint32_t draw_count,
                                           uint32_t stride)
{
    assert(self != NULL);
    assert(buffer != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // retain the buffer
    i3_vk_use_list_add(&cmd_buffer->use_list, buffer);

    i3_vk_cmd_draw_indirect_t* cmd = i3_vk_cmd_write_draw_indirect(&cmd_buffer->cmd_list);
    cmd->buffer = ((i3_vk_buffer_o*)buffer->self)->handle;
    cmd->offset = offset;
    cmd->draw_count = draw_count;
    cmd->stride = stride;
}

// draw indexed indirect
static void i3_vk_cmd_buffer_draw_indexed_indirect(i3_rbk_cmd_buffer_o* self,
                                                   i3_rbk_buffer_i* buffer,
                                                   uint32_t offset,
                                                   uint32_t draw_count,
                                                   uint32_t stride)
{
    assert(self != NULL);
    assert(buffer != NULL);

    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    // retain the buffer
    i3_vk_use_list_add(&cmd_buffer->use_list, buffer);

    i3_vk_cmd_draw_indexed_indirect_t* cmd = i3_vk_cmd_write_draw_indexed_indirect(&cmd_buffer->cmd_list);
    cmd->buffer = ((i3_vk_buffer_o*)buffer->self)->handle;
    cmd->offset = offset;
    cmd->draw_count = draw_count;
    cmd->stride = stride;
}

// destroy command buffer
static void i3_vk_cmd_buffer_destroy(i3_rbk_cmd_buffer_o* self)
{
    assert(self != NULL);
    i3_vk_cmd_buffer_o* cmd_buffer = (i3_vk_cmd_buffer_o*)self;

    cmd_buffer->base.release((i3_rbk_resource_o*)self);
}

static i3_vk_cmd_buffer_o i3_vk_cmd_buffer_iface_ =
{
    .base =
    {
        .add_ref = i3_vk_cmd_buffer_add_ref,
        .release = i3_vk_cmd_buffer_release,
        .get_use_count = i3_vk_cmd_buffer_get_use_count,
        .set_debug_name = i3_vk_cmd_buffer_set_debug_name,
    },
    .iface =
    {
        .get_resource = i3_vk_cmd_buffer_get_resource,
        .clear_image = i3_vk_cmd_buffer_clear_image,
        .write_buffer = i3_vk_cmd_buffer_write_buffer,
        .copy_buffer = i3_vk_cmd_buffer_copy_buffer,
        .bind_vertex_buffers = i3_vk_cmd_buffer_bind_vertex_buffers,
        .bind_index_buffer = i3_vk_cmd_buffer_bind_index_buffer,
        .bind_descriptor_sets = i3_vk_cmd_buffer_bind_descriptor_sets,
        .update_descriptor_sets = i3_vk_cmd_buffer_update_descriptor_sets,
        .bind_pipeline = i3_vk_cmd_buffer_bind_pipeline,
        .set_viewports = i3_vk_cmd_buffer_set_viewports,
        .set_scissors = i3_vk_cmd_buffer_set_scissors,
        .begin_rendering = i3_vk_cmd_buffer_begin_rendering,
        .end_rendering = i3_vk_cmd_buffer_end_rendering,
        .push_constants = i3_vk_cmd_buffer_push_constants,
        .draw = i3_vk_cmd_buffer_draw,
        .draw_indexed = i3_vk_cmd_buffer_draw_indexed,
        .draw_indirect = i3_vk_cmd_buffer_draw_indirect,
        .draw_indexed_indirect = i3_vk_cmd_buffer_draw_indexed_indirect,
        .destroy = i3_vk_cmd_buffer_destroy,
    },
};

// create cmd buffer

i3_rbk_cmd_buffer_i* i3_vk_device_create_cmd_buffer(i3_rbk_device_o* self)
{
    assert(self != NULL);

    i3_vk_device_o* device = (i3_vk_device_o*)self;
    i3_vk_cmd_buffer_o* cmd_buffer = i3_memory_pool_alloc(&device->cmd_buffer_pool);

    *cmd_buffer = i3_vk_cmd_buffer_iface_;
    cmd_buffer->base.self = (i3_rbk_resource_o*)cmd_buffer;
    cmd_buffer->iface.self = (i3_rbk_cmd_buffer_o*)cmd_buffer;
    cmd_buffer->device = device;
    cmd_buffer->use_count = 1;

    i3_vk_use_list_init(&cmd_buffer->use_list, device);
    i3_vk_cmd_list_init(&cmd_buffer->cmd_list, device);

    return &cmd_buffer->iface;
}