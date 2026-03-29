use ash::vk;
use crate::backend::VulkanBackend;
use i3_gfx::graph::types::{QueueType, ResourceUsage};

/// Get the queue type for a given family index.
pub fn get_queue_type_from_family(backend: &VulkanBackend, family: u32) -> QueueType {
    get_queue_type_from_family_info(
        family,
        backend.graphics_family,
        backend.compute_family,
        backend.transfer_family,
    )
}

/// Get the queue type for a given family index using raw info.
pub fn get_queue_type_from_family_info(
    family: u32,
    graphics_family: u32,
    compute_family: u32,
    transfer_family: u32,
) -> QueueType {
    if family == graphics_family {
        return QueueType::Graphics;
    }
    if family == compute_family {
        return QueueType::AsyncCompute;
    }
    if family == transfer_family {
        return QueueType::Transfer;
    }
    QueueType::Graphics // Default
}

pub fn get_queue_family(backend: &VulkanBackend, queue_type: QueueType) -> u32 {
    match queue_type {
        QueueType::Graphics => backend.graphics.as_ref().unwrap().family,
        QueueType::AsyncCompute => backend.compute.as_ref()
            .map(|c| c.family)
            .unwrap_or(backend.graphics_family),
        QueueType::Transfer => backend.transfer.as_ref()
            .map(|t| t.family)
            .unwrap_or(backend.graphics_family),
    }
}

/// Generate an image memory barrier if needed.
pub fn get_image_barrier(
    backend: &mut VulkanBackend,
    physical_id: u64,
    new_layout: vk::ImageLayout,
    dst_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    current_queue_family: u32,
) -> Option<vk::ImageMemoryBarrier2<'static>> {
    let (last_layout, last_access, last_stage, last_queue_family, format, image, concurrent) = {
        let img = backend.images.get(physical_id).unwrap();
        (img.last_layout, img.last_access, img.last_stage, img.last_queue_family, img.format, img.image, img.concurrent)
    };

    let needs_barrier = last_layout != new_layout
        || (last_access & dst_access) != dst_access
        || (last_stage & dst_stage) != dst_stage
        || (!concurrent && last_queue_family != current_queue_family);

    if !needs_barrier {
        // Still update tracking even if no barrier needed (queue family changed on concurrent resource)
        if last_queue_family != current_queue_family {
            let img = backend.images.get_mut(physical_id).unwrap();
            img.last_queue_family = current_queue_family;
        }
        return None;
    }

    let aspect_mask = if format == vk::Format::D32_SFLOAT || format == vk::Format::D24_UNORM_S8_UINT || format == vk::Format::D32_SFLOAT_S8_UINT {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let current_queue_type = get_queue_type_from_family_info(current_queue_family, backend.graphics_family, backend.compute_family, backend.transfer_family);

    // ONLY handle same-family transitions here. Ownership transfers (EXCLUSIVE)
    // are now explicitly managed by the graph compiler via get_image_ownership_barrier.
    if !concurrent && last_queue_family != current_queue_family {
        // We expect an explicit Acquire barrier to have been recorded already.
        // Update tracking to reflect the new owner and state.
        let img = backend.images.get_mut(physical_id).unwrap();
        img.last_layout = new_layout;
        img.last_access = dst_access;
        img.last_stage = dst_stage;
        img.last_queue_family = current_queue_family;
        return None;
    }

    // For concurrent resources that crossed queue families, last_stage may contain
    // stages from a different queue type (e.g. FRAGMENT_SHADER from graphics).
    // The timeline semaphore already provides cross-queue ordering, so TOP_OF_PIPE
    // is the correct src_stage on the new queue — no need to sanitize with a warning.
    let effective_last_stage = if concurrent && last_queue_family != current_queue_family {
        vk::PipelineStageFlags2::TOP_OF_PIPE
    } else {
        last_stage
    };

    let sanitized_src_stage = sanitize_stages(effective_last_stage, current_queue_type);
    let sanitized_dst_stage = sanitize_stages(dst_stage, current_queue_type);
    let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(sanitized_src_stage)
        .src_access_mask(sanitize_access(last_access, sanitized_src_stage, current_queue_type))
        .dst_stage_mask(sanitized_dst_stage)
        .dst_access_mask(sanitize_access(dst_access, sanitized_dst_stage, current_queue_type))
        .old_layout(last_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    let img = backend.images.get_mut(physical_id).unwrap();
    img.last_layout = new_layout;
    img.last_access = dst_access;
    img.last_stage = dst_stage;
    img.last_queue_family = current_queue_family;

    Some(barrier)
}

/// Generate an ownership transfer barrier for an image (Release or Acquire).
pub fn get_image_ownership_barrier(
    backend: &mut VulkanBackend,
    physical_id: u64,
    src_queue_family: u32,
    dst_queue_family: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_usage: ResourceUsage,
    dst_usage: ResourceUsage,
    is_release: bool,
) -> vk::ImageMemoryBarrier2<'static> {
    let (image, format) = {
        let img = backend.images.get(physical_id).unwrap();
        (img.image, img.format)
    };

    let aspect_mask = if format == vk::Format::D32_SFLOAT 
        || format == vk::Format::D24_UNORM_S8_UINT 
        || format == vk::Format::D32_SFLOAT_S8_UINT {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let src_queue_type = get_queue_type_from_family(backend, src_queue_family);
    let dst_queue_type = get_queue_type_from_family(backend, dst_queue_family);

    let src_bind_point = match src_queue_type {
        QueueType::Graphics => vk::PipelineBindPoint::GRAPHICS,
        _ => vk::PipelineBindPoint::COMPUTE,
    };
    let dst_bind_point = match dst_queue_type {
        QueueType::Graphics => vk::PipelineBindPoint::GRAPHICS,
        _ => vk::PipelineBindPoint::COMPUTE,
    };

    let (_, src_access, src_stage) = get_image_state(src_usage, true, src_bind_point);
    let (_, dst_access, dst_stage) = get_image_state(dst_usage, false, dst_bind_point);

    let mut barrier = vk::ImageMemoryBarrier2::default()
        .image(image)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(src_queue_family)
        .dst_queue_family_index(dst_queue_family)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });

    if is_release {
        let sanitized_src = sanitize_stages(src_stage, src_queue_type);
        barrier = barrier.src_stage_mask(sanitized_src)
            .src_access_mask(sanitize_access(src_access, sanitized_src, src_queue_type))
            .dst_stage_mask(vk::PipelineStageFlags2::empty())
            .dst_access_mask(vk::AccessFlags2::empty());
    } else {
        let sanitized_dst = sanitize_stages(dst_stage, dst_queue_type);
        barrier = barrier.src_stage_mask(vk::PipelineStageFlags2::empty())
            .src_access_mask(vk::AccessFlags2::empty())
            .dst_stage_mask(sanitized_dst)
            .dst_access_mask(sanitize_access(dst_access, sanitized_dst, dst_queue_type));
            
        // Update tracking during Acquire
        let img = backend.images.get_mut(physical_id).unwrap();
        img.last_layout = new_layout;
        img.last_access = dst_access;
        img.last_stage = dst_stage;
        img.last_queue_family = dst_queue_family;
    }

    barrier
}

/// Generate an ownership transfer barrier for a buffer (Release or Acquire).
pub fn get_buffer_ownership_barrier(
    backend: &mut VulkanBackend,
    physical_id: u64,
    src_queue_family: u32,
    dst_queue_family: u32,
    src_usage: ResourceUsage,
    dst_usage: ResourceUsage,
    is_release: bool,
) -> vk::BufferMemoryBarrier2<'static> {
    let buffer = {
        let buf = backend.buffers.get(physical_id).unwrap();
        buf.buffer
    };
    
    let src_queue_type = get_queue_type_from_family(backend, src_queue_family);
    let dst_queue_type = get_queue_type_from_family(backend, dst_queue_family);

    let src_bind_point = match src_queue_type {
        QueueType::Graphics => vk::PipelineBindPoint::GRAPHICS,
        _ => vk::PipelineBindPoint::COMPUTE,
    };
    let dst_bind_point = match dst_queue_type {
        QueueType::Graphics => vk::PipelineBindPoint::GRAPHICS,
        _ => vk::PipelineBindPoint::COMPUTE,
    };

    let (src_access, src_stage) = get_buffer_state(src_usage, src_bind_point);
    let (dst_access, dst_stage) = get_buffer_state(dst_usage, dst_bind_point);

    let mut barrier = vk::BufferMemoryBarrier2::default()
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE)
        .src_queue_family_index(src_queue_family)
        .dst_queue_family_index(dst_queue_family);

    if is_release {
        let sanitized_src = sanitize_stages(src_stage, src_queue_type);
        barrier = barrier.src_stage_mask(sanitized_src)
            .src_access_mask(sanitize_access(src_access, sanitized_src, src_queue_type))
            .dst_stage_mask(vk::PipelineStageFlags2::empty())
            .dst_access_mask(vk::AccessFlags2::empty());
    } else {
        let sanitized_dst = sanitize_stages(dst_stage, dst_queue_type);
        barrier = barrier.src_stage_mask(vk::PipelineStageFlags2::empty())
            .src_access_mask(vk::AccessFlags2::empty())
            .dst_stage_mask(sanitized_dst)
            .dst_access_mask(sanitize_access(dst_access, sanitized_dst, dst_queue_type));
            
        // Update tracking during Acquire
        let buf = backend.buffers.get_mut(physical_id).unwrap();
        buf.last_access = dst_access;
        buf.last_stage = dst_stage;
        buf.last_queue_family = dst_queue_family;
    }

    barrier
}

/// Generate a buffer memory barrier if needed.
pub fn get_buffer_barrier(
    backend: &mut VulkanBackend,
    physical_id: u64,
    dst_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    current_queue_family: u32,
) -> Option<vk::BufferMemoryBarrier2<'static>> {
    let (last_access, last_stage, last_queue_family, buffer, concurrent) = {
        let buf = backend.buffers.get(physical_id).unwrap();
        (buf.last_access, buf.last_stage, buf.last_queue_family, buf.buffer, buf.concurrent)
    };

    let needs_barrier = (last_access & dst_access) != dst_access
        || (last_stage & dst_stage) != dst_stage
        || (!concurrent && last_queue_family != current_queue_family);

    if !needs_barrier {
        if last_queue_family != current_queue_family {
            let buf = backend.buffers.get_mut(physical_id).unwrap();
            buf.last_queue_family = current_queue_family;
        }
        return None;
    }

    let current_queue_type = get_queue_type_from_family_info(current_queue_family, backend.graphics_family, backend.compute_family, backend.transfer_family);

    // ONLY handle same-family transitions here. Ownership transfers (EXCLUSIVE)
    // are now explicitly managed by the graph compiler via get_buffer_ownership_barrier.
    if !concurrent && last_queue_family != current_queue_family {
        let buf = backend.buffers.get_mut(physical_id).unwrap();
        buf.last_access = dst_access;
        buf.last_stage = dst_stage;
        buf.last_queue_family = current_queue_family;
        return None;
    }

    // For concurrent resources that crossed queue families, last_stage may contain
    // stages from a different queue type. The timeline semaphore already provides
    // cross-queue ordering, so TOP_OF_PIPE is the correct src_stage on the new queue.
    let effective_last_stage = if concurrent && last_queue_family != current_queue_family {
        vk::PipelineStageFlags2::TOP_OF_PIPE
    } else {
        last_stage
    };

    let sanitized_src_stage = sanitize_stages(effective_last_stage, current_queue_type);
    let sanitized_dst_stage = sanitize_stages(dst_stage, current_queue_type);
    let barrier = vk::BufferMemoryBarrier2::default()
        .src_stage_mask(sanitized_src_stage)
        .src_access_mask(sanitize_access(last_access, sanitized_src_stage, current_queue_type))
        .dst_stage_mask(sanitized_dst_stage)
        .dst_access_mask(sanitize_access(dst_access, sanitized_dst_stage, current_queue_type))
        .buffer(buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE);

    let buf = backend.buffers.get_mut(physical_id).unwrap();
    buf.last_access = dst_access;
    buf.last_stage = dst_stage;
    buf.last_queue_family = current_queue_family;

    Some(barrier)
}

/// Mask pipeline stages based on what's supported by the queue family.
pub fn sanitize_stages(
    stages: vk::PipelineStageFlags2,
    queue_type: QueueType,
) -> vk::PipelineStageFlags2 {
    match queue_type {
        QueueType::Graphics => stages,
        QueueType::AsyncCompute => {
            let supported = vk::PipelineStageFlags2::TOP_OF_PIPE
                | vk::PipelineStageFlags2::BOTTOM_OF_PIPE
                | vk::PipelineStageFlags2::COMPUTE_SHADER
                | vk::PipelineStageFlags2::TRANSFER
                | vk::PipelineStageFlags2::DRAW_INDIRECT;

            let result = stages & supported;
            if result.is_empty() && !stages.is_empty() {
                tracing::warn!(
                    unsupported = ?stages,
                    "sanitize_stages: graphics stage on AsyncCompute queue, falling back to TOP_OF_PIPE"
                );
                vk::PipelineStageFlags2::TOP_OF_PIPE
            } else {
                result
            }
        }
        QueueType::Transfer => {
            let supported = vk::PipelineStageFlags2::TOP_OF_PIPE
                | vk::PipelineStageFlags2::BOTTOM_OF_PIPE
                | vk::PipelineStageFlags2::TRANSFER;

            let result = stages & supported;
            if result.is_empty() && !stages.is_empty() {
                tracing::warn!(
                    unsupported = ?stages,
                    "sanitize_stages: unsupported stage on Transfer queue, falling back to TOP_OF_PIPE"
                );
                vk::PipelineStageFlags2::TOP_OF_PIPE
            } else {
                result
            }
        }
    }
}

pub fn sanitize_access(
    access: vk::AccessFlags2,
    stages: vk::PipelineStageFlags2,
    queue_type: QueueType,
) -> vk::AccessFlags2 {
    // VUID-VkImageMemoryBarrier2-srcAccessMask-03909/07454: 
    // If stages contains ONLY TOP_OF_PIPE or BOTTOM_OF_PIPE, access must be EMPTY
    let effective_stages = stages & !(vk::PipelineStageFlags2::TOP_OF_PIPE | vk::PipelineStageFlags2::BOTTOM_OF_PIPE);
    if effective_stages.is_empty() {
        return vk::AccessFlags2::empty();
    }

    let mask = match queue_type {
        QueueType::Graphics => access,
        QueueType::AsyncCompute => {
            let supported = vk::AccessFlags2::INDIRECT_COMMAND_READ
                | vk::AccessFlags2::INDEX_READ
                | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
                | vk::AccessFlags2::UNIFORM_READ
                | vk::AccessFlags2::SHADER_SAMPLED_READ
                | vk::AccessFlags2::SHADER_STORAGE_READ
                | vk::AccessFlags2::SHADER_STORAGE_WRITE
                | vk::AccessFlags2::SHADER_READ
                | vk::AccessFlags2::SHADER_WRITE
                | vk::AccessFlags2::TRANSFER_READ
                | vk::AccessFlags2::TRANSFER_WRITE
                | vk::AccessFlags2::MEMORY_READ
                | vk::AccessFlags2::MEMORY_WRITE;
            access & supported
        }
        QueueType::Transfer => {
            let supported = vk::AccessFlags2::TRANSFER_READ
                | vk::AccessFlags2::TRANSFER_WRITE
                | vk::AccessFlags2::MEMORY_READ
                | vk::AccessFlags2::MEMORY_WRITE;
            access & supported
        }
    };
    mask
}

pub fn sanitize_image_barrier(
    mut barrier: vk::ImageMemoryBarrier2<'static>,
    graphics_family: u32,
    compute_family: u32,
    transfer_family: u32,
    current_queue_type: QueueType,
) -> vk::ImageMemoryBarrier2<'static> {
    let src_family = barrier.src_queue_family_index;
    let dst_family = barrier.dst_queue_family_index;

    // If it's an ownership transfer, we MUST sanitize against the respective families
    if src_family != dst_family && src_family != vk::QUEUE_FAMILY_IGNORED && dst_family != vk::QUEUE_FAMILY_IGNORED {
        let src_type = get_queue_type_from_family_info(src_family, graphics_family, compute_family, transfer_family);
        let dst_type = get_queue_type_from_family_info(dst_family, graphics_family, compute_family, transfer_family);
        
        barrier.src_stage_mask = sanitize_stages(barrier.src_stage_mask, src_type);
        barrier.dst_stage_mask = sanitize_stages(barrier.dst_stage_mask, dst_type);
        // Important: Sanitize access masks against THEIR stages AFTER stage sanitization
        barrier.src_access_mask = sanitize_access(barrier.src_access_mask, barrier.src_stage_mask, src_type);
        barrier.dst_access_mask = sanitize_access(barrier.dst_access_mask, barrier.dst_stage_mask, dst_type);
    } else {
        // Regular barrier on a single queue
        barrier.src_stage_mask = sanitize_stages(barrier.src_stage_mask, current_queue_type);
        barrier.dst_stage_mask = sanitize_stages(barrier.dst_stage_mask, current_queue_type);
        barrier.src_access_mask = sanitize_access(barrier.src_access_mask, barrier.src_stage_mask, current_queue_type);
        barrier.dst_access_mask = sanitize_access(barrier.dst_access_mask, barrier.dst_stage_mask, current_queue_type);
    }

    barrier
}

pub fn sanitize_buffer_barrier(
    mut barrier: vk::BufferMemoryBarrier2<'static>,
    graphics_family: u32,
    compute_family: u32,
    transfer_family: u32,
    current_queue_type: QueueType,
) -> vk::BufferMemoryBarrier2<'static> {
    let src_family = barrier.src_queue_family_index;
    let dst_family = barrier.dst_queue_family_index;

    if src_family != dst_family && src_family != vk::QUEUE_FAMILY_IGNORED && dst_family != vk::QUEUE_FAMILY_IGNORED {
        let src_type = get_queue_type_from_family_info(src_family, graphics_family, compute_family, transfer_family);
        let dst_type = get_queue_type_from_family_info(dst_family, graphics_family, compute_family, transfer_family);
        
        barrier.src_stage_mask = sanitize_stages(barrier.src_stage_mask, src_type);
        barrier.dst_stage_mask = sanitize_stages(barrier.dst_stage_mask, dst_type);
        barrier.src_access_mask = sanitize_access(barrier.src_access_mask, barrier.src_stage_mask, src_type);
        barrier.dst_access_mask = sanitize_access(barrier.dst_access_mask, barrier.dst_stage_mask, dst_type);
    } else {
        barrier.src_stage_mask = sanitize_stages(barrier.src_stage_mask, current_queue_type);
        barrier.dst_stage_mask = sanitize_stages(barrier.dst_stage_mask, current_queue_type);
        barrier.src_access_mask = sanitize_access(barrier.src_access_mask, barrier.src_stage_mask, current_queue_type);
        barrier.dst_access_mask = sanitize_access(barrier.dst_access_mask, barrier.dst_stage_mask, current_queue_type);
    }

    barrier
}

pub fn get_image_state(
    usage: ResourceUsage,
    _is_write: bool,
    bind_point: vk::PipelineBindPoint,
) -> (vk::ImageLayout, vk::AccessFlags2, vk::PipelineStageFlags2) {
    if usage.contains(ResourceUsage::COLOR_ATTACHMENT) {
        return (
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );
    }
    if usage.contains(ResourceUsage::DEPTH_STENCIL) {
        return (
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        );
    }
    if usage.contains(ResourceUsage::SHADER_READ) {
        return (
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags2::SHADER_READ,
            match bind_point {
                vk::PipelineBindPoint::COMPUTE => vk::PipelineStageFlags2::COMPUTE_SHADER,
                _ => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            },
        );
    }
    if usage.contains(ResourceUsage::SHADER_WRITE) {
        return (
            vk::ImageLayout::GENERAL,
            vk::AccessFlags2::SHADER_WRITE,
            match bind_point {
                vk::PipelineBindPoint::COMPUTE => vk::PipelineStageFlags2::COMPUTE_SHADER,
                _ => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            },
        );
    }
    if usage.contains(ResourceUsage::TRANSFER_READ) {
        return (
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags2::TRANSFER_READ,
            vk::PipelineStageFlags2::TRANSFER,
        );
    }
    if usage.contains(ResourceUsage::TRANSFER_WRITE) {
        return (
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
        );
    }
    if usage.contains(ResourceUsage::PRESENT) {
        return (
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        );
    }

    (
        vk::ImageLayout::UNDEFINED,
        vk::AccessFlags2::empty(),
        vk::PipelineStageFlags2::TOP_OF_PIPE,
    )
}

pub fn get_buffer_state(
    usage: ResourceUsage,
    bind_point: vk::PipelineBindPoint,
) -> (vk::AccessFlags2, vk::PipelineStageFlags2) {
    if usage.contains(ResourceUsage::SHADER_READ) || usage.contains(ResourceUsage::SHADER_WRITE) {
        return (
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            match bind_point {
                vk::PipelineBindPoint::COMPUTE => vk::PipelineStageFlags2::COMPUTE_SHADER,
                _ => vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            },
        );
    }
    if usage.contains(ResourceUsage::INDIRECT_READ) {
        return (
            vk::AccessFlags2::INDIRECT_COMMAND_READ,
            vk::PipelineStageFlags2::DRAW_INDIRECT,
        );
    }
    if usage.contains(ResourceUsage::TRANSFER_READ) {
        return (
            vk::AccessFlags2::TRANSFER_READ,
            vk::PipelineStageFlags2::TRANSFER,
        );
    }
    if usage.contains(ResourceUsage::TRANSFER_WRITE) {
        return (
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
        );
    }

    (
        vk::AccessFlags2::empty(),
        vk::PipelineStageFlags2::TOP_OF_PIPE,
    )
}

pub fn create_semaphore(backend: &mut VulkanBackend, is_timeline: bool) -> u64 {
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(if is_timeline {
            vk::SemaphoreType::TIMELINE
        } else {
            vk::SemaphoreType::BINARY
        })
        .initial_value(0);

    let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

    let handle = unsafe {
        backend
            .get_device()
            .handle
            .create_semaphore(&create_info, None)
            .unwrap()
    };

    backend.semaphores.insert(handle)
}
