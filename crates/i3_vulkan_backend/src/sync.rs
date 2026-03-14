//! # Synchronization - Barrier Management
//!
//! This module handles GPU synchronization via Vulkan memory barriers (VK_KHR_synchronization2).
//! This is a critical point for rendering correctness and performance.
//!
//! ## Problem
//!
//! Vulkan requires explicit synchronization between GPU operations. Without barriers:
//! - **Write-After-Write (WAW) hazard**: two passes write to the same resource
//! - **Read-After-Write (RAW) hazard**: a pass reads before the write is complete
//! - **Write-After-Read (WAR) hazard**: a pass writes while another is reading
//!
//! ## Optimization Strategy
//!
//! The backend uses **state tracking** to avoid unnecessary barriers:
//! - Each resource tracks its last layout, access flags, and pipeline stage
//! - Barriers are only emitted if the state actually changes
//! - Read-After-Read (RAR) is skipped if the stage is compatible
//!
//! ## Barrier Types
//!
//! - **Image Memory Barrier**: layout transitions (e.g., UNDEFINED → COLOR_ATTACHMENT_OPTIMAL)
//! - **Buffer Memory Barrier**: buffer access synchronization
//!
//! ## Flow Example
//!
//! ```text
//! Pass A (writes image) → [Image Barrier] → Pass B (reads image)
//!                        old_layout: COLOR_ATTACHMENT_OPTIMAL
//!                        new_layout: SHADER_READ_ONLY_OPTIMAL
//! ```

use ash::vk;
use tracing::debug;

use i3_gfx::graph::types::ResourceUsage;

use crate::backend::VulkanBackend;

/// Generate an image memory barrier if needed.
///
/// This function is the core of image synchronization. It compares the current state
/// of the resource with the requested state and generates a barrier only if necessary.
///
/// # Optimizations
///
/// - **Skip RAR**: If both accesses are reads and the stage is compatible, no barrier
/// - **Skip identical layout**: If the layout doesn't change and there's no write, no barrier
///
/// # Arguments
///
/// * `backend` - Mutable reference to the backend (to update state)
/// * `physical_id` - Physical ID of the image in the ResourceArena
/// * `new_layout` - Target Vulkan layout
/// * `dst_access` - Destination access flags
/// * `dst_stage` - Destination pipeline stage
///
/// # Returns
///
/// `Some(barrier)` if a barrier is needed, `None` otherwise
pub fn get_image_barrier(
    backend: &mut VulkanBackend,
    physical_id: u64,
    new_layout: vk::ImageLayout,
    dst_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
) -> Option<vk::ImageMemoryBarrier2<'static>> {
    if let Some(img) = backend.images.get_mut(physical_id) {
        // Optimization: Skip only for Read-After-Read (RAR) where layout and stage already match
        let is_write = |access: vk::AccessFlags2| {
            access.intersects(
                vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE
                    | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags2::TRANSFER_WRITE
                    | vk::AccessFlags2::MEMORY_WRITE,
            )
        };

        let needs_barrier =
            img.last_layout != new_layout || is_write(img.last_access) || is_write(dst_access);

        if !needs_barrier && (img.last_stage & dst_stage) == dst_stage {
            return None;
        }

        debug!(
            "Transition Image {:?}: {:?} -> {:?}",
            physical_id, img.last_layout, new_layout
        );

        let aspect_mask = if img.format == vk::Format::D32_SFLOAT {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(img.last_stage)
            .src_access_mask(img.last_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(img.last_layout)
            .new_layout(new_layout)
            .image(img.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        img.last_layout = new_layout;
        img.last_access = dst_access;
        img.last_stage = dst_stage;

        Some(barrier)
    } else {
        None
    }
}

/// Generate a buffer memory barrier if needed.
///
/// Similar to [`get_image_barrier`] but for buffers. Buffers don't have layouts,
/// so only access flags and pipeline stages are compared.
///
/// # Arguments
///
/// * `backend` - Mutable reference to the backend
/// * `physical_id` - Physical ID of the buffer in the ResourceArena
/// * `dst_access` - Destination access flags
/// * `dst_stage` - Destination pipeline stage
///
/// # Returns
///
/// `Some(barrier)` if a barrier is needed, `None` otherwise
pub fn get_buffer_barrier(
    backend: &mut VulkanBackend,
    physical_id: u64,
    dst_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
) -> Option<vk::BufferMemoryBarrier2<'static>> {
    if let Some(buf) = backend.buffers.get_mut(physical_id) {
        // Optimization: Skip only for Read-After-Read (RAR) where state already matches
        let is_write = |access: vk::AccessFlags2| {
            access.intersects(
                vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE
                    | vk::AccessFlags2::TRANSFER_WRITE
                    | vk::AccessFlags2::MEMORY_WRITE,
            )
        };

        let needs_barrier = is_write(buf.last_access) || is_write(dst_access);

        if !needs_barrier && (buf.last_stage & dst_stage) == dst_stage {
            return None;
        }

        debug!(
            "Transition Buffer {:?}: {:?} -> {:?} / {:?} -> {:?}",
            physical_id, buf.last_stage, dst_stage, buf.last_access, dst_access
        );

        let barrier = vk::BufferMemoryBarrier2::default()
            .src_stage_mask(buf.last_stage)
            .src_access_mask(buf.last_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .buffer(buf.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        buf.last_access = dst_access;
        buf.last_stage = dst_stage;

        Some(barrier)
    } else {
        None
    }
}

/// Determine the optimal image layout, access flags, and pipeline stage for a given usage.
///
/// This function translates abstract [`ResourceUsage`] flags into concrete Vulkan state.
/// It's used by the barrier generation system to determine target states.
///
/// # Arguments
///
/// * `usage` - Abstract resource usage flags from the render graph
/// * `is_write` - Whether this is a write operation
/// * `bind_point` - Pipeline bind point (graphics or compute)
///
/// # Returns
///
/// A tuple of (layout, access_flags, pipeline_stage)
pub fn get_image_state(
    usage: ResourceUsage,
    is_write: bool,
    bind_point: vk::PipelineBindPoint,
) -> (vk::ImageLayout, vk::AccessFlags2, vk::PipelineStageFlags2) {
    let mut layout = vk::ImageLayout::GENERAL;
    let mut access = vk::AccessFlags2::empty();
    let mut stage = vk::PipelineStageFlags2::NONE;

    if usage.intersects(ResourceUsage::COLOR_ATTACHMENT) {
        layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        access = vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
        if !is_write {
            access |= vk::AccessFlags2::COLOR_ATTACHMENT_READ;
        }
        stage = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
    } else if usage.intersects(ResourceUsage::DEPTH_STENCIL) {
        if is_write || usage.intersects(ResourceUsage::WRITE) {
            layout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            access = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
        } else {
            layout = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            access = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
        }
        stage = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
    } else if usage.intersects(ResourceUsage::SHADER_WRITE) {
        layout = vk::ImageLayout::GENERAL;
        access = vk::AccessFlags2::SHADER_STORAGE_WRITE
            | vk::AccessFlags2::SHADER_STORAGE_READ
            | vk::AccessFlags2::SHADER_WRITE;
        stage = if bind_point == vk::PipelineBindPoint::COMPUTE {
            vk::PipelineStageFlags2::COMPUTE_SHADER
        } else {
            vk::PipelineStageFlags2::FRAGMENT_SHADER
        };
    } else if usage.intersects(ResourceUsage::SHADER_READ) {
        layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        access = vk::AccessFlags2::SHADER_READ;
        stage = if bind_point == vk::PipelineBindPoint::COMPUTE {
            vk::PipelineStageFlags2::COMPUTE_SHADER
        } else {
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::VERTEX_SHADER
        };
    } else if usage.intersects(ResourceUsage::TRANSFER_WRITE) {
        layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        access = vk::AccessFlags2::TRANSFER_WRITE;
        stage = vk::PipelineStageFlags2::TRANSFER;
    } else if usage.intersects(ResourceUsage::TRANSFER_READ) {
        layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        access = vk::AccessFlags2::TRANSFER_READ;
        stage = vk::PipelineStageFlags2::TRANSFER;
    }

    (layout, access, stage)
}

/// Determine the optimal access flags and pipeline stage for a buffer.
///
/// Similar to [`get_image_state`] but for buffers (no layout transitions).
///
/// # Arguments
///
/// * `usage` - Abstract resource usage flags from the render graph
/// * `bind_point` - Pipeline bind point (graphics or compute)
///
/// # Returns
///
/// A tuple of (access_flags, pipeline_stage)
pub fn get_buffer_state(
    usage: ResourceUsage,
    bind_point: vk::PipelineBindPoint,
) -> (vk::AccessFlags2, vk::PipelineStageFlags2) {
    let mut access = vk::AccessFlags2::empty();
    let mut stage = vk::PipelineStageFlags2::NONE;

    if usage.intersects(ResourceUsage::SHADER_WRITE) {
        access = vk::AccessFlags2::SHADER_STORAGE_WRITE
            | vk::AccessFlags2::SHADER_STORAGE_READ
            | vk::AccessFlags2::SHADER_WRITE;
        stage = if bind_point == vk::PipelineBindPoint::COMPUTE {
            vk::PipelineStageFlags2::COMPUTE_SHADER
        } else {
            vk::PipelineStageFlags2::FRAGMENT_SHADER
        };
    } else if usage.intersects(ResourceUsage::SHADER_READ) {
        access = vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::UNIFORM_READ;
        stage = if bind_point == vk::PipelineBindPoint::COMPUTE {
            vk::PipelineStageFlags2::COMPUTE_SHADER
        } else {
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::VERTEX_SHADER
        };
    } else if usage.intersects(ResourceUsage::TRANSFER_WRITE) {
        access = vk::AccessFlags2::TRANSFER_WRITE;
        stage = vk::PipelineStageFlags2::TRANSFER;
    } else if usage.intersects(ResourceUsage::TRANSFER_READ) {
        access = vk::AccessFlags2::TRANSFER_READ;
        stage = vk::PipelineStageFlags2::TRANSFER;
    }

    (access, stage)
}

/// Create a new semaphore and return its handle.
///
/// Semaphores are used for GPU-GPU synchronization between queue submissions.
/// This function recycles semaphores from a pool to avoid frequent Vulkan allocations.
///
/// # Arguments
///
/// * `backend` - Mutable reference to the backend
///
/// # Returns
///
/// Handle to the semaphore in the ResourceArena
pub fn create_semaphore(backend: &mut VulkanBackend) -> u64 {
    let sem = create_semaphore_raw(backend);
    backend.semaphores.insert(sem)
}

/// Create a raw Vulkan semaphore, recycling from the pool if available.
///
/// This is the low-level semaphore creation function. It first checks the
/// recycled semaphore pool before creating a new Vulkan semaphore.
///
/// # Arguments
///
/// * `backend` - Mutable reference to the backend
///
/// # Returns
///
/// Raw Vulkan semaphore handle
pub fn create_semaphore_raw(backend: &mut VulkanBackend) -> vk::Semaphore {
    if let Some(recycled) = backend.recycled_semaphores.pop() {
        recycled
    } else {
        let device = backend.get_device();
        let create_info = vk::SemaphoreCreateInfo::default();
        unsafe { device.handle.create_semaphore(&create_info, None) }.unwrap()
    }
}
