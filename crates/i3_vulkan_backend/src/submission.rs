//! # Queue Submission - Timeline Semaphores
//!
//! This module handles GPU queue submission and frame lifecycle management.
//! It uses **timeline semaphores** (VK_KHR_timeline_semaphore) for efficient
//! CPU-GPU synchronization without busy-waiting.
//!
//! ## Timeline Semaphores
//!
//! Timeline semaphores are monotonically increasing counters that can be waited on
//! by both CPU and GPU. This is more efficient than binary semaphores because:
//! - No need to recreate semaphores each frame
//! - Can wait for specific values (not just signaled/unsignaled)
//! - Enables precise frame-in-flight management
//!
//! ## Frame Lifecycle
//!
//! ```text
//! begin_frame() → acquire_swapchain_image() → record_pass() → submit() → end_frame()
//!       ↓                                                                    ↓
//!   Wait for previous frame                                          Garbage collection
//!   Reset command pools
//! ```
//!
//! ## Frame-in-Flight
//!
//! The backend uses multiple frame contexts (typically 2-3) to allow the CPU
//! to record commands while the GPU is still processing previous frames.
//! Each frame context has its own command pool and descriptor pool.

use ash::vk;
use ash::vk::Handle;
use i3_gfx::graph::backend::*;
use i3_gfx::graph::types::*;
use tracing::debug;

use std::collections::HashMap;

use crate::backend::VulkanBackend;

/// Wait for the timeline semaphore to reach a specific value on the host (CPU).
///
/// This is a blocking wait that stalls the CPU until the GPU has completed
/// all work up to the specified timeline value.
///
/// # Arguments
///
/// * `backend` - Reference to the backend
/// * `value` - Timeline value to wait for
/// * `timeout_ns` - Timeout in nanoseconds (u64::MAX for infinite)
///
/// # Returns
///
/// `Ok(())` if the timeline was reached, `Err` on timeout or error
pub fn wait_for_timeline(
    backend: &VulkanBackend,
    value: u64,
    timeout_ns: u64,
) -> Result<(), String> {
    let graphics = backend.graphics.as_ref().unwrap();
    let semaphores = [graphics.timeline_sem];
    let values = [value];
    let wait_info = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&values);
    unsafe {
        backend
            .get_device()
            .handle
            .wait_semaphores(&wait_info, timeout_ns)
            .map_err(|e| format!("Wait for timeline error: {}", e))
    }
}

/// Begin a new frame: wait for previous frame to complete, reset pools.
///
/// This function implements the frame-in-flight synchronization pattern:
///
/// 1. **Wait for previous frame**: Uses timeline semaphore to wait until the GPU
///    has finished processing the previous frame that used this slot
/// 2. **Reset pools**: Resets command pools and descriptor pools for reuse
/// 3. **Update timeline**: Increments the CPU timeline value
///
/// # Frame-in-Flight Pattern
///
/// The backend uses multiple frame contexts (typically 2-3) to allow the CPU
/// to record commands while the GPU is still processing previous frames.
/// Each frame context has its own command pool and descriptor pool.
///
/// ```text
/// Frame 0: [Record] → [Submit] → [GPU Process] → [Wait] → [Reset] → [Record] ...
/// Frame 1:            [Record] → [Submit] → [GPU Process] → [Wait] → [Reset] ...
/// Frame 2:                       [Record] → [Submit] → [GPU Process] → [Wait] ...
/// ```
pub fn begin_frame(backend: &mut VulkanBackend) {
    if backend.frame_started {
        return;
    }

    let device = backend.get_device().clone();
    
    let (num_contexts, timeline_sem) = {
        let g = backend.graphics.as_ref().expect("Graphics queue not initialized");
        (g.frame_contexts.len(), g.timeline_sem)
    };
    
    backend.global_frame_index = (backend.global_frame_index + 1) % num_contexts;
    backend.frame_count += 1;

    let graphics = backend.graphics.as_mut().unwrap();
    graphics.cpu_timeline += 1;
    let ctx = &mut graphics.frame_contexts[backend.global_frame_index];

    // Wait for this frame slot to be ready
    if ctx.last_completion_value > 0 {
        let semaphores = [timeline_sem];
        let values = [ctx.last_completion_value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe {
            device
                .handle
                .wait_semaphores(&wait_info, u64::MAX)
                .expect("Failed to wait for frame timeline");
        }
    }

    // Reset the pools for this frame
    unsafe {
        device
            .handle
            .reset_command_pool(ctx.command_pool, vk::CommandPoolResetFlags::empty())
            .expect("Failed to reset command pool");
        device
            .handle
            .reset_descriptor_pool(ctx.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .expect("Failed to reset descriptor pool");

        for tp_mutex in &ctx.per_thread_pools {
            let mut tp = tp_mutex.lock().unwrap();
            device
                .handle
                .reset_command_pool(tp.pool, vk::CommandPoolResetFlags::empty())
                .expect("Failed to reset thread command pool");
            device
                .handle
                .reset_descriptor_pool(tp.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .expect("Failed to reset thread descriptor pool");
            tp.cursor = 0;
        }
    }

    ctx.cursor = 0;
    ctx.submitted_cursor = 0;
    backend.frame_started = true;
}

/// End the current frame: run garbage collection.
///
/// This function performs cleanup of resources that are no longer needed:
/// - Destroys images, buffers, and samplers that were marked for deletion
/// - Recycles semaphores for reuse
/// - Cleans up dead descriptor sets
///
/// # Garbage Collection Strategy
///
/// Resources are not destroyed immediately when they become unused.
/// Instead, they are marked for deletion and destroyed at the end of the frame
/// when it's safe to do so (no GPU work is using them).
pub fn end_frame(backend: &mut VulkanBackend) {
    backend.garbage_collect();
    backend.frame_started = false;
}

/// Acquire the next swapchain image for a window.
///
/// This function acquires the next available image from the swapchain for rendering.
/// It handles swapchain recreation when the window is resized or becomes suboptimal.
///
/// # Arguments
///
/// * `backend` - Mutable reference to the backend
/// * `window` - Handle to the window
///
/// # Returns
///
/// `Ok(Some((image, semaphore_id, image_index)))` if an image was acquired,
/// `Ok(None)` if the window is minimized (zero extent),
/// `Err` on failure
///
/// # Swapchain Recreation
///
/// If the swapchain becomes suboptimal (e.g., window resized), it is invalidated
/// and recreated on the next call. This ensures optimal presentation performance.
pub fn acquire_swapchain_image(
    backend: &mut VulkanBackend,
    window: WindowHandle,
) -> Result<Option<(BackendImage, u64, u32)>, String> {
    let device = backend.get_device().clone();
    let frame_slot = backend.global_frame_index;

    loop {
        let (sc_handle, acquire_sem_id, semaphore) = {
            let mut sc_to_init = None;
            {
                let ctx = backend
                    .windows
                    .get(&window.0)
                    .ok_or("Invalid window handle")?;
                let size = ctx.raw.handle.drawable_size();
                if size.0 == 0 || size.1 == 0 {
                    return Ok(None);
                }

                if ctx.swapchain.is_none() {
                    sc_to_init = Some((ctx.raw.surface, size.0, size.1, ctx.config));
                }
            }

            if let Some((surface, w, h, config)) = sc_to_init {
                let sc_res = crate::swapchain::VulkanSwapchain::new(
                    device.clone(),
                    surface,
                    w,
                    h,
                    config,
                );

                match sc_res {
                    Ok(sc) => {
                        let num_images = sc.images.len();
                        let mut new_ids = Vec::with_capacity(num_images);
                        let mut new_sems = Vec::with_capacity(num_images);
                        for _ in 0..num_images {
                            let p_id = backend.create_semaphore();
                            let p_sem = backend.semaphores.get(p_id).cloned().unwrap();
                            new_ids.push(p_id);
                            new_sems.push(p_sem);
                        }

                        let ctx = backend.windows.get_mut(&window.0).unwrap();
                        ctx.swapchain = Some(sc);
                        
                        let old_ids: Vec<_> = ctx.present_semaphore_ids.drain(..).collect();
                        for id in old_ids {
                            if let Some(sem) = backend.semaphores.remove(id) {
                                backend.recycled_semaphores.push(sem);
                            }
                        }
                        ctx.present_semaphore_ids = new_ids;
                        ctx.present_semaphores = new_sems;
                    }
                    Err(e) if e == "ZeroExtent" => return Ok(None),
                    Err(e) => return Err(e),
                }
            }

            let ctx = backend.windows.get_mut(&window.0).unwrap();
            let swapchain = ctx.swapchain.as_ref().unwrap();
            let sem_id = ctx.acquire_semaphore_ids[frame_slot % ctx.acquire_semaphore_ids.len()];
            let sem = ctx.acquire_semaphores[frame_slot % ctx.acquire_semaphores.len()];

            (swapchain.handle, sem_id, sem)
        };

        let fp = backend.swapchain_loader.as_ref().unwrap();
        let res =
            unsafe { fp.acquire_next_image(sc_handle, u64::MAX, semaphore, vk::Fence::null()) };

        match res {
            Ok((index, suboptimal)) => {
                if suboptimal {
                    debug!("Swapchain is suboptimal, invalidating for recreation");

                    unsafe {
                        backend.get_device().handle.device_wait_idle().ok();
                    }

                    let (old_id, images_to_remove) = {
                        let ctx = backend.windows.get_mut(&window.0).expect("Window context missing");
                        let sc = ctx.swapchain.take().expect("Swapchain missing in suboptimal state");
                        let old_id = ctx.acquire_semaphore_ids[frame_slot % ctx.acquire_semaphore_ids.len()];
                        (old_id, sc.images.clone())
                    };

                    // Destroy old signaled semaphore and create a replacement
                    backend.destroy_semaphore_internal(old_id);
                    let new_id = backend.create_semaphore();
                    let new_sem = backend.semaphores.get(new_id).cloned().unwrap();

                    if let Some(ctx) = backend.windows.get_mut(&window.0) {
                        let idx = frame_slot % ctx.acquire_semaphore_ids.len();
                        ctx.acquire_semaphore_ids[idx] = new_id;
                        ctx.acquire_semaphores[idx] = new_sem;
                    }

                    backend.unregister_swapchain_images(&images_to_remove);
                    continue;
                }

                let ctx = backend.windows.get_mut(&window.0).unwrap();
                let swapchain = ctx.swapchain.as_ref().unwrap();
                ctx.current_acquire_sem_id = Some(acquire_sem_id);
                ctx.current_image_index = Some(index);

                let image_raw = swapchain.images[index as usize];
                let image_id = image_raw.as_raw();
                let arena_id = if let Some(&id) = backend.external_to_physical.get(&image_id) {
                    if let Some(img) = backend.images.get_mut(id) {
                        img.last_layout = vk::ImageLayout::UNDEFINED;
                        img.last_access = vk::AccessFlags2::empty();
                        img.last_stage = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
                    }
                    id
                } else {
                    let view_raw = swapchain.image_views[index as usize];
                    let new_id = backend.images.insert(crate::resource_arena::PhysicalImage {
                        image: image_raw,
                        view: view_raw,
                        allocation: None,
                        desc: ImageDesc::new(
                            swapchain.extent.width,
                            swapchain.extent.height,
                            crate::convert::convert_vk_format(swapchain.format),
                        ),
                        format: swapchain.format,
                        last_layout: vk::ImageLayout::UNDEFINED,
                        last_access: vk::AccessFlags2::empty(),
                        last_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        last_write_frame: 0,
                        last_queue_family: backend.graphics.as_ref().unwrap().family,
                    });
                    backend.external_to_physical.insert(image_id, new_id);
                    new_id
                };

                return Ok(Some((BackendImage(arena_id), acquire_sem_id, index)));
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                debug!("Swapchain out of date during acquire, invalidating...");
                let images_to_remove = {
                    let ctx = backend.windows.get_mut(&window.0).unwrap();
                    if let Some(sc) = ctx.swapchain.take() {
                        sc.images.clone()
                    } else {
                        Vec::new()
                    }
                };
                unsafe {
                    backend.get_device().handle.device_wait_idle().ok();
                }
                backend.unregister_swapchain_images(&images_to_remove);
                continue; // Loop and recreate
            }
            Err(e) => {
                return Err(format!("Failed to acquire swapchain image: {}", e));
            }
        }
    }
}

/// Submit a command batch to the GPU.
///
/// This function submits recorded command buffers to the graphics queue and
/// presents the rendered images to the screen.
///
/// # Submission Flow
///
/// 1. **Advance timeline**: Increment the CPU timeline value
/// 2. **Collect semaphores**: Gather binary semaphores from windows that acquired images
/// 3. **Submit to queue**: Submit command buffers with timeline and binary semaphore synchronization
/// 4. **Present**: Present rendered images to all active windows
///
/// # Synchronization
///
/// The submission uses a combination of:
/// - **Timeline semaphore**: For CPU-GPU synchronization (wait for previous frame)
/// - **Binary semaphores**: For GPU-GPU synchronization (acquire → render → present)
///
/// # Arguments
///
/// * `backend` - Mutable reference to the backend
/// * `batch` - Command batch containing command buffers to submit
/// * `_wait_sems` - Unused (reserved for future use)
/// * `_signal_sems` - Unused (reserved for future use)
///
/// # Returns
///
/// The timeline value that will be signaled when this submission completes
pub fn submit(
    backend: &mut VulkanBackend,
    batch: CommandBatch,
) -> Result<u64, String> {
    let device = backend.get_device().clone();

    // 1. Capture base timeline values for mapping relative compiler syncs to absolute timeline values
    let graphics_base = backend.graphics.as_ref().map(|q| q.cpu_timeline).unwrap_or(0);
    let compute_base = backend.compute.as_ref().map(|q| q.cpu_timeline).unwrap_or(0);
    let transfer_base = backend.transfer.as_ref().map(|q| q.cpu_timeline).unwrap_or(0);

    let get_base = |q: QueueType| match q {
        QueueType::Graphics => graphics_base,
        QueueType::AsyncCompute => compute_base,
        QueueType::Transfer => transfer_base,
    };

    // 2. Collect all binary semaphores from windows that acquired images
    let mut active_windows = Vec::with_capacity(2);
    for ctx in backend.windows.values_mut() {
        if let (Some(a_id), Some(i)) = (
            ctx.current_acquire_sem_id.take(),
            ctx.current_image_index.take(),
        ) {
            let release_sem = ctx.present_semaphores[i as usize];
            let acquire_sem = backend.semaphores.get(a_id).cloned().unwrap();
            active_windows.push((
                ctx.swapchain.as_ref().unwrap().handle,
                i,
                acquire_sem,
                release_sem,
            ));
        }
    }

    let mut present_info = Vec::with_capacity(active_windows.len());
    let mut graphics_wait_binary = Vec::with_capacity(active_windows.len());
    let mut graphics_signal_binary = Vec::with_capacity(active_windows.len());
    for (sc_handle, image_index, acquire_sem, release_sem) in active_windows {
        graphics_wait_binary.push(acquire_sem);
        graphics_signal_binary.push(release_sem);
        present_info.push((sc_handle, image_index, release_sem));
    }

    // 3. Perform submissions for each queue
    // We do them in order: Transfer, Compute, then Graphics (which handles presentation and acquires)

    // Extract sync metadata from batch
    let mut q_waits = HashMap::new();
    for (target, on, val) in &batch.waits {
        q_waits.entry(*target).or_insert_with(Vec::new).push((*on, *val));
    }

    let mut q_signals = HashMap::new();
    for (queue, val) in &batch.signals {
        q_signals.entry(*queue).or_insert_with(Vec::new).push(*val);
    }

    for queue_type in [
        QueueType::Transfer,
        QueueType::AsyncCompute,
        QueueType::Graphics,
    ] {
        let cmds = match queue_type {
            QueueType::Graphics => &batch.graphics_commands,
            QueueType::AsyncCompute => &batch.compute_commands,
            QueueType::Transfer => &batch.transfer_commands,
        };

        let relative_waits = q_waits.get(&queue_type).map(|v| v.as_slice()).unwrap_or(&[]);
        let relative_signals = q_signals.get(&queue_type).map(|v| v.as_slice()).unwrap_or(&[]);
        let wait_binary: &[vk::Semaphore] = if queue_type == QueueType::Graphics {
            &graphics_wait_binary
        } else {
            &[]
        };
        let signal_binary: &[vk::Semaphore] = if queue_type == QueueType::Graphics {
            &graphics_signal_binary
        } else {
            &[]
        };

        let q_ctx_exists = match queue_type {
            QueueType::Graphics => backend.graphics.is_some(),
            QueueType::AsyncCompute => backend.compute.is_some(),
            QueueType::Transfer => backend.transfer.is_some(),
        };

        if !q_ctx_exists {
            if !cmds.is_empty() {
                return Err(format!("Queue {:?} not available for commands", queue_type));
            }
            continue;
        }

        let mut vk_cmds: Vec<vk::CommandBuffer> = cmds
            .iter()
            .map(|cb| unsafe { std::mem::transmute::<u64, vk::CommandBuffer>(cb.0) })
            .collect();

        // If this is the graphics queue, also include legacy main pool recordings
        if queue_type == QueueType::Graphics {
            let q_ctx = backend.graphics.as_mut().unwrap();
            let frame_ctx = &mut q_ctx.frame_contexts[backend.global_frame_index];
            let legacy_cmds =
                &frame_ctx.allocated_command_buffers[frame_ctx.submitted_cursor..frame_ctx.cursor];
            vk_cmds.extend_from_slice(legacy_cmds);
            frame_ctx.submitted_cursor = frame_ctx.cursor;
        }

        if vk_cmds.is_empty()
            && relative_waits.is_empty()
            && relative_signals.is_empty()
            && wait_binary.is_empty()
            && signal_binary.is_empty()
        {
            continue;
        }

        let mut wait_sems = Vec::new();
        let mut wait_values = Vec::new();
        let mut wait_stages = Vec::new();

        // Add timeline waits
        for (on_queue, rel_value) in relative_waits {
            let (on_sem, on_base) = match on_queue {
                QueueType::Graphics => {
                    let ctx = backend.graphics.as_ref().unwrap();
                    (ctx.timeline_sem, graphics_base)
                }
                QueueType::AsyncCompute => {
                    let ctx = backend.compute.as_ref().unwrap();
                    (ctx.timeline_sem, compute_base)
                }
                QueueType::Transfer => {
                    let ctx = backend.transfer.as_ref().unwrap();
                    (ctx.timeline_sem, transfer_base)
                }
            };
            wait_sems.push(on_sem);
            wait_values.push(on_base + rel_value);
            wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
        }

        // Add binary waits (typically only for graphics queue)
        for &sem in wait_binary {
            wait_sems.push(sem);
            wait_values.push(0);
            wait_stages.push(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
        }

        let mut signal_sems = Vec::new();
        let mut signal_values = Vec::new();

        let (q_handle, q_timeline_sem, q_base, mut q_cpu_timeline) = {
            let q_ctx = match queue_type {
                QueueType::Graphics => backend.graphics.as_mut().unwrap(),
                QueueType::AsyncCompute => backend.compute.as_mut().unwrap(),
                QueueType::Transfer => backend.transfer.as_mut().unwrap(),
            };
            (
                q_ctx.queue,
                q_ctx.timeline_sem,
                get_base(queue_type),
                q_ctx.cpu_timeline,
            )
        };

        // Add timeline signals
        for rel_value in relative_signals {
            signal_sems.push(q_timeline_sem);
            let abs_value = q_base + rel_value;
            signal_values.push(abs_value);
            q_cpu_timeline = q_cpu_timeline.max(abs_value);
        }

        // Add binary signals
        for &sem in signal_binary {
            signal_sems.push(sem);
            signal_values.push(0);
        }

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);

        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_info)
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&vk_cmds)
            .signal_semaphores(&signal_sems);

        unsafe {
            device
                .handle
                .queue_submit(q_handle, &[submit_info], vk::Fence::null())
                .map_err(|e| e.to_string())?;
        }

        // Update cpu_timeline in backend
        {
            let q_ctx = match queue_type {
                QueueType::Graphics => backend.graphics.as_mut().unwrap(),
                QueueType::AsyncCompute => backend.compute.as_mut().unwrap(),
                QueueType::Transfer => backend.transfer.as_mut().unwrap(),
            };
            q_ctx.cpu_timeline = q_cpu_timeline;
        }
    }

    // 4. Present all windows
    let fp = backend.swapchain_loader.as_ref().unwrap();
    let graphics_queue = backend.graphics.as_ref().unwrap().queue;

    for (swapchain, index, wait_sem) in present_info {
        let scs = [swapchain];
        let idxs = [index];
        let sems = [wait_sem];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&sems)
            .swapchains(&scs)
            .image_indices(&idxs);

        unsafe {
            fp.queue_present(graphics_queue, &present_info).ok(); // Presentation errors handled on next acquire
        }
    }

    // Final result is the graphics timeline value
    let graphics_signal = backend.graphics.as_ref().unwrap().cpu_timeline;

    // Update frame slot state
    {
        let graphics = backend.graphics.as_mut().unwrap();
        let frame_ctx = &mut graphics.frame_contexts[backend.global_frame_index];
        frame_ctx.last_completion_value = graphics_signal;
    }

    Ok(graphics_signal)
}
