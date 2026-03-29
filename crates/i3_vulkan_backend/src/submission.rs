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
use i3_gfx::graph::backend::{BatchStep, CommandBatch, *};
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

    // Wait for compute/transfer frame slots before resetting their pools.
    // Graphics wait (above) only covers the graphics queue; compute and transfer
    // are independent queues that may still be processing frame N work.
    let frame_index = backend.global_frame_index;
    let compute_wait = backend.compute.as_ref()
        .map(|c| (c.timeline_sem, c.frame_contexts[frame_index].last_completion_value));
    let transfer_wait = backend.transfer.as_ref()
        .map(|t| (t.timeline_sem, t.frame_contexts[frame_index].last_completion_value));

    if let Some((sem, value)) = compute_wait {
        if value > 0 {
            let semaphores = [sem];
            let values = [value];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            unsafe {
                device.handle.wait_semaphores(&wait_info, u64::MAX)
                    .expect("Failed to wait for compute frame timeline");
            }
        }
    }
    if let Some((sem, value)) = transfer_wait {
        if value > 0 {
            let semaphores = [sem];
            let values = [value];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            unsafe {
                device.handle.wait_semaphores(&wait_info, u64::MAX)
                    .expect("Failed to wait for transfer frame timeline");
            }
        }
    }

    if let Some(compute) = backend.compute.as_mut() {
        let extra_ctx = &mut compute.frame_contexts[frame_index];
        unsafe {
            device.handle.reset_command_pool(extra_ctx.command_pool, vk::CommandPoolResetFlags::empty())
                .expect("Failed to reset compute command pool");
            device.handle.reset_descriptor_pool(extra_ctx.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .expect("Failed to reset compute descriptor pool");
            for tp_mutex in &extra_ctx.per_thread_pools {
                let mut tp = tp_mutex.lock().unwrap();
                device.handle.reset_command_pool(tp.pool, vk::CommandPoolResetFlags::empty())
                    .expect("Failed to reset compute thread command pool");
                device.handle.reset_descriptor_pool(tp.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                    .expect("Failed to reset compute thread descriptor pool");
                tp.cursor = 0;
            }
        }
        extra_ctx.cursor = 0;
        extra_ctx.submitted_cursor = 0;
    }

    if let Some(transfer) = backend.transfer.as_mut() {
        let extra_ctx = &mut transfer.frame_contexts[frame_index];
        unsafe {
            device.handle.reset_command_pool(extra_ctx.command_pool, vk::CommandPoolResetFlags::empty())
                .expect("Failed to reset transfer command pool");
            device.handle.reset_descriptor_pool(extra_ctx.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .expect("Failed to reset transfer descriptor pool");
            for tp_mutex in &extra_ctx.per_thread_pools {
                let mut tp = tp_mutex.lock().unwrap();
                device.handle.reset_command_pool(tp.pool, vk::CommandPoolResetFlags::empty())
                    .expect("Failed to reset transfer thread command pool");
                device.handle.reset_descriptor_pool(tp.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                    .expect("Failed to reset transfer thread descriptor pool");
                tp.cursor = 0;
            }
        }
        extra_ctx.cursor = 0;
        extra_ctx.submitted_cursor = 0;
    }

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
                            let sem_id = backend.create_semaphore(false);
                            let p_sem = backend.semaphores.get(sem_id).cloned().unwrap();
                            new_ids.push(sem_id);
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
                    let new_id = backend.create_semaphore(false);
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
                        last_queue_family: backend.graphics_family,
                        is_swapchain: true,
                        concurrent: false, // swapchain images are always EXCLUSIVE
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

/// A single pending sub-batch for one queue, accumulating commands/waits/signals
/// before the next vkQueueSubmit call.
struct SubBatch {
    commands: Vec<vk::CommandBuffer>,
    wait_sems: Vec<vk::Semaphore>,
    wait_values: Vec<u64>,
    wait_stages: Vec<vk::PipelineStageFlags>,
    signal_sems: Vec<vk::Semaphore>,
    signal_values: Vec<u64>,
}

impl SubBatch {
    fn new() -> Self {
        Self {
            commands: Vec::new(),
            wait_sems: Vec::new(),
            wait_values: Vec::new(),
            wait_stages: Vec::new(),
            signal_sems: Vec::new(),
            signal_values: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.commands.is_empty()
            && self.wait_sems.is_empty()
            && self.signal_sems.is_empty()
    }
}

/// Submit a command batch to the GPU.
///
/// Processes the ordered BatchSteps to emit separate vkQueueSubmit calls at each
/// Signal boundary. This prevents cross-queue deadlocks when Graphics → Compute
/// → Graphics dependency chains exist — each "phase" of a queue gets its own
/// submission with only the waits that apply to that phase.
pub fn submit(
    backend: &mut VulkanBackend,
    batch: CommandBatch,
) -> Result<u64, String> {
    let device = backend.get_device().clone();

    // 1. Snapshot base timeline values (before any signals this frame)
    let graphics_base = backend.graphics.as_ref().map(|q| q.cpu_timeline).unwrap_or(0);
    let compute_base  = backend.compute.as_ref().map(|q| q.cpu_timeline).unwrap_or(0);
    let transfer_base = backend.transfer.as_ref().map(|q| q.cpu_timeline).unwrap_or(0);

    let (graphics_timeline, graphics_family) = backend.graphics.as_ref()
        .map(|q| (q.timeline_sem, q.family)).unwrap_or((vk::Semaphore::null(), 0));
    let (compute_timeline,  compute_family)  = backend.compute.as_ref()
        .map(|q| (q.timeline_sem, q.family)).unwrap_or((vk::Semaphore::null(), 0));
    let (transfer_timeline, transfer_family) = backend.transfer.as_ref()
        .map(|q| (q.timeline_sem, q.family)).unwrap_or((vk::Semaphore::null(), 0));

    let base_for = |q: QueueType| match q {
        QueueType::Graphics    => graphics_base,
        QueueType::AsyncCompute => compute_base,
        QueueType::Transfer    => transfer_base,
    };
    let timeline_sem_for = |q: QueueType| match q {
        QueueType::Graphics    => graphics_timeline,
        QueueType::AsyncCompute => compute_timeline,
        QueueType::Transfer    => transfer_timeline,
    };
    let family_for = |q: QueueType| match q {
        QueueType::Graphics    => graphics_family,
        QueueType::AsyncCompute => compute_family,
        QueueType::Transfer    => transfer_family,
    };
    let queue_handle_for = |backend: &VulkanBackend, q: QueueType| match q {
        QueueType::Graphics    => backend.graphics.as_ref().unwrap().queue,
        QueueType::AsyncCompute => backend.compute.as_ref().unwrap().queue,
        QueueType::Transfer    => backend.transfer.as_ref().unwrap().queue,
    };

    // 2. Collect swapchain binary semaphores
    let mut active_windows = Vec::with_capacity(2);
    for ctx in backend.windows.values_mut() {
        if let (Some(a_id), Some(i)) = (ctx.current_acquire_sem_id.take(), ctx.current_image_index.take()) {
            let release_sem = ctx.present_semaphores[i as usize];
            let acquire_sem = backend.semaphores.get(a_id).cloned().unwrap();
            active_windows.push((ctx.swapchain.as_ref().unwrap().handle, i, acquire_sem, release_sem));
        }
    }

    let mut present_info_list = Vec::with_capacity(active_windows.len());

    // Find which family last touched the swapchain image for present signal assignment
    let mut swapchain_last_family = graphics_family;
    for (_, img) in backend.images.iter() {
        if img.is_swapchain && img.last_layout != vk::ImageLayout::UNDEFINED {
            swapchain_last_family = img.last_queue_family;
            break;
        }
    }

    for (sc_handle, image_index, _acquire_sem, release_sem) in &active_windows {
        present_info_list.push((*sc_handle, *image_index, *release_sem));
    }

    // 3. Process ordered steps → build ordered list of (queue, SubBatch)
    // Per-queue open sub-batch (not yet submitted)
    let mut open: HashMap<QueueType, SubBatch> = HashMap::new();
    // Finalized sub-batches in submission order
    let mut finalized: Vec<(QueueType, SubBatch)> = Vec::new();

    // Track the highest absolute signal value per queue (for cpu_timeline update)
    let mut max_signal: HashMap<QueueType, u64> = HashMap::new();

    let global_frame_index = backend.global_frame_index;

    for step in &batch.steps {
        match step {
            BatchStep::Command { queue, cb } => {
                let sub = open.entry(*queue).or_insert_with(SubBatch::new);
                sub.commands.push(unsafe { std::mem::transmute::<u64, vk::CommandBuffer>(cb.0) });
            }
            BatchStep::Wait { queue, on, value } => {
                let sub = open.entry(*queue).or_insert_with(SubBatch::new);
                let abs = base_for(*on) + value;
                sub.wait_sems.push(timeline_sem_for(*on));
                sub.wait_values.push(abs);
                sub.wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
            BatchStep::Signal { queue, value } => {
                let abs = (base_for(*queue) + value).max(1);
                let sub = open.entry(*queue).or_insert_with(SubBatch::new);
                sub.signal_sems.push(timeline_sem_for(*queue));
                sub.signal_values.push(abs);
                *max_signal.entry(*queue).or_insert(0) = (*max_signal.get(queue).unwrap_or(&0)).max(abs);

                // Close this sub-batch and start fresh for the next phase
                let closed = open.remove(queue).unwrap();
                finalized.push((*queue, closed));
            }
        }
    }

    // Flush any remaining open sub-batches (passes with no trailing Signal)
    // Append legacy graphics commands (allocated outside the graph) to the last Graphics sub
    {
        let legacy: Vec<vk::CommandBuffer> = if let Some(gfx) = backend.graphics.as_mut() {
            let ctx = &mut gfx.frame_contexts[global_frame_index];
            let cmds = ctx.allocated_command_buffers[ctx.submitted_cursor..ctx.cursor].to_vec();
            ctx.submitted_cursor = ctx.cursor;
            cmds
        } else {
            Vec::new()
        };

        // Find or create the last Graphics open sub-batch and append legacy commands
        if !legacy.is_empty() {
            let sub = open.entry(QueueType::Graphics).or_insert_with(SubBatch::new);
            sub.commands.extend(legacy);
        }
    }

    for (q, sub) in open {
        if !sub.is_empty() {
            finalized.push((q, sub));
        }
    }

    // 4. Attach binary semaphores (swapchain acquire/present) to the correct sub-batches.
    // Acquire binary → first Graphics sub-batch (or the only one).
    // Present binary → last sub-batch that belongs to the queue family that last touched the swapchain.
    let first_graphics_idx = finalized.iter().position(|(q, _)| *q == QueueType::Graphics);
    let last_swapchain_idx = finalized.iter().rposition(|(q, _)| family_for(*q) == swapchain_last_family);

    if let Some(idx) = first_graphics_idx {
        for (_, _, acquire_sem, _) in &active_windows {
            finalized[idx].1.wait_sems.push(*acquire_sem);
            finalized[idx].1.wait_values.push(0);
            finalized[idx].1.wait_stages.push(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
        }
    }
    if let Some(idx) = last_swapchain_idx {
        for (_, _, _, release_sem) in &active_windows {
            finalized[idx].1.signal_sems.push(*release_sem);
            finalized[idx].1.signal_values.push(0);
        }
    }

    // 5. Submit in order (finalization order = correct CPU submission order)
    for (queue_type, sub) in finalized {
        if sub.is_empty() {
            continue;
        }

        let q_handle = queue_handle_for(backend, queue_type);

        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&sub.wait_values)
            .signal_semaphore_values(&sub.signal_values);

        let submit_info = vk::SubmitInfo::default()
            .push_next(&mut timeline_info)
            .wait_semaphores(&sub.wait_sems)
            .wait_dst_stage_mask(&sub.wait_stages)
            .command_buffers(&sub.commands)
            .signal_semaphores(&sub.signal_sems);

        unsafe {
            device
                .handle
                .queue_submit(q_handle, &[submit_info], vk::Fence::null())
                .map_err(|e| e.to_string())?;
        }
    }

    // 6. Update cpu_timeline for all queues
    let graphics_signal = {
        if let Some(gfx) = backend.graphics.as_mut() {
            if let Some(&v) = max_signal.get(&QueueType::Graphics) {
                gfx.cpu_timeline = gfx.cpu_timeline.max(v);
            }
            gfx.cpu_timeline
        } else { 0 }
    };
    if let Some(compute) = backend.compute.as_mut() {
        if let Some(&v) = max_signal.get(&QueueType::AsyncCompute) {
            compute.cpu_timeline = compute.cpu_timeline.max(v);
            compute.frame_contexts[global_frame_index].last_completion_value = v;
        }
    }
    if let Some(transfer) = backend.transfer.as_mut() {
        if let Some(&v) = max_signal.get(&QueueType::Transfer) {
            transfer.cpu_timeline = transfer.cpu_timeline.max(v);
            transfer.frame_contexts[global_frame_index].last_completion_value = v;
        }
    }

    // 7. Present all windows
    let fp = backend.swapchain_loader.as_ref().unwrap();
    let graphics_queue = backend.graphics.as_ref().unwrap().queue;
    for (swapchain, index, wait_sem) in present_info_list {
        let scs = [swapchain];
        let idxs = [index];
        let sems = [wait_sem];
        let pi = vk::PresentInfoKHR::default()
            .wait_semaphores(&sems)
            .swapchains(&scs)
            .image_indices(&idxs);
        unsafe { fp.queue_present(graphics_queue, &pi).ok(); }
    }

    // 8. Update frame completion value for next-frame wait
    {
        let graphics = backend.graphics.as_mut().unwrap();
        let frame_ctx = &mut graphics.frame_contexts[global_frame_index];
        frame_ctx.last_completion_value = graphics_signal;
    }

    Ok(graphics_signal)
}
