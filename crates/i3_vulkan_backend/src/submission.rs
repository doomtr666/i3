use ash::vk;
use ash::vk::Handle;
use i3_gfx::graph::backend::*;
use i3_gfx::graph::types::*;
use tracing::debug;

use crate::backend::VulkanBackend;

/// Wait for the timeline semaphore to reach a specific value on the host (CPU).
pub fn wait_for_timeline(
    backend: &VulkanBackend,
    value: u64,
    timeout_ns: u64,
) -> Result<(), String> {
    let semaphores = [backend.timeline_sem];
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
pub fn begin_frame(backend: &mut VulkanBackend) {
    if backend.frame_started {
        return;
    }

    let device = backend.get_device().clone();
    backend.global_frame_index = (backend.global_frame_index + 1) % backend.frame_contexts.len();
    backend.frame_count += 1;
    backend.cpu_timeline += 1;

    let ctx = &mut backend.frame_contexts[backend.global_frame_index];

    // Wait for this frame slot to be ready
    if ctx.last_completion_value > 0 {
        let semaphores = [backend.timeline_sem];
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
            tp.cursor = 0;
        }
    }

    ctx.cursor = 0;
    ctx.submitted_cursor = 0;
    backend.frame_started = true;
}

/// End the current frame: run garbage collection.
pub fn end_frame(backend: &mut VulkanBackend) {
    backend.garbage_collect();
    backend.frame_started = false;
}

/// Acquire the next swapchain image for a window.
pub fn acquire_swapchain_image(
    backend: &mut VulkanBackend,
    window: WindowHandle,
) -> Result<Option<(BackendImage, u64, u32)>, String> {
    let device = backend.get_device().clone();
    let frame_slot = backend.global_frame_index;

    loop {
        let (sc_handle, acquire_sem_id, semaphore) = {
            let ctx = backend
                .windows
                .get_mut(&window.0)
                .ok_or("Invalid window handle")?;
            let size = ctx.raw.handle.drawable_size();
            if size.0 == 0 || size.1 == 0 {
                return Ok(None);
            }

            if ctx.swapchain.is_none() {
                let sc_res = crate::swapchain::VulkanSwapchain::new(
                    device.clone(),
                    ctx.raw.surface,
                    size.0,
                    size.1,
                    ctx.config,
                );

                match sc_res {
                    Ok(sc) => ctx.swapchain = Some(sc),
                    Err(e) if e == "ZeroExtent" => return Ok(None),
                    Err(e) => return Err(e),
                }
            }

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
                    let images_to_remove = {
                        let ctx = backend.windows.get_mut(&window.0).unwrap();
                        let sc = ctx.swapchain.take().unwrap();
                        let imgs = sc.images.clone();
                        ctx.swapchain = Some(sc); // Put it back if we still want to use it
                        imgs
                    };
                    unsafe {
                        backend.get_device().handle.device_wait_idle().ok();
                    }
                    backend.unregister_swapchain_images(&images_to_remove);
                    let ctx = backend.windows.get_mut(&window.0).unwrap();
                    ctx.swapchain = None;
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
pub fn submit(
    backend: &mut VulkanBackend,
    batch: CommandBatch,
    _wait_sems: &[u64],
    _signal_sems: &[u64],
) -> Result<u64, String> {
    // Timeline advancement
    backend.cpu_timeline += 1;
    let signal_value = backend.cpu_timeline;

    // Collect all binary semaphores from windows that acquired images
    // 1. Collect Active Window Contexts (Borrow scope)
    let mut active_windows = Vec::with_capacity(2);
    let frame_slot = backend.global_frame_index;
    for ctx in backend.windows.values_mut() {
        if let (Some(a_id), Some(i)) = (
            ctx.current_acquire_sem_id.take(),
            ctx.current_image_index.take(),
        ) {
            let release_sem = ctx.present_semaphores[frame_slot % ctx.present_semaphores.len()];
            let acquire_sem = backend.semaphores.get(a_id).cloned().unwrap();
            active_windows.push((
                ctx.swapchain.as_ref().unwrap().handle,
                i,
                acquire_sem,
                release_sem,
            ));
        }
    }

    // 2. Process Binary Semaphores (Outside borrow scope)
    let mut wait_binary: Vec<vk::Semaphore> = Vec::with_capacity(active_windows.len());
    let mut signal_binary: Vec<vk::Semaphore> = Vec::with_capacity(active_windows.len());
    let mut present_info = Vec::with_capacity(active_windows.len());

    for (sc_handle, image_index, acquire_sem, release_sem) in active_windows {
        wait_binary.push(acquire_sem);
        signal_binary.push(release_sem);
        present_info.push((sc_handle, image_index, release_sem));
    }

    let device = backend.get_device().clone();

    let wait_values = [0u64; 8];
    let mut signal_values = [0u64; 8];
    signal_values[0] = signal_value;

    let num_binary = signal_binary.len();
    let mut all_signals = Vec::with_capacity(num_binary + 1);
    all_signals.push(backend.timeline_sem);
    all_signals.extend(&signal_binary);

    let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT; 8];

    let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&wait_values[..wait_binary.len()])
        .signal_semaphore_values(&signal_values[..all_signals.len()]);

    let submit_info = vk::SubmitInfo::default()
        .push_next(&mut timeline_info)
        .wait_semaphores(&wait_binary)
        .wait_dst_stage_mask(&wait_stages[..wait_binary.len()])
        .signal_semaphores(&all_signals);

    // Collect all command buffers: Batch + any legacy main pool recordings
    let mut cmds: Vec<vk::CommandBuffer> = batch
        .command_buffers
        .iter()
        .map(|cb| unsafe { std::mem::transmute::<u64, vk::CommandBuffer>(cb.0) })
        .collect();

    let frame_ctx = &mut backend.frame_contexts[backend.global_frame_index];
    let legacy_cmds =
        &frame_ctx.allocated_command_buffers[frame_ctx.submitted_cursor..frame_ctx.cursor];
    cmds.extend_from_slice(legacy_cmds);

    let submit_info = submit_info.command_buffers(&cmds);

    unsafe {
        device
            .handle
            .queue_submit(device.graphics_queue, &[submit_info], vk::Fence::null())
            .map_err(|e| e.to_string())?;
    }

    // Update submitted_cursor to current cursor
    frame_ctx.submitted_cursor = frame_ctx.cursor;

    // Present all windows
    let fp = backend.swapchain_loader.as_ref().unwrap();
    for (swapchain, index, wait_sem) in present_info {
        let swapchains = [swapchain];
        let indices = [index];
        let wait_sems = [wait_sem];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_sems)
            .swapchains(&swapchains)
            .image_indices(&indices);

        unsafe {
            fp.queue_present(device.graphics_queue, &present_info).ok(); // Presentation errors handled on next acquire
        }
    }

    // Advance slot's last completion value
    frame_ctx.last_completion_value = signal_value;

    Ok(signal_value)
}
