use ash::vk;
use tracing::debug;
use vk_mem::Alloc;

use i3_gfx::graph::backend::*;
use i3_gfx::graph::types::*;

use crate::backend::VulkanBackend;
use crate::resource_arena::PhysicalImage;

pub fn create_image(backend: &mut VulkanBackend, desc: &ImageDesc) -> BackendImage {
    let device = backend.get_device().clone();
    debug!("Creating Image: {:?}", desc);

    let extent = vk::Extent3D {
        width: desc.width,
        height: desc.height,
        depth: desc.depth,
    };

    // Translate format
    let format = crate::convert::convert_format(desc.format);

    // Use provided usage flags, but add common bits for flexibility
    let mut usage = crate::convert::convert_image_usage_flags(desc.usage);
    usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;

    let create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(desc.mip_levels.max(1))
        .array_layers(desc.array_layers.max(1))
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let allocation_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::AutoPreferDevice,
        ..Default::default()
    };

    let (image, allocation) = unsafe {
        let allocator = device.allocator.lock().unwrap();
        allocator
            .create_image(&create_info, &allocation_info)
            .expect("Failed to create image")
    };

    // Create View
    let aspect_mask = if format == vk::Format::D32_SFLOAT {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    };

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(crate::convert::convert_image_view_type(desc.view_type))
        .format(format)
        .components(crate::convert::convert_component_mapping(desc.swizzle))
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: desc.mip_levels.max(1),
            base_array_layer: 0,
            layer_count: desc.array_layers.max(1),
        });

    let view = unsafe { device.handle.create_image_view(&view_info, None) }.unwrap();

    let physical = PhysicalImage {
        image,
        view,
        allocation: Some(allocation),
        format,
        desc: desc.clone(),
        last_layout: vk::ImageLayout::UNDEFINED,
        last_access: vk::AccessFlags2::empty(),
        last_stage: vk::PipelineStageFlags2::NONE,
        last_write_frame: 0,
    };

    let id = backend.images.insert(physical);
    BackendImage(id)
}

pub fn destroy_image(backend: &mut VulkanBackend, handle: BackendImage) {
    if let Some(img) = backend.images.remove(handle.0) {
        if let Some(alloc) = img.allocation {
            backend
                .dead_images
                .push((backend.frame_count, img.image, img.view, alloc));
        }
    }
}

pub fn create_buffer(backend: &mut VulkanBackend, desc: &BufferDesc) -> BackendBuffer {
    let device = backend.get_device().clone();
    debug!("Creating Buffer: {:?}", desc);

    let create_info = vk::BufferCreateInfo::default()
        .size(desc.size)
        .usage(crate::convert::convert_buffer_usage_flags(desc.usage))
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let (mem_usage, alloc_flags) = match desc.memory {
        MemoryType::GpuOnly => (
            vk_mem::MemoryUsage::AutoPreferDevice,
            vk_mem::AllocationCreateFlags::empty(),
        ),
        MemoryType::CpuToGpu => (
            vk_mem::MemoryUsage::AutoPreferHost,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | vk_mem::AllocationCreateFlags::MAPPED,
        ),
        MemoryType::GpuToCpu => (
            vk_mem::MemoryUsage::AutoPreferHost,
            vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM
                | vk_mem::AllocationCreateFlags::MAPPED,
        ),
    };

    let allocation_info = vk_mem::AllocationCreateInfo {
        usage: mem_usage,
        flags: alloc_flags,
        ..Default::default()
    };

    let (buffer, allocation) = unsafe {
        let allocator = device.allocator.lock().unwrap();
        allocator
            .create_buffer(&create_info, &allocation_info)
            .expect("Failed to create buffer")
    };

    let physical = crate::resource_arena::PhysicalBuffer {
        buffer,
        allocation: Some(allocation),
        desc: desc.clone(),
        last_access: vk::AccessFlags2::empty(),
        last_stage: vk::PipelineStageFlags2::NONE,
    };

    let id = backend.buffers.insert(physical);
    BackendBuffer(id)
}

pub fn destroy_buffer(backend: &mut VulkanBackend, handle: BackendBuffer) {
    if let Some(buf) = backend.buffers.remove(handle.0) {
        if let Some(alloc) = buf.allocation {
            backend
                .dead_buffers
                .push((backend.frame_count, buf.buffer, alloc));
        }
    }
}

pub fn create_sampler(backend: &mut VulkanBackend, desc: &SamplerDesc) -> SamplerHandle {
    let mag_filter = match desc.mag_filter {
        Filter::Nearest => vk::Filter::NEAREST,
        Filter::Linear => vk::Filter::LINEAR,
    };

    let min_filter = match desc.min_filter {
        Filter::Nearest => vk::Filter::NEAREST,
        Filter::Linear => vk::Filter::LINEAR,
    };

    let mipmap_mode = match desc.mipmap_mode {
        MipmapMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        MipmapMode::Linear => vk::SamplerMipmapMode::LINEAR,
    };

    let convert_address = |mode: AddressMode| match mode {
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        AddressMode::MirrorClampToEdge => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
    };

    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(mag_filter)
        .min_filter(min_filter)
        .mipmap_mode(mipmap_mode)
        .address_mode_u(convert_address(desc.address_mode_u))
        .address_mode_v(convert_address(desc.address_mode_v))
        .address_mode_w(convert_address(desc.address_mode_w))
        .anisotropy_enable(desc.anisotropy > 1)
        .max_anisotropy(desc.anisotropy as f32)
        .min_lod(0.0)
        .max_lod(vk::LOD_CLAMP_NONE);

    let sampler = unsafe {
        backend
            .get_device()
            .handle
            .create_sampler(&create_info, None)
            .expect("Failed to create sampler")
    };

    let handle = backend.samplers.insert(sampler);
    SamplerHandle(handle)
}

pub fn destroy_sampler(backend: &mut VulkanBackend, handle: SamplerHandle) {
    if let Some(sampler) = backend.samplers.remove(handle.0) {
        backend.dead_samplers.push((backend.frame_count, sampler));
    }
}

pub fn upload_buffer(
    backend: &mut VulkanBackend,
    handle: BackendBuffer,
    data: &[u8],
    offset: u64,
) -> Result<(), String> {
    let device = backend.get_device().clone();
    if let Some(buf) = backend.buffers.get_mut(handle.0) {
        if let Some(alloc) = &mut buf.allocation {
            unsafe {
                let allocator = device.allocator.lock().unwrap();
                let ptr = allocator.map_memory(alloc).map_err(|e| e.to_string())?;

                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset as usize), data.len());

                let _ = allocator.flush_allocation(alloc, offset, data.len() as u64);
                allocator.unmap_memory(alloc);
            }

            // Update state to reflect HOST_WRITE
            buf.last_access = vk::AccessFlags2::HOST_WRITE;
            buf.last_stage = vk::PipelineStageFlags2::HOST;
            Ok(())
        } else {
            Err("Buffer has no allocation (external?)".to_string())
        }
    } else {
        Err(format!("Buffer not found: {:?}", handle))
    }
}

pub fn upload_image(
    backend: &mut VulkanBackend,
    handle: BackendImage,
    data: &[u8],
    mip_level: u32,
    array_layer: u32,
) -> Result<(), String> {
    let device = backend.get_device().clone();

    let physical = backend
        .images
        .get(handle.0)
        .ok_or_else(|| format!("Image not found: {:?}", handle))?;

    let image = physical.image;
    let width = physical.desc.width;
    let height = physical.desc.height;
    let depth = physical.desc.depth;

    // 1. Create Staging Buffer
    let create_info = vk::BufferCreateInfo::default()
        .size(data.len() as u64)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let allocation_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::AutoPreferHost,
        flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
            | vk_mem::AllocationCreateFlags::MAPPED,
        ..Default::default()
    };

    let (staging_buffer, mut staging_alloc) = unsafe {
        let allocator = device.allocator.lock().unwrap();
        let mut res = allocator
            .create_buffer(&create_info, &allocation_info)
            .map_err(|e| e.to_string())?;

        // 2. Copy Data
        let ptr = allocator
            .map_memory(&mut res.1)
            .map_err(|e| e.to_string())?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        allocator.unmap_memory(&mut res.1);
        res
    };

    // 3. Command Buffer for Transfer
    unsafe {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.graphics_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);

        let cmd_pool = device
            .handle
            .create_command_pool(&pool_info, None)
            .map_err(|e| e.to_string())?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = device
            .handle
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| e.to_string())?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .handle
            .begin_command_buffer(cmd, &begin_info)
            .map_err(|e| e.to_string())?;

        // Transition to TRANSFER_DST
        let barrier_to_dst = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::empty())
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: mip_level,
                level_count: 1,
                base_array_layer: array_layer,
                layer_count: 1,
            });

        let barriers = [barrier_to_dst];
        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        device.handle.cmd_pipeline_barrier2(cmd, &dependency);

        // Copy Buffer to Image
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level,
                base_array_layer: array_layer,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: (width >> mip_level).max(1),
                height: (height >> mip_level).max(1),
                depth: (depth >> mip_level).max(1),
            });

        device.handle.cmd_copy_buffer_to_image(
            cmd,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        // Transition to SHADER_READ
        let barrier_to_read = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            )
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: mip_level,
                level_count: 1,
                base_array_layer: array_layer,
                layer_count: 1,
            });

        let barriers2 = [barrier_to_read];
        let dependency2 = vk::DependencyInfo::default().image_memory_barriers(&barriers2);
        device.handle.cmd_pipeline_barrier2(cmd, &dependency2);

        device
            .handle
            .end_command_buffer(cmd)
            .map_err(|e| e.to_string())?;

        // Submit
        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);

        device
            .handle
            .queue_submit(device.graphics_queue, &[submit_info], vk::Fence::null())
            .map_err(|e| e.to_string())?;

        device
            .handle
            .device_wait_idle()
            .map_err(|e| e.to_string())?;

        // Cleanup
        device.handle.destroy_command_pool(cmd_pool, None);

        let allocator = device.allocator.lock().unwrap();
        allocator.destroy_buffer(staging_buffer, &mut staging_alloc);
    }

    // Update tracking state
    if let Some(img) = backend.images.get_mut(handle.0) {
        img.last_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        img.last_access = vk::AccessFlags2::SHADER_READ;
        img.last_stage =
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER;
    }

    Ok(())
}

pub fn register_external_image(
    backend: &mut VulkanBackend,
    handle: ImageHandle,
    physical: BackendImage,
) {
    backend.external_to_physical.insert(handle.0.0, physical.0);
}

pub fn register_external_buffer(
    backend: &mut VulkanBackend,
    handle: BufferHandle,
    physical: BackendBuffer,
) {
    backend
        .external_buffer_to_physical
        .insert(handle.0.0, physical.0);
}

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

// --- Transient Resource Management (Pooling) ---

pub fn create_transient_image(backend: &mut VulkanBackend, desc: &ImageDesc) -> BackendImage {
    if let Some(pool) = backend.transient_image_pool.get_mut(desc) {
        if let Some(id) = pool.pop() {
            return BackendImage(id);
        }
    }
    create_image(backend, desc)
}

pub fn create_transient_buffer(backend: &mut VulkanBackend, desc: &BufferDesc) -> BackendBuffer {
    if let Some(pool) = backend.transient_buffer_pool.get_mut(desc) {
        if let Some(id) = pool.pop() {
            return BackendBuffer(id);
        }
    }
    create_buffer(backend, desc)
}

pub fn release_transient_image(backend: &mut VulkanBackend, handle: BackendImage) {
    if let Some(img) = backend.images.get(handle.0) {
        let desc = img.desc.clone();
        backend
            .transient_image_pool
            .entry(desc)
            .or_default()
            .push(handle.0);
    }
}

pub fn release_transient_buffer(backend: &mut VulkanBackend, handle: BackendBuffer) {
    if let Some(buf) = backend.buffers.get(handle.0) {
        let desc = buf.desc.clone();
        backend
            .transient_buffer_pool
            .entry(desc)
            .or_default()
            .push(handle.0);
    }
}

pub fn resolve_image(backend: &VulkanBackend, handle: ImageHandle) -> BackendImage {
    if let Some(&physical_id) = backend.external_to_physical.get(&handle.0.0) {
        BackendImage(physical_id)
    } else {
        BackendImage(handle.0.0)
    }
}

pub fn resolve_buffer(backend: &VulkanBackend, handle: BufferHandle) -> BackendBuffer {
    if let Some(&physical_id) = backend.external_buffer_to_physical.get(&handle.0.0) {
        BackendBuffer(physical_id)
    } else {
        BackendBuffer(handle.0.0)
    }
}
