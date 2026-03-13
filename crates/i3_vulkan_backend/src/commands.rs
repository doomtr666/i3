use ash::vk;
use ash::vk::Handle;
use i3_gfx::graph::backend::*;
use i3_gfx::graph::pass::RenderPass;
use i3_gfx::graph::pipeline::*;
use i3_gfx::graph::types::*;
use std::sync::Arc;

use crate::backend::VulkanBackend;
use crate::convert::*;
use crate::resource_arena::PhysicalPipeline;

/// Vulkan implementation of the PassContext trait.
///
/// This context is passed to each render pass during execution and provides
/// methods for binding pipelines, descriptor sets, and recording draw/dispatch commands.
pub struct VulkanPassContext {
    pub cmd: vk::CommandBuffer,
    pub device: Arc<crate::device::VulkanDevice>,
    pub present_request: Option<ImageHandle>,
    pub backend: *mut VulkanBackend,
    pub pipeline: Option<PhysicalPipeline>,
    pub descriptor_pool: vk::DescriptorPool,
    pub current_pipeline_layout: vk::PipelineLayout,
    pub current_bind_point: vk::PipelineBindPoint,
    pub pending_descriptor_sets: Vec<(u32, Vec<DescriptorWrite>)>,
}

impl VulkanPassContext {
    /// Get an immutable reference to the backend.
    ///
    /// # Safety
    /// The caller must ensure that the backend pointer is valid and that
    /// no mutable aliasing occurs.
    pub fn backend(&self) -> &VulkanBackend {
        unsafe { &*self.backend }
    }

    /// Get a mutable reference to the backend.
    ///
    /// # Safety
    /// The caller must ensure that the backend pointer is valid and that
    /// no other references exist.
    pub fn backend_mut(&mut self) -> &mut VulkanBackend {
        unsafe { &mut *self.backend }
    }

    /// Flush pending descriptor set writes.
    ///
    /// This method processes all pending descriptor set writes and either
    /// pushes them directly (for push descriptor sets) or allocates and
    /// updates descriptor sets from the pool.
    pub fn flush_descriptors(&mut self) {
        if self.pending_descriptor_sets.is_empty() {
            return;
        }

        let pipe = if let Some(p) = &self.pipeline {
            p.clone()
        } else {
            return;
        };

        let sets = std::mem::take(&mut self.pending_descriptor_sets);
        let cmd = self.cmd;

        for (set_index, writes) in sets {
            if (pipe.pushable_sets_mask & (1 << set_index)) != 0 {
                // Push Descriptor Path
                let mut buffer_infos = Vec::with_capacity(writes.len());
                let mut image_infos = Vec::with_capacity(writes.len());

                // Pass 1: Resolve and collect infos
                for write in writes.iter() {
                    match write.descriptor_type {
                        BindingType::UniformBuffer
                        | BindingType::StorageBuffer
                        | BindingType::RawBuffer
                        | BindingType::MutableRawBuffer => {
                            if let Some(info) = &write.buffer_info {
                                let pid = self.backend().resolve_buffer(info.buffer).0;
                                if let Some(buf) = self.backend().buffers.get(pid) {
                                    buffer_infos.push(vk::DescriptorBufferInfo {
                                        buffer: buf.buffer,
                                        offset: info.offset,
                                        range: if info.range == 0 {
                                            vk::WHOLE_SIZE
                                        } else {
                                            info.range
                                        },
                                    });
                                }
                            }
                        }
                        BindingType::CombinedImageSampler
                        | BindingType::Texture
                        | BindingType::StorageTexture
                        | BindingType::Sampler => {
                            if let Some(info) = &write.image_info {
                                let pid = self.backend().resolve_image(info.image).0;
                                if let Some(img) = self.backend().images.get(pid) {
                                    let layout = match info.image_layout {
                                        DescriptorImageLayout::General => vk::ImageLayout::GENERAL,
                                        DescriptorImageLayout::ShaderReadOnlyOptimal => {
                                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                                        }
                                    };
                                    let vk_sampler = if let Some(sampler_handle) = info.sampler {
                                        self.backend()
                                            .samplers
                                            .get(sampler_handle.0)
                                            .cloned()
                                            .unwrap_or(vk::Sampler::null())
                                    } else {
                                        vk::Sampler::null()
                                    };
                                    image_infos.push(vk::DescriptorImageInfo {
                                        sampler: vk_sampler,
                                        image_view: img.view,
                                        image_layout: layout,
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // Pass 2: Build writes
                let mut descriptor_writes = Vec::with_capacity(writes.len());
                let mut buf_ptr = 0;
                let mut img_ptr = 0;

                for write in writes.iter() {
                    let mut vk_write = vk::WriteDescriptorSet::default()
                        .dst_binding(write.binding)
                        .dst_array_element(write.array_element)
                        .descriptor_count(1);

                    match write.descriptor_type {
                        BindingType::UniformBuffer => {
                            if buf_ptr < buffer_infos.len() {
                                vk_write = vk_write
                                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                    .buffer_info(std::slice::from_ref(&buffer_infos[buf_ptr]));
                                buf_ptr += 1;
                                descriptor_writes.push(vk_write);
                            }
                        }
                        BindingType::StorageBuffer
                        | BindingType::RawBuffer
                        | BindingType::MutableRawBuffer => {
                            if buf_ptr < buffer_infos.len() {
                                vk_write = vk_write
                                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                    .buffer_info(std::slice::from_ref(&buffer_infos[buf_ptr]));
                                buf_ptr += 1;
                                descriptor_writes.push(vk_write);
                            }
                        }
                        BindingType::CombinedImageSampler => {
                            if img_ptr < image_infos.len() {
                                vk_write = vk_write
                                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                    .image_info(std::slice::from_ref(&image_infos[img_ptr]));
                                img_ptr += 1;
                                descriptor_writes.push(vk_write);
                            }
                        }
                        BindingType::Texture => {
                            if img_ptr < image_infos.len() {
                                vk_write = vk_write
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(std::slice::from_ref(&image_infos[img_ptr]));
                                img_ptr += 1;
                                descriptor_writes.push(vk_write);
                            }
                        }
                        BindingType::Sampler => {
                            if img_ptr < image_infos.len() {
                                vk_write = vk_write
                                    .descriptor_type(vk::DescriptorType::SAMPLER)
                                    .image_info(std::slice::from_ref(&image_infos[img_ptr]));
                                img_ptr += 1;
                                descriptor_writes.push(vk_write);
                            }
                        }
                        BindingType::StorageTexture => {
                            if img_ptr < image_infos.len() {
                                vk_write = vk_write
                                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                    .image_info(std::slice::from_ref(&image_infos[img_ptr]));
                                img_ptr += 1;
                                descriptor_writes.push(vk_write);
                            }
                        }
                        _ => {}
                    }
                }

                unsafe {
                    self.device.push_descriptor.cmd_push_descriptor_set(
                        cmd,
                        pipe.bind_point,
                        pipe.layout,
                        set_index,
                        &descriptor_writes,
                    );
                }
            } else {
                // Pool Path
                let layout = {
                    let p = self
                        .backend()
                        .pipeline_resources
                        .get(pipe.physical_id)
                        .unwrap();
                    p.set_layouts[set_index as usize]
                };

                let layouts_to_alloc = [layout];
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts_to_alloc);

                let set = unsafe {
                    self.device
                        .handle
                        .allocate_descriptor_sets(&alloc_info)
                        .expect("Failed to allocate per-frame descriptor set")
                }[0];

                let backend = self.backend_mut();
                let handle_id = backend.descriptor_sets.lock().unwrap().insert(set);
                let set_handle = DescriptorSetHandle(handle_id);

                backend.update_descriptor_set(set_handle, &writes);
                self.bind_descriptor_set(set_index, set_handle);
            }
        }
    }
}

impl PassContext for VulkanPassContext {
    fn bind_pipeline(&mut self, pipeline: PipelineHandle) {
        let p = if let Some(p) = self.backend().pipeline_resources.get(pipeline.0.0) {
            p.clone()
        } else {
            return;
        };

        unsafe {
            self.device
                .handle
                .cmd_bind_pipeline(self.cmd, p.bind_point, p.handle);
        }
        self.pipeline = Some(p.clone());
        self.current_pipeline_layout = p.layout;
        self.current_bind_point = p.bind_point;

        self.flush_descriptors();
    }

    fn bind_pipeline_raw(&mut self, pipeline: BackendPipeline) {
        let p = if let Some(p) = self.backend().pipeline_resources.get(pipeline.0) {
            p.clone()
        } else {
            return;
        };

        unsafe {
            self.device
                .handle
                .cmd_bind_pipeline(self.cmd, p.bind_point, p.handle);
        }
        self.pipeline = Some(p.clone());
        self.current_pipeline_layout = p.layout;
        self.current_bind_point = p.bind_point;

        self.flush_descriptors();
    }

    fn bind_vertex_buffer(&mut self, binding: u32, handle: BufferHandle) {
        let physical_id =
            if let Some(&phy) = self.backend().external_buffer_to_physical.get(&handle.0.0) {
                phy
            } else {
                handle.0.0
            };

        if let Some(buf) = self.backend().buffers.get(physical_id) {
            unsafe {
                self.device
                    .handle
                    .cmd_bind_vertex_buffers(self.cmd, binding, &[buf.buffer], &[0]);
            }
        }
    }

    fn bind_index_buffer(&mut self, handle: BufferHandle, index_type: IndexType) {
        let physical_id =
            if let Some(&phy) = self.backend().external_buffer_to_physical.get(&handle.0.0) {
                phy
            } else {
                handle.0.0
            };

        if let Some(buf) = self.backend().buffers.get(physical_id) {
            let vk_type = match index_type {
                IndexType::Uint16 => vk::IndexType::UINT16,
                IndexType::Uint32 => vk::IndexType::UINT32,
            };
            unsafe {
                self.device
                    .handle
                    .cmd_bind_index_buffer(self.cmd, buf.buffer, 0, vk_type);
            }
        }
    }

    fn bind_descriptor_set(&mut self, set_index: u32, handle: DescriptorSetHandle) {
        if let Some(set) = self.backend().descriptor_sets.lock().unwrap().get(handle.0) {
            unsafe {
                self.device.handle.cmd_bind_descriptor_sets(
                    self.cmd,
                    self.current_bind_point,
                    self.current_pipeline_layout,
                    set_index,
                    &[*set],
                    &[],
                );
            }
        }
    }

    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32) {
        // Engine Convention: Vulkan uses Negative Viewport to flip Y-Up → Y-Down.
        // (see engine_conventions.md §2). The caller passes logical (Y-Up) values;
        // the backend transparently applies the flip.
        let viewport = vk::Viewport {
            x,
            y: y + height,
            width,
            height: -height,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            self.device
                .handle
                .cmd_set_viewport(self.cmd, 0, &[viewport]);
        }
    }

    fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D { width, height },
        };
        unsafe {
            self.device.handle.cmd_set_scissor(self.cmd, 0, &[scissor]);
        }
    }

    fn draw(&mut self, vertex_count: u32, first_vertex: u32) {
        unsafe {
            self.device
                .handle
                .cmd_draw(self.cmd, vertex_count, 1, first_vertex, 0);
        }
    }

    fn draw_indexed(&mut self, index_count: u32, first_index: u32, vertex_offset: i32) {
        unsafe {
            self.device.handle.cmd_draw_indexed(
                self.cmd,
                index_count,
                1,
                first_index,
                vertex_offset,
                0,
            );
        }
    }

    fn push_bytes(&mut self, stages: ShaderStageFlags, offset: u32, data: &[u8]) {
        unsafe {
            self.device.handle.cmd_push_constants(
                self.cmd,
                self.current_pipeline_layout,
                convert_shader_stage_flags(stages),
                offset,
                data,
            );
        }
    }

    fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        unsafe {
            let device = self.backend().get_device();
            device.handle.cmd_dispatch(self.cmd, x, y, z);
        }
    }

    fn clear_buffer(&mut self, buffer: BufferHandle, clear_value: u32) {
        let physical_id =
            if let Some(&phy) = self.backend().external_buffer_to_physical.get(&buffer.0.0) {
                phy
            } else {
                buffer.0.0
            };

        if let Some(buf) = self.backend().buffers.get(physical_id) {
            unsafe {
                let device = self.backend().get_device();
                device.handle.cmd_fill_buffer(
                    self.cmd,
                    buf.buffer,
                    0,
                    ash::vk::WHOLE_SIZE,
                    clear_value,
                );
            }

            // Update state to reflect TRANSFER_WRITE
            if let Some(buf) = self.backend_mut().buffers.get_mut(physical_id) {
                buf.last_access = vk::AccessFlags2::TRANSFER_WRITE;
                buf.last_stage = vk::PipelineStageFlags2::TRANSFER;
            }
        }
    }

    fn present(&mut self, image: ImageHandle) {
        let physical_id = if let Some(&phy) = self.backend().external_to_physical.get(&image.0.0) {
            phy
        } else {
            image.0.0
        };
        self.present_request = Some(ImageHandle(SymbolId(physical_id)));
    }

    fn copy_buffer(
        &mut self,
        src: BufferHandle,
        dst: BufferHandle,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) {
        let src_buf = self.backend().resolve_buffer(src);
        let dst_buf = self.backend().resolve_buffer(dst);

        let src_vk = self.backend().buffers.get(src_buf.0).unwrap().buffer;
        let dst_vk = self.backend().buffers.get(dst_buf.0).unwrap().buffer;

        let region = vk::BufferCopy::default()
            .src_offset(src_offset)
            .dst_offset(dst_offset)
            .size(size);

        unsafe {
            self.device
                .handle
                .cmd_copy_buffer(self.cmd, src_vk, dst_vk, &[region]);
        }

        // Update destination state to reflect TRANSFER_WRITE
        if let Some(buf) = self.backend_mut().buffers.get_mut(dst_buf.0) {
            buf.last_access = vk::AccessFlags2::TRANSFER_WRITE;
            buf.last_stage = vk::PipelineStageFlags2::TRANSFER;
        }
    }

    fn map_buffer(&mut self, handle: BufferHandle) -> *mut u8 {
        let device = self.device.clone();
        let buf_id = self.backend_mut().resolve_buffer(handle).0;
        let physical = self.backend_mut().buffers.get_mut(buf_id).unwrap();

        if let Some(alloc) = &mut physical.allocation {
            let allocator = device.allocator.lock().unwrap();
            unsafe { allocator.map_memory(alloc).unwrap() }
        } else {
            std::ptr::null_mut()
        }
    }

    fn create_descriptor_set(
        &mut self,
        pipeline: BackendPipeline,
        set_index: u32,
        writes: &[DescriptorWrite],
    ) -> DescriptorSetHandle {
        let pipeline_id = pipeline.0;
        let layout = {
            let p = self
                .backend()
                .pipeline_resources
                .get(pipeline_id)
                .expect("Pipeline not found");
            p.set_layouts[set_index as usize]
        };

        let layouts_to_alloc = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts_to_alloc);

        let set = unsafe {
            self.device
                .handle
                .allocate_descriptor_sets(&alloc_info)
                .expect("Failed to allocate per-frame descriptor set")
        }[0];

        let backend = self.backend_mut();
        let handle_id = backend.descriptor_sets.lock().unwrap().insert(set);
        let set_handle = DescriptorSetHandle(handle_id);

        backend.update_descriptor_set(set_handle, writes);
        set_handle
    }

    fn bind_descriptor_set_raw(&mut self, set_index: u32, handle: u64) {
        self.bind_descriptor_set(set_index, DescriptorSetHandle(handle));
    }

    fn unmap_buffer(&mut self, handle: BufferHandle) {
        let device = self.device.clone();
        let buf_id = self.backend_mut().resolve_buffer(handle).0;
        let physical = self.backend_mut().buffers.get_mut(buf_id).unwrap();

        if let Some(alloc) = &mut physical.allocation {
            let allocator = device.allocator.lock().unwrap();
            unsafe {
                let _ = allocator.flush_allocation(alloc, 0, vk::WHOLE_SIZE);
                allocator.unmap_memory(alloc);
            }
        }
    }
}

use std::collections::HashMap;

use crate::backend::{PreparedDomain, VulkanPreparedPass};

/// Prepare a pass for recording by resolving resources and building barriers.
pub fn prepare_pass(backend: &mut VulkanBackend, desc: PassDescriptor<'_>) -> VulkanPreparedPass {
    // Clear scratch vectors for this pass
    backend.image_barrier_scratch.clear();
    backend.buffer_barrier_scratch.clear();

    // Resolve target physical IDs from writes (Using scratch)
    backend.target_id_scratch.clear();
    for (handle, _) in desc.image_writes {
        let pid = if let Some(&p) = backend.external_to_physical.get(&handle.0.0) {
            p
        } else {
            handle.0.0
        };
        backend.target_id_scratch.push(pid);
    }

    // Identify Target Window & Extent (for Viewport/Pool)
    let mut viewport_extent = vk::Extent2D {
        width: 800,
        height: 600,
    }; // Fallback

    if let Some(&first_pid) = backend.target_id_scratch.first() {
        if let Some(img) = backend.images.get(first_pid) {
            viewport_extent = vk::Extent2D {
                width: img.desc.width,
                height: img.desc.height,
            };
        }
        // Fast window lookup (Match Arena ID)
        for ctx_win in backend.windows.values() {
            if let (Some(sc), Some(idx)) = (&ctx_win.swapchain, ctx_win.current_image_index) {
                let sc_handle = sc.images[idx as usize].as_raw();
                if let Some(&sc_arena_id) = backend.external_to_physical.get(&sc_handle) {
                    if sc_arena_id == first_pid {
                        viewport_extent = sc.extent;
                        break;
                    }
                }
            }
        }
    }

    // Infer domain from pipeline bind point (no user-declared domain)
    let is_compute = if let Some(h) = desc.pipeline {
        backend
            .pipeline_resources
            .get(h.0.0)
            .map(|p| p.bind_point == vk::PipelineBindPoint::COMPUTE)
            .unwrap_or(false)
    } else {
        false
    };

    let current_bind_point = if is_compute {
        vk::PipelineBindPoint::COMPUTE
    } else {
        vk::PipelineBindPoint::GRAPHICS
    };

    // Prepare attachments
    // --- Unified Resource Synchronization & Attachment Discovery ---
    let mut color_attachments = [vk::RenderingAttachmentInfo::default(); 8];
    let mut color_count = 0;
    let mut depth_attachment_info = None;

    // Dedup and merge usages for all images while preserving order
    let mut pass_images_order = Vec::new();
    let mut pass_images_map: HashMap<ImageHandle, (ResourceUsage, bool)> = HashMap::new();

    for (handle, usage) in desc.image_writes {
        if !pass_images_map.contains_key(handle) {
            pass_images_order.push(*handle);
        }
        pass_images_map.insert(*handle, (*usage, true));
    }
    for (handle, usage) in desc.image_reads {
        if !pass_images_map.contains_key(handle) {
            pass_images_order.push(*handle);
            pass_images_map.insert(*handle, (*usage, false));
        } else {
            let entry = pass_images_map.get_mut(handle).unwrap();
            entry.0 |= *usage;
        }
    }

    // Synchronize and collect attachments in deterministic order
    for handle in pass_images_order {
        let (usage, is_write) = pass_images_map[&handle];
        let pid = backend.resolve_image(handle).0;
        let (target_layout, target_access, target_stage) =
            backend.get_image_state(usage, is_write, current_bind_point);

        if let Some(barrier) =
            backend.get_image_barrier(pid, target_layout, target_access, target_stage)
        {
            backend.image_barrier_scratch.push(barrier);
        }

        if usage.intersects(ResourceUsage::COLOR_ATTACHMENT | ResourceUsage::DEPTH_STENCIL) {
            let img_info = if let Some(img) = backend.images.get(pid) {
                (img.format, img.view)
            } else {
                continue;
            };

            let load_op = if is_write {
                if let Some(img) = backend.images.get_mut(pid) {
                    if img.last_write_frame < backend.frame_count {
                        img.last_write_frame = backend.frame_count;
                        vk::AttachmentLoadOp::CLEAR
                    } else {
                        vk::AttachmentLoadOp::LOAD
                    }
                } else {
                    vk::AttachmentLoadOp::LOAD
                }
            } else {
                vk::AttachmentLoadOp::LOAD
            };

            let clear_value = if usage.intersects(ResourceUsage::DEPTH_STENCIL) {
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                }
            } else {
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }
            };

            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(img_info.1)
                .image_layout(target_layout)
                .load_op(load_op)
                .store_op(if is_write {
                    vk::AttachmentStoreOp::STORE
                } else {
                    vk::AttachmentStoreOp::NONE
                })
                .clear_value(clear_value);

            if usage.intersects(ResourceUsage::DEPTH_STENCIL) {
                depth_attachment_info = Some(attachment);
            } else if color_count < 8 {
                color_attachments[color_count] = attachment;
                color_count += 1;
            }
        }
    }

    // Deduplicate and synchronize buffers while preserving order
    let mut pass_buffers_order = Vec::new();
    let mut pass_buffers_map: HashMap<BufferHandle, ResourceUsage> = HashMap::new();
    for (handle, usage) in desc.buffer_writes {
        if !pass_buffers_map.contains_key(handle) {
            pass_buffers_order.push(*handle);
        }
        pass_buffers_map.insert(*handle, *usage);
    }
    for (handle, usage) in desc.buffer_reads {
        if !pass_buffers_map.contains_key(handle) {
            pass_buffers_order.push(*handle);
            pass_buffers_map.insert(*handle, *usage);
        } else {
            let entry = pass_buffers_map.get_mut(handle).unwrap();
            *entry |= *usage;
        }
    }

    for handle in pass_buffers_order {
        let usage = pass_buffers_map[&handle];
        let pid = backend.resolve_buffer(handle).0;
        let (target_access, target_stage) = backend.get_buffer_state(usage, current_bind_point);
        if let Some(barrier) = backend.get_buffer_barrier(pid, target_access, target_stage) {
            backend.buffer_barrier_scratch.push(barrier);
        }
    }

    let domain = if is_compute {
        PreparedDomain::Compute
    } else {
        PreparedDomain::Graphics {
            color_attachments,
            color_count,
            depth_attachment: depth_attachment_info,
        }
    };

    VulkanPreparedPass {
        name: desc.name.to_string(),
        domain,
        pipeline: desc.pipeline,
        viewport_extent,
        image_barriers: backend.image_barrier_scratch.clone(),
        buffer_barriers: backend.buffer_barrier_scratch.clone(),
        descriptor_sets: desc.descriptor_sets.to_vec(),
    }
}

/// Record barriers for a set of prepared passes.
pub fn record_barriers(
    backend: &VulkanBackend,
    passes: &[&VulkanPreparedPass],
) -> Option<BackendCommandBuffer> {
    let mut total_image_barriers = 0;
    let mut total_buffer_barriers = 0;
    for p in passes {
        total_image_barriers += p.image_barriers.len();
        total_buffer_barriers += p.buffer_barriers.len();
    }

    if total_image_barriers == 0 && total_buffer_barriers == 0 {
        return None;
    }

    let device = backend.get_device().clone();
    let thread_idx = 0;
    let frame_ctx = &backend.frame_contexts[backend.global_frame_index];
    let mut tp = frame_ctx.per_thread_pools[thread_idx].lock().unwrap();

    // Allocate Command Buffer from Thread Pool
    let cmd = if tp.cursor < tp.allocated.len() {
        let cmd = tp.allocated[tp.cursor];
        tp.cursor += 1;
        cmd
    } else {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(tp.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.handle.allocate_command_buffers(&alloc_info).unwrap()[0] };
        tp.allocated.push(cmd);
        tp.cursor += 1;
        cmd
    };

    // Begin Recording
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .handle
            .begin_command_buffer(cmd, &begin_info)
            .unwrap();
    }

    let mut all_image_barriers = Vec::with_capacity(total_image_barriers);
    let mut all_buffer_barriers = Vec::with_capacity(total_buffer_barriers);
    for p in passes {
        all_image_barriers.extend_from_slice(&p.image_barriers);
        all_buffer_barriers.extend_from_slice(&p.buffer_barriers);
    }

    let dependency_info = vk::DependencyInfo::default()
        .image_memory_barriers(&all_image_barriers)
        .buffer_memory_barriers(&all_buffer_barriers);

    unsafe {
        device.handle.cmd_pipeline_barrier2(cmd, &dependency_info);
        device.handle.end_command_buffer(cmd).unwrap();
    }

    Some(BackendCommandBuffer(unsafe {
        std::mem::transmute::<vk::CommandBuffer, u64>(cmd)
    }))
}

/// Begin a debug label (debug builds only).
#[cfg(debug_assertions)]
pub fn begin_debug_label(
    backend: &VulkanBackend,
    command_buffer: BackendCommandBuffer,
    name: &str,
    color: [f32; 4],
) {
    let c_name = std::ffi::CString::new(name).unwrap();
    let label = vk::DebugUtilsLabelEXT::default()
        .label_name(&c_name)
        .color(color);
    unsafe {
        let cb = vk::CommandBuffer::from_raw(command_buffer.0);
        backend
            .get_device()
            .debug_utils
            .cmd_begin_debug_utils_label(cb, &label);
    }
}

/// End a debug label (debug builds only).
#[cfg(debug_assertions)]
pub fn end_debug_label(backend: &VulkanBackend, command_buffer: BackendCommandBuffer) {
    unsafe {
        let cb = vk::CommandBuffer::from_raw(command_buffer.0);
        backend
            .get_device()
            .debug_utils
            .cmd_end_debug_utils_label(cb);
    }
}

/// Record a pass for execution.
pub fn record_pass(
    backend: &VulkanBackend,
    prepared: &VulkanPreparedPass,
    pass: &dyn RenderPass,
) -> (
    Option<u64>,
    Option<BackendCommandBuffer>,
    Option<ImageHandle>,
) {
    let device = backend.get_device().clone();

    let thread_idx = 0;
    let frame_ctx = &backend.frame_contexts[backend.global_frame_index];
    let mut tp = frame_ctx.per_thread_pools[thread_idx].lock().unwrap();

    // Allocate Command Buffer from Thread Pool
    let cmd = if tp.cursor < tp.allocated.len() {
        let cmd = tp.allocated[tp.cursor];
        tp.cursor += 1;
        cmd
    } else {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(tp.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.handle.allocate_command_buffers(&alloc_info).unwrap()[0] };
        tp.allocated.push(cmd);
        tp.cursor += 1;
        cmd
    };

    // Begin Recording
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .handle
            .begin_command_buffer(cmd, &begin_info)
            .unwrap()
    };

    #[cfg(debug_assertions)]
    begin_debug_label(
        backend,
        BackendCommandBuffer(cmd.as_raw()),
        &prepared.name,
        [1.0, 1.0, 1.0, 1.0],
    );

    let mut ctx = VulkanPassContext {
        cmd,
        device: backend.get_device().clone(),
        present_request: None,
        backend: backend as *const VulkanBackend as *mut VulkanBackend,
        pipeline: None,
        descriptor_pool: frame_ctx.descriptor_pool,
        current_pipeline_layout: vk::PipelineLayout::null(),
        current_bind_point: vk::PipelineBindPoint::GRAPHICS,
        pending_descriptor_sets: prepared.descriptor_sets.clone(),
    };

    // If pipeline is set, determine bind point and bind it
    if let Some(pipe_handle) = prepared.pipeline {
        ctx.bind_pipeline(pipe_handle);
    }

    // (Barriers were already emitted globally via submit_barriers before the pass recording started)

    let is_compute = matches!(prepared.domain, PreparedDomain::Compute);

    if !is_compute {
        // Dynamic Viewport/Scissor setup (Use resolved extent)
        let viewport_extent = prepared.viewport_extent;
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(viewport_extent.height as f32)
            .width(viewport_extent.width as f32)
            .height(-(viewport_extent.height as f32))
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(viewport_extent);

        unsafe {
            device.handle.cmd_set_viewport(cmd, 0, &[viewport]);
            device.handle.cmd_set_scissor(cmd, 0, &[scissor]);
        }

        if let PreparedDomain::Graphics {
            color_attachments,
            color_count,
            depth_attachment,
        } = &prepared.domain
        {
            if *color_count > 0 || depth_attachment.is_some() {
                let rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: viewport_extent,
                    })
                    .layer_count(1)
                    .color_attachments(&color_attachments[..*color_count]);

                let rendering_info = if let Some(depth) = depth_attachment {
                    rendering_info.depth_attachment(depth)
                } else {
                    rendering_info
                };

                unsafe {
                    device.handle.cmd_begin_rendering(cmd, &rendering_info);
                }
            }
        }
    }

    pass.execute(&mut ctx);

    if !is_compute {
        if let PreparedDomain::Graphics {
            color_attachments: _,
            color_count,
            depth_attachment,
        } = &prepared.domain
        {
            if *color_count > 0 || depth_attachment.is_some() {
                unsafe {
                    device.handle.cmd_end_rendering(cmd);
                }
            }
        }
    }

    // Handle explicit transition for Present if requested
    if let Some(handle) = ctx.present_request {
        let pid = backend.resolve_image(handle).0;
        if let Some(img) = backend.images.get(pid) {
            let aspect_mask = if img.format == vk::Format::D32_SFLOAT {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };

            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(img.last_stage)
                .src_access_mask(img.last_access)
                .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .dst_access_mask(vk::AccessFlags2::empty())
                .old_layout(img.last_layout)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(img.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let barriers = [barrier];
            let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            unsafe {
                device.handle.cmd_pipeline_barrier2(cmd, &dependency_info);
            }
        }
    }

    #[cfg(debug_assertions)]
    end_debug_label(backend, BackendCommandBuffer(cmd.as_raw()));

    unsafe {
        device.handle.end_command_buffer(cmd).unwrap();
    }

    (
        Some(backend.cpu_timeline),
        Some(BackendCommandBuffer(unsafe {
            std::mem::transmute::<vk::CommandBuffer, u64>(cmd)
        })),
        ctx.present_request,
    )
}

/// Mark an image as presented (transition to PRESENT_SRC_KHR layout).
pub fn mark_image_as_presented(backend: &mut VulkanBackend, handle: ImageHandle) {
    let pid = backend.resolve_image(handle).0;
    if let Some(img) = backend.images.get_mut(pid) {
        img.last_layout = vk::ImageLayout::PRESENT_SRC_KHR;
        img.last_access = vk::AccessFlags2::empty();
        img.last_stage = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;
    }
}
