//! # Command Recording - Pass Context
//!
//! This module implements the [`PassContext`] trait for Vulkan, providing the interface
//! for render passes to declare GPU commands.
//!
//! ## PassContext Pattern
//!
//! The [`VulkanPassContext`] is passed to each render pass during execution.
//! It provides methods for:
//! - Binding pipelines and descriptor sets
//! - Recording draw and dispatch commands
//! - Managing push constants
//! - Handling presentation requests
//!
//! ## Descriptor Set Management
//!
//! The context supports two descriptor set strategies:
//! - **Push descriptors**: For frequently updated sets (no allocation needed)
//! - **Pool allocation**: For static sets (allocated from a per-frame pool)
//!
//! ## Command Buffer Lifecycle
//!
//! ```text
//! prepare_pass() → record_pass() → submit()
//!      ↓               ↓              ↓
//!  Create context   declare commands  Submit to queue
//!  Bind pipeline    Set viewport     Signal timeline
//!  Set barriers     Draw/dispatch    Present
//! ```

use ash::vk;
use ash::vk::Handle;
use i3_gfx::graph::backend::*;
use i3_gfx::graph::pass::RenderPass;
use i3_gfx::graph::pipeline::*;
use i3_gfx::graph::types::*;
use std::sync::Arc;
use std::sync::Mutex;

use crate::backend::VulkanBackend;
use crate::convert::*;
use crate::resource_arena::PhysicalPipeline;

/// Per-thread command pool for parallel command recording.
pub(crate) struct ThreadCommandPool {
    pub(crate) pool: vk::CommandPool,
    pub(crate) descriptor_pool: vk::DescriptorPool,
    pub(crate) allocated: Vec<vk::CommandBuffer>,
    pub(crate) cursor: usize,
}

/// Per-frame context for managing command buffers and descriptor pools.
pub(crate) struct VulkanFrameContext {
    pub(crate) command_pool: vk::CommandPool,
    pub(crate) descriptor_pool: vk::DescriptorPool,
    pub(crate) allocated_command_buffers: Vec<vk::CommandBuffer>,
    pub(crate) cursor: usize,
    pub(crate) submitted_cursor: usize,
    pub(crate) last_completion_value: u64,
    pub(crate) per_thread_pools: Vec<Mutex<ThreadCommandPool>>,
}

/// Domain of a prepared pass (graphics, compute, transfer, or CPU).
pub enum PreparedDomain {
    Graphics {
        color_attachments: [vk::RenderingAttachmentInfo<'static>; 8],
        color_count: usize,
        depth_attachment: Option<vk::RenderingAttachmentInfo<'static>>,
    },
    Compute,
    Transfer,
    Cpu,
}

/// Unified barrier type for images and buffers.
#[derive(Clone)]
pub enum SyncBarrier {
    Image(vk::ImageMemoryBarrier2<'static>),
    Buffer(vk::BufferMemoryBarrier2<'static>),
}

/// Prepared pass ready for recording.
pub struct VulkanPreparedPass {
    pub name: String,
    pub domain: PreparedDomain,
    pub queue: i3_gfx::graph::types::QueueType,
    pub pipeline: Option<i3_gfx::graph::types::PipelineHandle>,
    pub viewport_extent: vk::Extent2D,
    pub sync: crate::sync::PassSyncData,
    pub descriptor_sets: Vec<(u32, Vec<i3_gfx::graph::backend::DescriptorWrite>)>,
}

unsafe impl Send for VulkanPreparedPass {}
unsafe impl Sync for VulkanPreparedPass {}

/// Vulkan implementation of the PassContext trait.
///
/// This context is passed to each render pass during execution and provides
/// methods for binding pipelines, descriptor sets, and recording draw/dispatch commands.
///
/// # Fields
///
/// * `cmd` - Vulkan command buffer for recording
/// * `device` - Reference to the Vulkan device
/// * `present_request` - Optional image handle to present after the pass
/// * `backend` - Raw pointer to the backend (for resource access)
/// * `pipeline` - Currently bound pipeline
/// * `descriptor_pool` - Per-frame descriptor pool for allocation
/// * `current_pipeline_layout` - Layout of the currently bound pipeline
/// * `current_bind_point` - Current pipeline bind point (graphics/compute)
/// * `pending_descriptor_sets` - Descriptor set writes pending flush
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
    ///
    /// # Push Descriptors vs Pool Allocation
    ///
    /// - **Push descriptors**: Used when `pushable_sets_mask` has the bit set for the set index.
    ///   These are pushed directly to the command buffer without allocation.
    /// - **Pool allocation**: Used for static descriptor sets. Allocated from the per-frame
    ///   descriptor pool and updated via `vkUpdateDescriptorSets`.
    ///
    /// # Performance Note
    ///
    /// Push descriptors are more efficient for frequently updated sets (e.g., per-draw data)
    /// because they avoid allocation overhead. Pool allocation is better for static sets
    /// that don't change often.
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

        for (set_index, writes) in sets {
            // Pool Path (now only path)
            let layout = {
                let p = self
                    .backend()
                    .pipeline_resources
                    .get(pipe.physical_id)
                    .expect("Failed to retrieve physical pipeline resource for descriptor layout");
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
            let handle_id = backend
                .descriptor_sets
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(set);
            let set_handle = DescriptorSetHandle(handle_id);

            backend.update_descriptor_set(set_handle, &writes);
            self.bind_descriptor_set(set_index, set_handle);
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

    fn draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: BufferHandle,
        indirect_offset: u64,
        count_buffer: BufferHandle,
        count_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ) {
        let indirect_buf = self.backend().resolve_buffer(indirect_buffer);
        let count_buf = self.backend().resolve_buffer(count_buffer);

        let indirect_vk = self.backend().buffers.get(indirect_buf.0).unwrap().buffer;
        let count_vk = self.backend().buffers.get(count_buf.0).unwrap().buffer;

        unsafe {
            self.device.handle.cmd_draw_indexed_indirect_count(
                self.cmd,
                indirect_vk,
                indirect_offset,
                count_vk,
                count_offset,
                max_draw_count,
                stride,
            );
        }
    }

    fn draw_indirect_count(
        &mut self,
        indirect_buffer: BufferHandle,
        indirect_offset: u64,
        count_buffer: BufferHandle,
        count_offset: u64,
        max_draw_count: u32,
        stride: u32,
    ) {
        let indirect_buf = self.backend().resolve_buffer(indirect_buffer);
        let count_buf = self.backend().resolve_buffer(count_buffer);

        let indirect_vk = self.backend().buffers.get(indirect_buf.0).unwrap().buffer;
        let count_vk = self.backend().buffers.get(count_buf.0).unwrap().buffer;

        unsafe {
            self.device.handle.cmd_draw_indirect_count(
                self.cmd,
                indirect_vk,
                indirect_offset,
                count_vk,
                count_offset,
                max_draw_count,
                stride,
            );
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

    fn build_blas(&mut self, handle: BackendAccelerationStructure, update: bool) {
        crate::accel_struct::build_blas(self, handle, update);
    }

    fn build_tlas(
        &mut self,
        handle: BackendAccelerationStructure,
        instances: &[TlasInstanceDesc],
        update: bool,
    ) {
        crate::accel_struct::build_tlas(self, handle, instances, update);
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

/// Prepare a pass for recording by resolving resources and building barriers.
///
/// This function is called before recording a render pass. It:
/// 1. Resolves virtual resource handles to physical IDs
/// 2. Determines the viewport extent from the first render target
/// 3. Retrieves pre-calculated synchronization data from the Oracle.
/// 4. Prepares attachment info for rendering
///
/// # Viewport Detection
///
/// The viewport extent is determined from the first image write target.
/// If the target is a swapchain image, the swapchain extent is used.
pub fn prepare_pass(
    backend: &mut VulkanBackend,
    pass_index: usize,
    desc: PassDescriptor<'_>,
) -> VulkanPreparedPass {
    // 1. Retrieve Sync Data
    let sync = backend
        .current_plan
        .as_ref()
        .and_then(|plan| plan.pass_sync.get(pass_index).cloned())
        .unwrap_or_default();

    // 2. Identify Target Window & Extent (for Viewport/Pool)
    let mut viewport_extent = vk::Extent2D {
        width: 800,
        height: 600,
    };

    if let Some(&(handle, _)) = desc.image_writes.first() {
        let pid = backend.resolve_image(handle).0;
        if let Some(img) = backend.images.get(pid) {
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
                    if sc_arena_id == pid {
                        viewport_extent = sc.extent;
                        break;
                    }
                }
            }
        }
    }

    // 3. Attachment Setup (Still needed for dynamic rendering)
    let is_compute_pipeline = if let Some(h) = desc.pipeline {
        backend
            .pipeline_resources
            .get(h.0.0)
            .map(|p| p.bind_point == vk::PipelineBindPoint::COMPUTE)
            .unwrap_or(false)
    } else {
        false
    };

    let is_compute =
        matches!(desc.queue, QueueType::AsyncCompute | QueueType::Transfer) || is_compute_pipeline;

    let mut color_attachments = [vk::RenderingAttachmentInfo::default(); 8];
    let mut color_count = 0;
    let mut depth_attachment_info = None;

    if !is_compute {
        let bind_point = vk::PipelineBindPoint::GRAPHICS;

        let mut processed_handles = std::collections::HashSet::new();
        for (handle, usage) in desc.image_writes.iter().chain(desc.image_reads.iter()) {
            if processed_handles.contains(handle) {
                continue;
            }
            if usage.intersects(ResourceUsage::COLOR_ATTACHMENT | ResourceUsage::DEPTH_STENCIL) {
                processed_handles.insert(*handle);
                let pid = backend.resolve_image(*handle).0;
                let img = backend.images.get(pid).expect("Attachment not found");
                let (layout, _, _) = backend.get_image_state(*usage, bind_point);

                let is_write = usage.intersects(
                    ResourceUsage::WRITE
                        | ResourceUsage::COLOR_ATTACHMENT
                        | ResourceUsage::DEPTH_STENCIL,
                );
                let load_op = sync
                    .load_ops
                    .get(&img.image)
                    .cloned()
                    .unwrap_or(vk::AttachmentLoadOp::LOAD);

                let clear_value = if usage.intersects(ResourceUsage::DEPTH_STENCIL) {
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0, // reverse-Z: far = 0
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
                    .image_view(img.view)
                    .image_layout(layout)
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
        queue: desc.queue,
        pipeline: desc.pipeline,
        viewport_extent,
        sync,
        descriptor_sets: desc.descriptor_sets.to_vec(),
    }
}

/// declare barriers for a set of prepared passes.
pub fn record_barriers(
    backend: &VulkanBackend,
    passes: &[&VulkanPreparedPass],
) -> Option<BackendCommandBuffer> {
    if passes.is_empty() {
        return None;
    }

    let mut total_image_barriers = 0;
    let mut total_buffer_barriers = 0;
    for p in passes {
        for b in &p.sync.pre_barriers {
            match b {
                crate::sync::Barrier::Image(_) => total_image_barriers += 1,
                crate::sync::Barrier::Buffer(_) => total_buffer_barriers += 1,
            }
        }
    }

    if total_image_barriers == 0 && total_buffer_barriers == 0 {
        tracing::debug!(
            "record_barriers: No barriers to declare for {} passes",
            passes.len()
        );
        return None;
    }

    tracing::debug!(
        "record_barriers: Recording {} image barriers and {} buffer barriers for {} passes",
        total_image_barriers,
        total_buffer_barriers,
        passes.len()
    );

    let device = backend.get_device().clone();
    let thread_idx = rayon::current_thread_index().unwrap_or(0);

    // Select Queue Context based on Pass Queue Type
    let queue_ctx = match passes[0].queue {
        QueueType::Graphics => backend
            .graphics
            .as_ref()
            .expect("Graphics queue not initialized for recording barriers"),
        QueueType::AsyncCompute => backend
            .compute
            .as_ref()
            .or(backend.graphics.as_ref())
            .expect("No queue context available for compute barriers"),
        QueueType::Transfer => backend
            .transfer
            .as_ref()
            .or(backend.graphics.as_ref())
            .expect("No queue context available for transfer barriers"),
    };

    let frame_ctx = &queue_ctx.frame_contexts[backend.global_frame_index];
    let mut tp = frame_ctx.per_thread_pools[thread_idx % frame_ctx.per_thread_pools.len()]
        .lock()
        .unwrap();

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
        for b in &p.sync.pre_barriers {
            match b {
                crate::sync::Barrier::Image(i) => all_image_barriers.push(i.clone()),
                crate::sync::Barrier::Buffer(b) => all_buffer_barriers.push(b.clone()),
            }
        }
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

/// declare a pass for execution.
///
/// This function records the actual rendering commands for a pass. It:
/// 1. Allocates a command buffer from the thread pool
/// 2. Sets up dynamic viewport and scissor
/// 3. Begins dynamic rendering (if graphics pass)
/// 4. Executes the user's render pass code
/// 5. Ends dynamic rendering
/// 6. Handles presentation transitions
///
/// # Thread Pool
///
/// Command buffers are allocated from a per-frame, per-thread pool to avoid
/// synchronization overhead. The pool is reset at the beginning of each frame.
///
/// # Dynamic Rendering
///
/// The backend uses VK_KHR_dynamic_rendering to avoid render pass objects.
/// This simplifies the code and allows more flexible attachment management.
///
/// # Presentation
///
/// If the pass requests presentation (via `present()`), the image is transitioned
/// to `PRESENT_SRC_KHR` layout at the end of the command buffer.
pub fn record_pass(
    backend: &VulkanBackend,
    prepared: &VulkanPreparedPass,
    pass: &dyn RenderPass,
    frame_data: &i3_gfx::graph::compiler::FrameBlackboard,
) -> (
    Option<u64>,
    Option<BackendCommandBuffer>,
    Option<ImageHandle>,
) {
    let device = backend.get_device().clone();

    let thread_idx = rayon::current_thread_index().unwrap_or(0);
    let queue_ctx = match prepared.queue {
        QueueType::Graphics => backend.graphics.as_ref().unwrap(),
        QueueType::AsyncCompute => backend
            .compute
            .as_ref()
            .unwrap_or_else(|| backend.graphics.as_ref().unwrap()),
        QueueType::Transfer => backend
            .transfer
            .as_ref()
            .unwrap_or_else(|| backend.graphics.as_ref().unwrap()),
    };
    let frame_ctx = &queue_ctx.frame_contexts[backend.global_frame_index];
    let mut tp = frame_ctx.per_thread_pools[thread_idx % frame_ctx.per_thread_pools.len()]
        .lock()
        .unwrap();

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

    let default_bind_point = match prepared.queue {
        QueueType::Graphics => vk::PipelineBindPoint::GRAPHICS,
        QueueType::AsyncCompute | QueueType::Transfer => vk::PipelineBindPoint::COMPUTE,
    };

    let mut ctx = VulkanPassContext {
        cmd,
        device: backend.get_device().clone(),
        present_request: None,
        backend: backend as *const VulkanBackend as *mut VulkanBackend,
        pipeline: None,
        descriptor_pool: frame_ctx.descriptor_pool,
        current_pipeline_layout: vk::PipelineLayout::null(),
        current_bind_point: default_bind_point,
        pending_descriptor_sets: prepared.descriptor_sets.clone(),
    };

    // If pipeline is set, determine bind point and bind it
    if let Some(pipe_handle) = prepared.pipeline {
        ctx.bind_pipeline(pipe_handle);
    }

    let is_compute = matches!(prepared.domain, PreparedDomain::Compute);

    if !is_compute {
        if let PreparedDomain::Graphics {
            color_attachments,
            color_count,
            depth_attachment,
        } = &prepared.domain
        {
            if *color_count > 0 || depth_attachment.is_some() {
                let viewport_extent = prepared.viewport_extent;
                let viewport = vk::Viewport::default()
                    .x(0.0)
                    .y(viewport_extent.height as f32)
                    .width(viewport_extent.width as f32)
                    .height(-(viewport_extent.height as f32))
                    .min_depth(0.0)
                    .max_depth(1.0);
                let scissor = vk::Rect2D::default().extent(viewport_extent);

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
                    device.handle.cmd_set_viewport(cmd, 0, &[viewport]);
                    device.handle.cmd_set_scissor(cmd, 0, &[scissor]);
                }
            }
        }
    }

    pass.execute(&mut ctx, frame_data);

    if !is_compute {
        if let PreparedDomain::Graphics {
            color_count,
            depth_attachment,
            ..
        } = &prepared.domain
        {
            if *color_count > 0 || depth_attachment.is_some() {
                unsafe {
                    device.handle.cmd_end_rendering(cmd);
                }
            }
        }
    }

    // Emit post-barriers (e.g. present transition: final layout → PresentSrc).
    if !prepared.sync.post_barriers.is_empty() {
        let mut img_barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();
        for b in &prepared.sync.post_barriers {
            if let crate::sync::Barrier::Image(b) = b {
                img_barriers.push(b.clone());
            }
        }
        if !img_barriers.is_empty() {
            let dep = vk::DependencyInfo::default().image_memory_barriers(&img_barriers);
            unsafe {
                device.handle.cmd_pipeline_barrier2(cmd, &dep);
            }
        }
    }

    #[cfg(debug_assertions)]
    end_debug_label(backend, BackendCommandBuffer(cmd.as_raw()));

    unsafe {
        device.handle.end_command_buffer(cmd).unwrap();
    }

    (
        Some(backend.graphics.as_ref().unwrap().cpu_timeline),
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
