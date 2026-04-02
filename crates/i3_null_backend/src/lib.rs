use i3_gfx::graph::backend::{
    BackendAccelerationStructure, BackendBuffer, BackendImage, DescriptorSetHandle,
    DescriptorWrite, PassContext, PassDescriptor, RenderBackend, RenderBackendInternal,
    SamplerHandle, TlasInstanceDesc,
};
use i3_gfx::graph::pass::RenderPass;
use i3_gfx::graph::pipeline::IndexType;
use i3_gfx::graph::types::{BufferDesc, BufferHandle, ImageDesc, ImageHandle, WindowHandle};
use i3_io;
pub mod prelude;
use std::collections::HashSet;
use thiserror::Error;
use tracing::{error, info};

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Resource not found: {0:?}")]
    ResourceNotFound(u64),
    #[error("Access conflict on resource {0:?}: {1}")]
    AccessConflict(u64, String),
}

pub struct NullBackend {
    allocated_images: HashSet<u64>,
    allocated_buffers: HashSet<u64>,
    allocated_pipelines: HashSet<u64>,
    image_map: std::collections::HashMap<ImageHandle, BackendImage>,
    next_handle: u64,
}

impl NullBackend {
    pub fn new() -> Self {
        Self {
            allocated_images: HashSet::new(),
            allocated_buffers: HashSet::new(),
            allocated_pipelines: HashSet::new(),
            image_map: std::collections::HashMap::new(),
            next_handle: 1, // 0 is INVALID
        }
    }

    fn next_handle(&mut self) -> u64 {
        let h = self.next_handle;
        self.next_handle += 1;
        h
    }
}

impl RenderBackend for NullBackend {
    fn capabilities(&self) -> i3_gfx::graph::backend::DeviceCapabilities {
        i3_gfx::graph::backend::DeviceCapabilities {
            ray_tracing: false,
            async_compute: false,
            async_transfer: false,
        }
    }

    fn create_image(&mut self, desc: &ImageDesc) -> BackendImage {
        let h = self.next_handle();
        self.allocated_images.insert(h);
        info!(handle = h, ?desc, "Created Image");
        BackendImage(h)
    }

    fn create_buffer(&mut self, _desc: &BufferDesc) -> BackendBuffer {
        let handle = self.next_handle();
        self.allocated_buffers.insert(handle);
        BackendBuffer(handle)
    }

    fn destroy_image(&mut self, handle: BackendImage) {
        self.allocated_images.remove(&handle.0);
        info!(handle = %handle.0, "Destroyed Image");
    }

    fn destroy_buffer(&mut self, _handle: BackendBuffer) {
        // No-op for null backend
    }

    fn create_sampler(&mut self, desc: &i3_gfx::graph::types::SamplerDesc) -> SamplerHandle {
        let handle = self.next_handle();
        info!(handle, ?desc, "Created Sampler");
        SamplerHandle(handle)
    }

    fn destroy_sampler(&mut self, handle: SamplerHandle) {
        info!(handle = handle.0, "Destroyed Sampler");
    }

    fn create_graphics_pipeline(
        &mut self,
        _desc: &i3_gfx::graph::pipeline::GraphicsPipelineCreateInfo,
    ) -> i3_gfx::graph::backend::BackendPipeline {
        let h = self.next_handle();
        self.allocated_pipelines.insert(h);
        info!(handle = h, "Created Graphics Pipeline");
        i3_gfx::graph::backend::BackendPipeline(h)
    }

    fn create_compute_pipeline(
        &mut self,
        _desc: &i3_gfx::graph::pipeline::ComputePipelineCreateInfo,
    ) -> i3_gfx::graph::backend::BackendPipeline {
        let h = self.next_handle();
        self.allocated_pipelines.insert(h);
        info!(handle = h, "Created Compute Pipeline");
        i3_gfx::graph::backend::BackendPipeline(h)
    }

    fn enumerate_devices(&self) -> Vec<i3_gfx::graph::backend::DeviceInfo> {
        vec![i3_gfx::graph::backend::DeviceInfo {
            id: 0,
            name: "Null GPU".to_string(),
            device_type: i3_gfx::graph::backend::DeviceType::Virtual,
        }]
    }

    fn initialize(&mut self, _device_id: u32) -> Result<(), String> {
        info!("Initialized Null Backend");
        Ok(())
    }

    fn create_blas(
        &mut self,
        _info: &i3_gfx::graph::backend::BlasCreateInfo,
    ) -> BackendAccelerationStructure {
        let handle = self.next_handle();
        info!(handle, "Created Null BLAS");
        BackendAccelerationStructure(handle)
    }

    fn destroy_blas(&mut self, handle: BackendAccelerationStructure) {
        info!(handle = handle.0, "Destroyed Null BLAS");
    }

    fn create_tlas(
        &mut self,
        _info: &i3_gfx::graph::backend::TlasCreateInfo,
    ) -> BackendAccelerationStructure {
        let handle = self.next_handle();
        info!(handle, "Created Null TLAS");
        BackendAccelerationStructure(handle)
    }

    fn destroy_tlas(&mut self, handle: BackendAccelerationStructure) {
        info!(handle = handle.0, "Destroyed Null TLAS");
    }

    fn create_window(
        &mut self,
        desc: i3_gfx::graph::backend::WindowDesc,
    ) -> Result<WindowHandle, String> {
        info!(?desc, "Created Null Window");
        Ok(WindowHandle(1))
    }

    fn destroy_window(&mut self, window: WindowHandle) {
        info!(?window, "Destroyed Null Window");
    }

    fn configure_window(
        &mut self,
        window: WindowHandle,
        config: i3_gfx::graph::backend::SwapchainConfig,
    ) -> Result<(), String> {
        info!(?window, ?config, "Configured Null Window");
        Ok(())
    }

    fn set_fullscreen(&mut self, window: WindowHandle, fullscreen: bool) {
        info!(?window, fullscreen, "Set Fullscreen Null Window");
    }

    fn poll_events(&mut self) -> Vec<i3_gfx::graph::backend::Event> {
        Vec::new()
    }

    fn upload_buffer(
        &mut self,
        _handle: BackendBuffer,
        _data: &[u8],
        _offset: u64,
    ) -> Result<(), String> {
        Ok(())
    }

    fn upload_image(
        &mut self,
        _handle: BackendImage,
        _data: &[u8],
        _offset_x: u32,
        _offset_y: u32,
        _data_width: u32,
        _data_height: u32,
        _mip_level: u32,
        _array_layer: u32,
    ) -> Result<(), String> {
        Ok(())
    }

    fn get_bindless_set_handle(&self) -> u64 {
        0
    }

    // --- Resource Resolution ---
    fn resolve_image(&self, handle: ImageHandle) -> BackendImage {
        if let Some(phy) = self.image_map.get(&handle) {
            *phy
        } else {
            BackendImage(handle.0.0)
        }
    }

    fn resolve_buffer(&self, handle: BufferHandle) -> BackendBuffer {
        BackendBuffer(handle.0.0)
    }

    fn resolve_pipeline(
        &self,
        handle: i3_gfx::graph::types::PipelineHandle,
    ) -> i3_gfx::graph::backend::BackendPipeline {
        i3_gfx::graph::backend::BackendPipeline(handle.0.0)
    }

    // --- Descriptor Management ---
    fn create_graphics_pipeline_from_baked(
        &mut self,
        _desc: &i3_io::pipeline_asset::BakeableGraphicsPipeline,
        _reflection: &[u8],
        _bytecode: &[u8],
    ) -> i3_gfx::graph::backend::BackendPipeline {
        let h = self.next_handle();
        self.allocated_pipelines.insert(h);
        i3_gfx::graph::backend::BackendPipeline(h)
    }

    fn create_compute_pipeline_from_baked(
        &mut self,
        _reflection: &[u8],
        _bytecode: &[u8],
    ) -> i3_gfx::graph::backend::BackendPipeline {
        let h = self.next_handle();
        self.allocated_pipelines.insert(h);
        i3_gfx::graph::backend::BackendPipeline(h)
    }

    fn update_bindless_texture(
        &mut self,
        _texture: ImageHandle,
        _sampler: SamplerHandle,
        index: u32,
        _set: u64,
        _binding: u32,
    ) {
        info!("Updated Bindless Texture index {}", index);
    }

    fn update_bindless_texture_raw(
        &mut self,
        _texture: BackendImage,
        _sampler: SamplerHandle,
        index: u32,
        _set: u64,
        _binding: u32,
    ) {
        info!("Updated Bindless Texture (raw) index {}", index);
    }

    fn update_bindless_sampler(&mut self, _sampler: SamplerHandle, _set: u64, _binding: u32) {
        info!("Updated Bindless Sampler");
    }

    // --- Handle Registration ---
    fn register_external_image(&mut self, handle: ImageHandle, physical: BackendImage) {
        info!(
            ?handle,
            ?physical,
            "Registered external image in NullBackend"
        );
        self.image_map.insert(handle, physical);
    }

    fn register_external_buffer(&mut self, _handle: BufferHandle, _physical: BackendBuffer) {
        info!("Registered external buffer in NullBackend");
    }

    fn wait_for_timeline(&self, value: u64, timeout_ns: u64) -> Result<(), String> {
        info!(value, timeout_ns, "Waiting for timeline (NullBackend)");
        Ok(())
    }

    // --- Transient Resource Management (Pooling) ---
    fn create_transient_image(&mut self, desc: &ImageDesc) -> BackendImage {
        self.create_image(desc)
    }

    fn create_transient_buffer(&mut self, desc: &BufferDesc) -> BackendBuffer {
        self.create_buffer(desc)
    }

    fn release_transient_image(&mut self, handle: BackendImage) {
        self.destroy_image(handle);
    }

    fn release_transient_buffer(&mut self, handle: BackendBuffer) {
        self.destroy_buffer(handle);
    }

    fn garbage_collect(&mut self) {
        // No-op
    }

    fn get_buffer_address(&self, _handle: BackendBuffer) -> u64 {
        0
    }
}

unsafe impl Send for NullBackend {}
unsafe impl Sync for NullBackend {}

impl RenderBackendInternal for NullBackend {
    fn begin_frame(&mut self) {}
    fn end_frame(&mut self) {}
    fn reset_frame_resources(&mut self) {}

    fn analyze_frame(&mut self, _passes: &[i3_gfx::graph::types::FlatPass]) -> i3_gfx::graph::sync::SyncPlan {
        i3_gfx::graph::sync::SyncPlan::default()
    }

    fn acquire_swapchain_image(
        &mut self,
        _window: WindowHandle,
    ) -> Result<Option<(BackendImage, u64, u32)>, String> {
        let handle = 1000;
        self.allocated_images.insert(handle);
        Ok(Some((BackendImage(handle), 1, 0)))
    }

    fn submit(&mut self, _batch: i3_gfx::graph::backend::CommandBatch) -> Result<u64, String> {
        Ok(0)
    }

    type PreparedPass = NullPreparedPass;

    fn prepare_pass(&mut self, _pass_index: usize, desc: PassDescriptor<'_>) -> Self::PreparedPass {
        info!(name = %desc.name, "Preparing null pass");
        NullPreparedPass {
            name: desc.name.to_string(),
            queue: desc.queue,
        }
    }

    fn get_prepared_pass_queue(
        &self,
        prepared: &Self::PreparedPass,
    ) -> i3_gfx::graph::types::QueueType {
        prepared.queue
    }

    fn record_barriers(
        &mut self,
        _passes: &[&Self::PreparedPass],
    ) -> Option<i3_gfx::graph::backend::BackendCommandBuffer> {
        // Null backend does not need to submit barriers
        None
    }

    fn record_pass(
        &self,
        prepared: &Self::PreparedPass,
        pass: &dyn RenderPass,
    ) -> (
        Option<u64>,
        Option<i3_gfx::graph::backend::BackendCommandBuffer>,
        Option<ImageHandle>,
    ) {
        info!(name = %prepared.name, "Recording null pass");
        let mut ctx = NullPassContext::new(
            &prepared.name,
            &self.allocated_images,
            &self.allocated_buffers,
            &self.allocated_pipelines,
            &self.image_map,
            self.next_handle, // Pass this to allow context to allocate handles if needed
        );
        pass.execute(&mut ctx);
        (Some(0), None, None)
    }

    fn allocate_descriptor_set(
        &mut self,
        _pipeline: i3_gfx::graph::types::PipelineHandle,
        set_index: u32,
    ) -> Result<DescriptorSetHandle, String> {
        let h = self.next_handle();
        info!(set_index, handle = h, "Allocated Descriptor Set");
        Ok(DescriptorSetHandle(h))
    }

    fn update_descriptor_set(&mut self, set: DescriptorSetHandle, writes: &[DescriptorWrite]) {
        info!(?set, writes = writes.len(), "Updated Descriptor Set");
    }
}

pub struct NullPassContext<'a> {
    pub pass_name: String,
    validation_failures: Vec<ValidationError>,
    allocated_images: &'a HashSet<u64>,
    allocated_buffers: &'a HashSet<u64>,
    allocated_pipelines: &'a HashSet<u64>,
    image_map: &'a std::collections::HashMap<ImageHandle, BackendImage>,
    next_handle: u64,
}

pub struct NullPreparedPass {
    pub name: String,
    pub queue: i3_gfx::graph::types::QueueType,
}

impl<'a> NullPassContext<'a> {
    pub fn new(
        name: &str,
        allocated_images: &'a HashSet<u64>,
        allocated_buffers: &'a HashSet<u64>,
        allocated_pipelines: &'a HashSet<u64>,
        image_map: &'a std::collections::HashMap<ImageHandle, BackendImage>,
        next_handle: u64,
    ) -> Self {
        Self {
            pass_name: name.to_string(),
            validation_failures: Vec::new(),
            allocated_images,
            allocated_buffers,
            allocated_pipelines,
            image_map,
            next_handle,
        }
    }

    pub fn report_error(&mut self, err: ValidationError) {
        error!(pass = %self.pass_name, error = %err, "Validation Failure");
        self.validation_failures.push(err);
    }

    pub fn failures(&self) -> &[ValidationError] {
        &self.validation_failures
    }

    fn next_handle(&mut self) -> u64 {
        let h = self.next_handle;
        self.next_handle += 1;
        h
    }
}

impl<'a> PassContext for NullPassContext<'a> {
    fn bind_pipeline(&mut self, pipeline: i3_gfx::graph::types::PipelineHandle) {
        info!(pass = %self.pass_name, ?pipeline, "BIND_PIPELINE");
        if !self.allocated_pipelines.contains(&pipeline.0.0) {
            self.report_error(ValidationError::ResourceNotFound(pipeline.0.0));
        }
    }

    fn bind_pipeline_raw(&mut self, pipeline: i3_gfx::graph::backend::BackendPipeline) {
        info!(pass = %self.pass_name, ?pipeline, "BIND_PIPELINE_RAW");
        if !self.allocated_pipelines.contains(&pipeline.0) {
            self.report_error(ValidationError::ResourceNotFound(pipeline.0));
        }
    }

    fn bind_vertex_buffer(&mut self, binding: u32, handle: BufferHandle) {
        info!(pass = %self.pass_name, binding, ?handle, "BIND_VERTEX_BUFFER");
        if !self.allocated_buffers.contains(&handle.0.0) {
            self.report_error(ValidationError::ResourceNotFound(handle.0.0));
        }
    }

    fn bind_index_buffer(&mut self, handle: BufferHandle, index_type: IndexType) {
        info!(pass = %self.pass_name, ?handle, ?index_type, "BIND_INDEX_BUFFER");
        if !self.allocated_buffers.contains(&handle.0.0) {
            self.report_error(ValidationError::ResourceNotFound(handle.0.0));
        }
    }

    fn bind_descriptor_set(&mut self, set_index: u32, handle: DescriptorSetHandle) {
        info!(pass = %self.pass_name, set_index, ?handle, "BIND_DESCRIPTOR_SET");
    }

    fn create_descriptor_set(
        &mut self,
        _pipeline: i3_gfx::graph::backend::BackendPipeline,
        _set_index: u32,
        _writes: &[DescriptorWrite],
    ) -> DescriptorSetHandle {
        let h = self.next_handle();
        DescriptorSetHandle(h)
    }

    fn bind_descriptor_set_raw(&mut self, _set_index: u32, _handle: u64) {}

    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32) {
        info!(pass = %self.pass_name, x, y, width, height, "SET_VIEWPORT");
    }

    fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        info!(pass = %self.pass_name, x, y, width, height, "SET_SCISSOR");
    }

    fn draw(&mut self, vertex_count: u32, first_vertex: u32) {
        info!(
            pass = %self.pass_name,
            vertices = vertex_count,
            first = first_vertex,
            "DRAW"
        );
    }

    fn draw_indexed(&mut self, index_count: u32, first_index: u32, vertex_offset: i32) {
        info!(
            pass = %self.pass_name,
            indices = index_count,
            first = first_index,
            offset = vertex_offset,
            "DRAW_INDEXED"
        );
    }

    fn push_bytes(
        &mut self,
        _stages: i3_gfx::graph::pipeline::ShaderStageFlags,
        _offset: u32,
        _data: &[u8],
    ) {
        info!(pass = %self.pass_name, "PUSH_BYTES");
    }

    fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        info!(pass = %self.pass_name, x, y, z, "DISPATCH");
    }

    fn build_blas(&mut self, handle: BackendAccelerationStructure, update: bool) {
        info!(pass = %self.pass_name, ?handle, update, "BUILD_BLAS");
    }

    fn build_tlas(
        &mut self,
        handle: BackendAccelerationStructure,
        instances: &[TlasInstanceDesc],
        update: bool,
    ) {
        info!(pass = %self.pass_name, ?handle, instance_count = instances.len(), update, "BUILD_TLAS");
    }

    fn draw_indexed_indirect_count(
        &mut self,
        _indirect_buffer: BufferHandle,
        _indirect_offset: u64,
        _count_buffer: BufferHandle,
        _count_offset: u64,
        _max_draw_count: u32,
        _stride: u32,
    ) {
        info!(
            pass = %self.pass_name,
            max_draw_count = _max_draw_count, "DRAW_INDEXED_INDIRECT_COUNT"
        );
    }

    fn draw_indirect_count(
        &mut self,
        _indirect_buffer: BufferHandle,
        _indirect_offset: u64,
        _count_buffer: BufferHandle,
        _count_offset: u64,
        _max_draw_count: u32,
        _stride: u32,
    ) {
        info!(
            pass = %self.pass_name,
            max_draw_count = _max_draw_count, "DRAW_INDIRECT_COUNT"
        );
    }

    fn clear_buffer(&mut self, _buffer: BufferHandle, _clear_value: u32) {
        info!(pass = %self.pass_name, "CLEAR_BUFFER");
    }

    fn present(&mut self, image: ImageHandle) {
        info!(pass = %self.pass_name, ?image, "PRESENT");
        let pid = if let Some(physical) = self.image_map.get(&image) {
            physical.0
        } else {
            image.0.0
        };

        if !self.allocated_images.contains(&pid) {
            self.report_error(ValidationError::ResourceNotFound(pid));
        }
    }

    fn copy_buffer(
        &mut self,
        _src: BufferHandle,
        _dst: BufferHandle,
        _src_offset: u64,
        _dst_offset: u64,
        _size: u64,
    ) {
        info!(pass = %self.pass_name, ?_src, ?_dst, "COPY_BUFFER");
    }

    fn map_buffer(&mut self, _handle: BufferHandle) -> *mut u8 {
        info!(pass = %self.pass_name, ?_handle, "MAP_BUFFER");
        std::ptr::null_mut()
    }

    fn unmap_buffer(&mut self, _handle: BufferHandle) {
        info!(pass = %self.pass_name, ?_handle, "UNMAP_BUFFER");
    }
}

#[cfg(test)]
mod tests;
