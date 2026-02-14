use i3_gfx::graph::backend::{BackendBuffer, BackendImage, PassContext, RenderBackend};
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
    next_handle: u64,
}

impl NullBackend {
    pub fn new() -> Self {
        Self {
            allocated_images: HashSet::new(),
            allocated_buffers: HashSet::new(),
            allocated_pipelines: HashSet::new(),
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
    fn create_image(&mut self, desc: &i3_gfx::graph::backend::ImageDesc) -> BackendImage {
        let h = self.next_handle();
        self.allocated_images.insert(h);
        info!(handle = h, ?desc, "Created Image");
        BackendImage(h)
    }

    fn create_buffer(&mut self, desc: &i3_gfx::graph::backend::BufferDesc) -> BackendBuffer {
        let h = self.next_handle();
        self.allocated_buffers.insert(h);
        info!(handle = h, ?desc, "Created Buffer");
        BackendBuffer(h)
    }

    fn create_graphics_pipeline(
        &mut self,
        desc: &i3_gfx::graph::backend::GraphicsPipelineDesc,
    ) -> i3_gfx::graph::backend::BackendPipeline {
        let h = self.next_handle();
        self.allocated_pipelines.insert(h);
        info!(handle = h, name = %desc.name, "Created Graphics Pipeline");
        i3_gfx::graph::backend::BackendPipeline(h)
    }

    fn create_swapchain(&mut self, window_handle: u64, usages: u32) -> u64 {
        info!(window_handle, usages, "Created null swapchain");
        0x1337 // Dummy SC handle
    }

    fn present_swapchain(&mut self, sc_handle: u64, image: BackendImage) {
        info!(
            sc_handle,
            image_handle = image.0,
            "Presented null swapchain"
        );
    }

    fn begin_pass(&mut self, name: &str, f: Box<dyn FnOnce(&mut dyn PassContext) + Send + Sync>) {
        info!(name, "Beginning null pass");
        let mut ctx = NullPassContext::new(
            name,
            &self.allocated_images,
            &self.allocated_buffers,
            &self.allocated_pipelines,
        );
        f(&mut ctx);
    }

    fn resolve_image(
        &self,
        handle: i3_gfx::graph::types::ImageHandle,
    ) -> i3_gfx::graph::backend::BackendImage {
        // In NullBackend we can just return a dummy or look up if we added a map.
        // For now, let's just return a deterministic handle based on SymbolId.
        i3_gfx::graph::backend::BackendImage(handle.0.0)
    }

    fn resolve_buffer(
        &self,
        handle: i3_gfx::graph::types::BufferHandle,
    ) -> i3_gfx::graph::backend::BackendBuffer {
        i3_gfx::graph::backend::BackendBuffer(handle.0.0)
    }

    fn resolve_pipeline(
        &self,
        handle: i3_gfx::graph::types::PipelineHandle,
    ) -> i3_gfx::graph::backend::BackendPipeline {
        i3_gfx::graph::backend::BackendPipeline(handle.0.0)
    }

    fn register_external_image(
        &mut self,
        handle: i3_gfx::graph::types::ImageHandle,
        physical: i3_gfx::graph::backend::BackendImage,
    ) {
        info!(
            ?handle,
            ?physical,
            "Registered external image in NullBackend"
        );
    }
}

pub struct NullPassContext<'a> {
    pass_name: String,
    validation_failures: Vec<ValidationError>,
    allocated_images: &'a HashSet<u64>,
    allocated_buffers: &'a HashSet<u64>,
    allocated_pipelines: &'a HashSet<u64>,
}

impl<'a> NullPassContext<'a> {
    pub fn new(
        name: &str,
        allocated_images: &'a HashSet<u64>,
        allocated_buffers: &'a HashSet<u64>,
        allocated_pipelines: &'a HashSet<u64>,
    ) -> Self {
        Self {
            pass_name: name.to_string(),
            validation_failures: Vec::new(),
            allocated_images,
            allocated_buffers,
            allocated_pipelines,
        }
    }

    pub fn report_error(&mut self, err: ValidationError) {
        error!(pass = %self.pass_name, error = %err, "Validation Failure");
        self.validation_failures.push(err);
    }

    pub fn failures(&self) -> &[ValidationError] {
        &self.validation_failures
    }
}

impl<'a> PassContext for NullPassContext<'a> {
    fn bind_pipeline(&mut self, pipeline: i3_gfx::graph::types::PipelineHandle) {
        info!(pass = %self.pass_name, ?pipeline, "BIND_PIPELINE");
        if !self.allocated_pipelines.contains(&pipeline.0.0) {
            self.report_error(ValidationError::ResourceNotFound(pipeline.0.0));
        }
    }

    fn bind_image(&mut self, slot: u32, handle: i3_gfx::graph::types::ImageHandle) {
        info!(pass = %self.pass_name, slot, ?handle, "BIND_IMAGE");
        if !self.allocated_images.contains(&handle.0.0) {
            self.report_error(ValidationError::ResourceNotFound(handle.0.0));
        }
    }

    fn bind_buffer(&mut self, slot: u32, handle: i3_gfx::graph::types::BufferHandle) {
        info!(pass = %self.pass_name, slot, ?handle, "BIND_BUFFER");
        if !self.allocated_buffers.contains(&handle.0.0) {
            self.report_error(ValidationError::ResourceNotFound(handle.0.0));
        }
    }

    fn draw(&mut self, vertex_count: u32, first_vertex: u32) {
        info!(
            pass = %self.pass_name,
            vertices = vertex_count,
            first = first_vertex,
            "DRAW"
        );
    }

    fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        info!(pass = %self.pass_name, x, y, z, "DISPATCH");
    }

    fn present(&mut self, handle: i3_gfx::graph::types::ImageHandle) {
        info!(pass = %self.pass_name, ?handle, "PRESENT");
    }
}

#[cfg(test)]
mod tests;
