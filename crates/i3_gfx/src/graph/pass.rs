use crate::graph::backend::{DescriptorWrite, PassContext, RenderBackend};
use crate::graph::types::{
    BufferHandle, ImageDesc, ImageHandle, PassDomain, ResourceUsage, WindowHandle,
};
use std::any::{Any, TypeId};

/// Context provided to a node during its recording phase.
/// This is a struct to allow for generic methods (publish/consume).
pub struct PassBuilder<'a> {
    pub(crate) inner: &'a mut dyn InternalPassBuilder,
}

impl<'a> PassBuilder<'a> {
    // --- Scoped Symbol Table ---
    /// Register a typed symbol in the current scope.
    pub fn publish<T: 'static + Send + Sync>(&mut self, name: &str, data: T) {
        self.inner
            .publish_erased(TypeId::of::<T>(), name, Box::new(data));
    }

    /// Resolve a typed symbol from the current or parent scope.
    pub fn consume<T: 'static + Send + Sync>(&mut self, name: &str) -> &T {
        let any = self.inner.consume_erased(TypeId::of::<T>(), name);
        any.downcast_ref::<T>()
            .expect("Type mismatch in symbol table")
    }

    // --- GPU Intents ---
    pub fn read_image(&mut self, handle: ImageHandle, usage: ResourceUsage) {
        self.inner.read_image(handle, usage);
    }
    pub fn write_image(&mut self, handle: ImageHandle, usage: ResourceUsage) {
        self.inner.write_image(handle, usage);
    }

    pub fn read_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage) {
        self.inner.read_buffer(handle, usage);
    }
    pub fn write_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage) {
        self.inner.write_buffer(handle, usage);
    }

    // --- Resource Generation ---
    /// Declares a new transient image and publishes it.
    pub fn declare_image(&mut self, name: &str, desc: ImageDesc) -> ImageHandle {
        self.inner.declare_image(name, desc)
    }

    /// Acquires a window-backed image (backbuffer) and publishes it.
    pub fn acquire_backbuffer(&mut self, window: WindowHandle) -> ImageHandle {
        self.inner.acquire_backbuffer(window)
    }

    /// Declares a new transient buffer and publishes it.
    pub fn declare_buffer(
        &mut self,
        name: &str,
        desc: crate::graph::types::BufferDesc,
    ) -> BufferHandle {
        self.inner.declare_buffer(name, desc)
    }

    /// Declares a persistent buffer that requires N-1 temporal history support.
    pub fn declare_buffer_history(
        &mut self,
        name: &str,
        desc: crate::graph::types::BufferDesc,
    ) -> crate::graph::types::BufferHandle {
        self.inner.declare_buffer_history(name, desc)
    }

    /// Reads the N-1 temporal history of a buffer as an external input.
    pub fn read_buffer_history(&mut self, name: &str) -> crate::graph::types::BufferHandle {
        self.inner.read_buffer_history(name)
    }

    /// Imports an existing physical buffer into the frame graph.
    pub fn import_buffer(
        &mut self,
        name: &str,
        physical: crate::graph::backend::BackendBuffer,
    ) -> BufferHandle {
        self.inner.import_buffer(name, physical)
    }

    pub fn bind_pipeline(&mut self, handle: crate::graph::types::PipelineHandle) {
        self.inner.bind_pipeline(handle);
    }

    pub fn bind_descriptor_set(&mut self, set_index: u32, writes: Vec<DescriptorWrite>) {
        self.inner.bind_descriptor_set(set_index, writes);
    }

    pub fn register_external_image(
        &mut self,
        handle: crate::graph::types::ImageHandle,
        physical: crate::graph::backend::BackendImage,
    ) {
        self.inner.register_external_image(handle, physical);
    }

    pub fn register_external_buffer(
        &mut self,
        handle: crate::graph::types::BufferHandle,
        physical: crate::graph::backend::BackendBuffer,
    ) {
        self.inner.register_external_buffer(handle, physical);
    }

    // --- Tree Construction ---
    /// Adds a structural node (Pass or Group) to the frame graph.
    pub fn add_node(&mut self, node: Box<dyn RenderPass>) {
        self.inner.add_node_erased(node);
    }

    /// Convenience method to add a render pass without explicit boxing.
    pub fn add_pass<P: RenderPass + 'static>(&mut self, pass: P) {
        self.add_node(Box::new(pass));
    }

    /// Helper to add a pass using two closures (record and execute).
    pub fn add_pass_from_closures<R, E>(&mut self, name: &str, record: R, execute: E)
    where
        R: FnMut(&mut PassBuilder) + Send + Sync + 'static,
        E: Fn(&mut dyn PassContext) + Send + Sync + 'static,
    {
        self.add_pass(SimplePass {
            name: name.to_string(),
            record,
            execute,
        });
    }
}

/// A Node in the Frame Graph (Pass or Group).
/// Users implement this trait to define rendering logic.
pub trait RenderPass: Any + Send + Sync {
    /// Name of the pass (used for debugging/profiling).
    fn name(&self) -> &str;

    /// Optional domain (Defaults to Graphics).
    fn domain(&self) -> PassDomain {
        PassDomain::Graphics
    }

    /// Called once after the graph is built, for creating pipelines/resources.
    fn init(&mut self, _backend: &mut dyn RenderBackend) {}

    /// Declare resource intents and symbols.
    fn record(&mut self, builder: &mut PassBuilder);

    /// Record GPU commands (optional for purely grouping nodes).
    fn execute(&self, _ctx: &mut dyn PassContext) {}
}

/// A simple pass implementation that uses closures.
pub struct SimplePass<R, E> {
    pub name: String,
    pub record: R,
    pub execute: E,
}

impl<R, E> RenderPass for SimplePass<R, E>
where
    R: FnMut(&mut PassBuilder) + Send + Sync + 'static,
    E: Fn(&mut dyn PassContext) + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        (self.record)(builder);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        (self.execute)(ctx);
    }
}

/// Internal trait to hide implementation details from the public PassBuilder API.
pub(crate) trait InternalPassBuilder {
    fn publish_erased(&mut self, type_id: TypeId, name: &str, data: Box<dyn Any + Send + Sync>);
    fn consume_erased(&self, type_id: TypeId, name: &str) -> &dyn Any;

    fn read_image(&mut self, handle: ImageHandle, usage: ResourceUsage);
    fn write_image(&mut self, handle: ImageHandle, usage: ResourceUsage);
    fn read_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage);
    fn write_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage);

    fn declare_image(&mut self, name: &str, desc: ImageDesc) -> ImageHandle;
    fn declare_buffer(&mut self, name: &str, desc: crate::graph::types::BufferDesc)
    -> BufferHandle;
    fn declare_buffer_history(
        &mut self,
        name: &str,
        desc: crate::graph::types::BufferDesc,
    ) -> BufferHandle;
    fn read_buffer_history(&mut self, name: &str) -> BufferHandle;

    fn import_buffer(
        &mut self,
        name: &str,
        physical: crate::graph::backend::BackendBuffer,
    ) -> BufferHandle;

    fn acquire_backbuffer(&mut self, window: WindowHandle) -> ImageHandle;

    fn bind_pipeline(&mut self, handle: crate::graph::types::PipelineHandle);
    fn bind_descriptor_set(&mut self, set_index: u32, writes: Vec<DescriptorWrite>);

    fn register_external_image(
        &mut self,
        handle: crate::graph::types::ImageHandle,
        physical: crate::graph::backend::BackendImage,
    );
    fn register_external_buffer(
        &mut self,
        handle: crate::graph::types::BufferHandle,
        physical: crate::graph::backend::BackendBuffer,
    );

    fn add_node_erased(&mut self, node: Box<dyn RenderPass>);
}
