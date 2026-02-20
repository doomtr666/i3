use crate::graph::backend::{DescriptorWrite, PassContext};
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

    // --- Tree Construction ---
    /// Adds a sub-node to the current node.
    pub fn add_node<F, E>(&mut self, name: &str, setup: F)
    where
        F: FnOnce(&mut PassBuilder) -> E + 'static,
        E: FnOnce(&mut dyn PassContext) + Send + Sync + 'static,
    {
        // Wrap the setup closure to handle the PassBuilder struct
        let wrapped_setup = Box::new(move |inner_builder: &mut dyn InternalPassBuilder| {
            let mut builder = PassBuilder {
                inner: inner_builder,
            };
            let execute = setup(&mut builder);
            Box::new(execute) as Box<dyn FnOnce(&mut dyn PassContext) + Send + Sync>
        });

        self.inner.add_node_erased(name, wrapped_setup);
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
    fn acquire_backbuffer(&mut self, window: WindowHandle) -> ImageHandle;

    fn bind_pipeline(&mut self, handle: crate::graph::types::PipelineHandle);
    fn bind_descriptor_set(&mut self, set_index: u32, writes: Vec<DescriptorWrite>);

    fn register_external_image(
        &mut self,
        handle: crate::graph::types::ImageHandle,
        physical: crate::graph::backend::BackendImage,
    );
    fn add_node_erased(
        &mut self,
        name: &str,
        setup: Box<
            dyn FnOnce(
                &mut dyn InternalPassBuilder,
            ) -> Box<dyn FnOnce(&mut dyn PassContext) + Send + Sync>,
        >,
    );
}

/// A Node in the Frame Graph (either a Pass or a Group).
/// (Internal trait used by the compiler to manage the tree).
pub trait Node: Send + Sync {
    fn name(&self) -> &str;
    fn domain(&self) -> PassDomain;
}
