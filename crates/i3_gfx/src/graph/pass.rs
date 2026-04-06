use crate::graph::backend::{DescriptorWrite, PassContext, RenderBackend};
use crate::graph::compiler::FrameBlackboard;
use crate::graph::types::{BufferHandle, ImageDesc, ImageHandle, ResourceUsage, WindowHandle};
use std::any::{Any, TypeId};

/// Context provided to a node during its recording phase.
/// This is a struct to allow for generic methods (publish/consume).
pub struct PassBuilder<'a> {
    pub(crate) inner: &'a mut dyn InternalPassBuilder,
}

impl<'a> PassBuilder<'a> {
    pub(crate) fn new(inner: &'a mut dyn InternalPassBuilder) -> Self {
        Self { inner }
    }

    // --- Scoped Symbol Table ---
    /// Register a typed symbol in the current scope.
    pub fn publish<T: 'static + Send + Sync>(&mut self, name: &str, data: T) {
        self.inner
            .publish_erased(TypeId::of::<T>(), name, Box::new(data));
    }

    /// Resolve a typed symbol from the current or parent scope. Panics if not found.
    pub fn consume<T: 'static + Send + Sync>(&mut self, name: &str) -> &T {
        let any = self.inner.consume_erased(TypeId::of::<T>(), name);
        any.downcast_ref::<T>()
            .unwrap_or_else(|| panic!("Type mismatch in symbol table for symbol: {}", name))
    }

    /// Resolve a typed symbol from the current or parent scope. Returns None if not found.
    pub fn try_consume<T: 'static + Send + Sync>(&mut self, name: &str) -> Option<&T> {
        self.inner
            .try_consume_erased(TypeId::of::<T>(), name)
            .map(|any| {
                any.downcast_ref::<T>().unwrap_or_else(|| {
                    panic!(
                        "Type mismatch in symbol table for optional symbol: {}",
                        name
                    )
                })
            })
    }

    /// Resolves an ImageHandle from the symbol table by name.
    pub fn resolve_image(&mut self, name: &str) -> ImageHandle {
        *self.consume::<ImageHandle>(name)
    }

    /// Resolves a BufferHandle from the symbol table by name.
    pub fn resolve_buffer(&mut self, name: &str) -> BufferHandle {
        *self.consume::<BufferHandle>(name)
    }

    // --- GPU Intents ---
    pub fn read_image(&mut self, handle: ImageHandle, usage: ResourceUsage) {
        self.inner.read_image(handle, usage);
    }
    pub fn write_image(&mut self, handle: ImageHandle, usage: ResourceUsage) {
        self.inner.write_image(handle, usage);
    }

    /// Declare that this pass will transition `handle` to PresentSrc at the end of execution.
    /// Must be called from `declare()`. The sync planner emits the layout transition as a
    /// post-pass barrier automatically; no manual barrier is needed in `execute()`.
    pub fn present_image(&mut self, handle: ImageHandle) {
        self.inner.declare_present_image(handle);
    }

    pub fn write_acceleration_structure(
        &mut self,
        _handle: crate::graph::backend::BackendAccelerationStructure,
        _usage: ResourceUsage,
    ) {
        // For now, AS dependencies are handled via sync passes or manual barriers in backend.
        // This method allows the renderer to declare intent.
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

    /// Declares a new transient image, publishes it, and promotes it to the parent scope.
    pub fn declare_image_output(&mut self, name: &str, desc: ImageDesc) -> ImageHandle {
        self.inner.declare_image_output(name, desc)
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

    /// Declares a new transient buffer, publishes it, and promotes it to the parent scope.
    pub fn declare_buffer_output(
        &mut self,
        name: &str,
        desc: crate::graph::types::BufferDesc,
    ) -> BufferHandle {
        self.inner.declare_buffer_output(name, desc)
    }

    /// Declares a persistent buffer that requires N-1 temporal history support.
    pub fn declare_buffer_history(
        &mut self,
        name: &str,
        desc: crate::graph::types::BufferDesc,
    ) -> crate::graph::types::BufferHandle {
        self.inner.declare_buffer_history(name, desc)
    }

    /// Declares a persistent buffer with temporal history, promoted to the parent scope.
    pub fn declare_buffer_history_output(
        &mut self,
        name: &str,
        desc: crate::graph::types::BufferDesc,
    ) -> crate::graph::types::BufferHandle {
        self.inner.declare_buffer_history_output(name, desc)
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
    /// Adds a structural node (Pass or Group) to the frame graph by reference.
    pub fn add_pass(&mut self, pass: &mut dyn RenderPass) {
        let trait_ptr: *mut dyn RenderPass = pass;
        // Cast to 'static to satisfy Box requirements.
        // Safety: The pass must outlive the FrameGraph execution for this frame.
        let static_ptr: *mut (dyn RenderPass + 'static) = unsafe { std::mem::transmute(trait_ptr) };
        self.inner
            .add_node_erased(Box::new(BoxedRef { inner: static_ptr }));
    }

    /// Adds an owned structural node to the frame graph.
    pub fn add_owned_pass<P: RenderPass + 'static>(&mut self, pass: P) {
        self.inner.add_node_erased(Box::new(pass));
    }
}

/// Internal wrapper to bridge &mut dyn RenderPass to Box<dyn RenderPass + 'static>.
struct BoxedRef {
    inner: *mut (dyn RenderPass + 'static),
}

unsafe impl Send for BoxedRef {}
unsafe impl Sync for BoxedRef {}

impl RenderPass for BoxedRef {
    fn name(&self) -> &str {
        unsafe { (*self.inner).name() }
    }

    fn prefer_async(&self) -> bool {
        unsafe { (*self.inner).prefer_async() }
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        unsafe { (*self.inner).init(backend, globals) }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        unsafe { (*self.inner).declare(builder) }
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        unsafe { (*self.inner).execute(ctx, frame) }
    }
}

/// A Node in the Frame Graph (Pass or Group).
/// Users implement this trait to define rendering logic.
pub trait RenderPass: Any + Send + Sync {
    /// Name of the pass (used for debugging/profiling).
    fn name(&self) -> &str;

    /// Hint: request execution on a dedicated async queue.
    /// Only meaningful for Compute/Transfer domains. Default: false.
    fn prefer_async(&self) -> bool {
        false
    }

    /// Called once during graph global initialization, for creating pipelines/resources.
    /// Can consume services from the global scope.
    fn init(&mut self, _backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {}

    /// Declare resource intents and symbols.
    fn declare(&mut self, builder: &mut PassBuilder);

    /// Record GPU commands (optional for purely grouping nodes).
    fn execute(&self, _ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {}
}

/// Internal trait to hide implementation details from the public PassBuilder API.
pub(crate) trait InternalPassBuilder {
    fn publish_erased(&mut self, type_id: TypeId, name: &str, data: Box<dyn Any + Send + Sync>);
    fn consume_erased(&mut self, type_id: TypeId, name: &str) -> &dyn Any;
    fn try_consume_erased(&mut self, type_id: TypeId, name: &str) -> Option<&dyn Any>;

    fn read_image(&mut self, handle: ImageHandle, usage: ResourceUsage);
    fn write_image(&mut self, handle: ImageHandle, usage: ResourceUsage);
    fn declare_present_image(&mut self, handle: ImageHandle);
    fn read_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage);
    fn write_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage);

    fn declare_image(&mut self, name: &str, desc: ImageDesc) -> ImageHandle;
    fn declare_image_output(&mut self, name: &str, desc: ImageDesc) -> ImageHandle;
    fn declare_buffer(&mut self, name: &str, desc: crate::graph::types::BufferDesc)
    -> BufferHandle;
    fn declare_buffer_output(
        &mut self,
        name: &str,
        desc: crate::graph::types::BufferDesc,
    ) -> BufferHandle;
    fn declare_buffer_history(
        &mut self,
        name: &str,
        desc: crate::graph::types::BufferDesc,
    ) -> BufferHandle;
    fn declare_buffer_history_output(
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
