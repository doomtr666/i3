use crate::graph::backend::{
    DescriptorImageLayout, DescriptorWrite, PassContext, RenderBackend, SamplerHandle,
};
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

    /// Resolves an AccelerationStructureHandle by name, or returns None if not published.
    pub fn try_resolve_acceleration_structure(
        &mut self,
        name: &str,
    ) -> Option<crate::graph::types::AccelerationStructureHandle> {
        self.inner.try_resolve_acceleration_structure(name)
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
        // AS dependencies are handled via sync passes or manual barriers in backend.
    }

    pub fn read_acceleration_structure(
        &mut self,
        _handle: crate::graph::types::AccelerationStructureHandle,
        _usage: ResourceUsage,
    ) {
        // AS read intent — sync handled by build passes ordering.
    }

    /// Imports an existing physical acceleration structure into the frame graph.
    /// Returns a virtual `AccelerationStructureHandle` that can be used in descriptor bindings.
    pub fn import_acceleration_structure(
        &mut self,
        name: &str,
        physical: crate::graph::backend::BackendAccelerationStructure,
    ) -> crate::graph::types::AccelerationStructureHandle {
        self.inner.import_acceleration_structure(name, physical)
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

    /// Premium fluent API for binding descriptor sets.
    pub fn descriptor_set<F>(&mut self, set_index: u32, f: F)
    where
        F: FnOnce(&mut DescriptorSetWriter),
    {
        let mut writer = DescriptorSetWriter::new();
        f(&mut writer);
        self.bind_descriptor_set(set_index, writer.writes);
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

/// Fluent builder for descriptor set writes.
pub struct DescriptorSetWriter {
    pub(crate) writes: Vec<DescriptorWrite>,
    current_binding: u32,
}

impl DescriptorSetWriter {
    pub fn new() -> Self {
        Self {
            writes: Vec::new(),
            current_binding: 0,
        }
    }

    /// Explicitly sets the binding index for the next write.
    pub fn bind(&mut self, binding: u32) -> &mut Self {
        self.current_binding = binding;
        self
    }

    /// Modifies the array element of the last added write.
    pub fn at(&mut self, element: u32) -> &mut Self {
        if let Some(last) = self.writes.last_mut() {
            last.array_element = element;
        }
        self
    }

    pub fn storage_buffer(&mut self, buffer: BufferHandle) -> &mut Self {
        let b = self.current_binding;
        self.writes.push(DescriptorWrite::storage_buffer(b, 0, buffer));
        self.current_binding += 1;
        self
    }

    pub fn uniform_buffer(&mut self, buffer: BufferHandle) -> &mut Self {
        let b = self.current_binding;
        self.writes.push(DescriptorWrite::uniform_buffer(b, 0, buffer));
        self.current_binding += 1;
        self
    }

    pub fn combined_image_sampler(
        &mut self,
        image: ImageHandle,
        layout: DescriptorImageLayout,
        sampler: SamplerHandle,
    ) -> &mut Self {
        let b = self.current_binding;
        self.writes
            .push(DescriptorWrite::combined_image_sampler(
                b, 0, image, layout, sampler,
            ));
        self.current_binding += 1;
        self
    }

    pub fn texture(&mut self, image: ImageHandle, layout: DescriptorImageLayout) -> &mut Self {
        let b = self.current_binding;
        self.writes.push(DescriptorWrite::texture(b, 0, image, layout));
        self.current_binding += 1;
        self
    }

    pub fn sampler(&mut self, sampler: SamplerHandle) -> &mut Self {
        let b = self.current_binding;
        self.writes.push(DescriptorWrite::sampler(b, 0, sampler));
        self.current_binding += 1;
        self
    }

    pub fn acceleration_structure(&mut self, handle: crate::graph::types::AccelerationStructureHandle) -> &mut Self {
        let b = self.current_binding;
        self.writes.push(DescriptorWrite::acceleration_structure(b, 0, handle));
        self.current_binding += 1;
        self
    }

    // --- Shortcuts ---

    pub fn storage_buffer_at(&mut self, binding: u32, buffer: BufferHandle) -> &mut Self {
        self.bind(binding).storage_buffer(buffer)
    }
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

    fn import_buffer(&mut self, name: &str, physical: crate::graph::backend::BackendBuffer) -> BufferHandle;

    fn import_acceleration_structure(
        &mut self,
        name: &str,
        physical: crate::graph::backend::BackendAccelerationStructure,
    ) -> crate::graph::types::AccelerationStructureHandle;

    fn try_resolve_acceleration_structure(
        &mut self,
        name: &str,
    ) -> Option<crate::graph::types::AccelerationStructureHandle>;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::SymbolId;

    #[test]
    fn test_descriptor_set_writer_fluent() {
        let mut writer = DescriptorSetWriter::new();
        let buffer = BufferHandle(SymbolId(1));
        let image = ImageHandle(SymbolId(2));
        let sampler = SamplerHandle(3);

        writer.bind(0)
            .storage_buffer(buffer)
            .uniform_buffer(buffer).at(1)
            .bind(5)
            .combined_image_sampler(image, DescriptorImageLayout::ShaderReadOnlyOptimal, sampler);

        assert_eq!(writer.writes.len(), 3);

        // Binding 0: storage_buffer
        assert_eq!(writer.writes[0].binding, 0);
        assert_eq!(writer.writes[0].array_element, 0);

        // Binding 1: uniform_buffer (auto-inc from 0), then .at(1)
        assert_eq!(writer.writes[1].binding, 1);
        assert_eq!(writer.writes[1].array_element, 1);

        // Binding 5: combined_image_sampler
        assert_eq!(writer.writes[2].binding, 5);
        assert_eq!(writer.writes[2].array_element, 0);
    }
}

