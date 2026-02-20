use crate::graph::backend::{
    BackendBuffer, BackendImage, DescriptorWrite, PassContext, PassDescriptor,
    RenderBackendInternal,
};
use crate::graph::pass::{InternalPassBuilder, Node, PassBuilder};
use crate::graph::types::*;
use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Metadata and data for an entry in the symbol table.
pub struct Symbol {
    pub name: String,
    pub symbol_type: SymbolType,
    pub lifetime: SymbolLifetime,
    pub data: Option<Box<dyn Any + Send + Sync>>,
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Symbol")
            .field("name", &self.name)
            .field("symbol_type", &self.symbol_type)
            .field("lifetime", &self.lifetime)
            .field("has_data", &self.data.is_some())
            .finish()
    }
}

/// A scope-local Symbol Table.
#[derive(Debug)]
pub struct SymbolTable {
    pub(crate) symbols: Vec<Symbol>,
    name_to_id: HashMap<String, SymbolId>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            symbols: Vec::new(),
            name_to_id: HashMap::new(),
        }
    }

    pub fn publish(&mut self, name: &str, symbol: Symbol) -> SymbolId {
        let id = SymbolId(self.symbols.len() as u64);
        self.name_to_id.insert(name.to_string(), id);
        self.symbols.push(symbol);
        id
    }

    pub fn resolve(&self, name: &str) -> Option<SymbolId> {
        self.name_to_id.get(name).copied()
    }

    pub fn get_data(&self, id: SymbolId) -> Option<&dyn Any> {
        self.symbols
            .get(id.0 as usize)
            .and_then(|s| s.data.as_ref().map(|d| d.as_ref() as &dyn Any))
    }
}

/// Storage for a specific node and its children.
pub struct NodeStorage {
    pub name: String,
    pub domain: PassDomain,
    pub symbols: SymbolTable,
    pub children: Vec<NodeStorage>,
    pub execute: Option<Box<dyn FnOnce(&mut dyn PassContext) + Send + Sync>>,

    pub pipeline: Option<PipelineHandle>,

    // Captured intents (for Leaf nodes)
    pub image_reads: Vec<(ImageHandle, ResourceUsage)>,
    pub image_writes: Vec<(ImageHandle, ResourceUsage)>,
    pub buffer_reads: Vec<(BufferHandle, ResourceUsage)>,
    pub buffer_writes: Vec<(BufferHandle, ResourceUsage)>,

    pub external_images: Vec<(ImageHandle, BackendImage)>,
    pub swapchain_requests: Vec<(ImageHandle, WindowHandle)>,
    pub descriptor_sets: Vec<(u32, Vec<DescriptorWrite>)>,
}

impl std::fmt::Debug for NodeStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeStorage")
            .field("name", &self.name)
            .field("domain", &self.domain)
            .field("symbols", &self.symbols)
            .field("children", &self.children)
            .field("has_execute", &self.execute.is_some())
            .field("image_reads", &self.image_reads)
            .field("image_writes", &self.image_writes)
            .field("external_images", &self.external_images)
            .field("swapchain_requests", &self.swapchain_requests)
            .finish()
    }
}

impl Node for NodeStorage {
    fn name(&self) -> &str {
        &self.name
    }
    fn domain(&self) -> PassDomain {
        self.domain
    }
}

/// Implementation of the internal PassBuilder trait.
pub struct PassRecorder<'a> {
    storage: &'a mut NodeStorage,
    parent_symbols: Option<&'a SymbolTable>,
}

impl<'a> InternalPassBuilder for PassRecorder<'a> {
    fn publish_erased(&mut self, _type_id: TypeId, name: &str, data: Box<dyn Any + Send + Sync>) {
        tracing::trace!(name, "Publishing CPU data");
        self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::CpuData(_type_id),
                lifetime: SymbolLifetime::Transient,
                data: Some(data),
            },
        );
    }

    fn consume_erased(&self, _type_id: TypeId, name: &str) -> &dyn Any {
        if let Some(id) = self.storage.symbols.resolve(name) {
            tracing::trace!(name, "Consuming CPU data (local)");
            return self
                .storage
                .symbols
                .get_data(id)
                .expect("Symbol exists but has no data");
        } else if let Some(parent) = self.parent_symbols {
            if let Some(id) = parent.resolve(name) {
                tracing::trace!(name, "Consuming CPU data (inherited)");
                return parent
                    .get_data(id)
                    .expect("Symbol in parent exists but has no data");
            }
        }

        panic!("Symbol '{}' not found in current or parent scope", name);
    }

    fn read_image(&mut self, handle: ImageHandle, usage: ResourceUsage) {
        self.storage.image_reads.push((handle, usage));
    }

    fn write_image(&mut self, handle: ImageHandle, usage: ResourceUsage) {
        self.storage.image_writes.push((handle, usage));
    }

    fn bind_pipeline(&mut self, handle: PipelineHandle) {
        self.storage.pipeline = Some(handle);
    }

    fn bind_descriptor_set(&mut self, set_index: u32, writes: Vec<DescriptorWrite>) {
        self.storage.descriptor_sets.push((set_index, writes));
    }

    fn register_external_image(&mut self, handle: ImageHandle, physical: BackendImage) {
        self.storage.external_images.push((handle, physical));
    }

    fn read_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage) {
        self.storage.buffer_reads.push((handle, usage));
    }

    fn write_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage) {
        self.storage.buffer_writes.push((handle, usage));
    }

    fn declare_image(&mut self, name: &str, desc: ImageDesc) -> ImageHandle {
        let id = self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::Image(desc),
                lifetime: SymbolLifetime::Transient,
                data: None, // Will update below
            },
        );
        let actual_handle = ImageHandle(id);
        self.storage.symbols.symbols[id.0 as usize].data = Some(Box::new(actual_handle));
        actual_handle
    }

    fn declare_buffer(&mut self, name: &str, desc: BufferDesc) -> BufferHandle {
        let id = self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::Buffer(desc),
                lifetime: SymbolLifetime::Transient,
                data: None,
            },
        );
        let actual_handle = BufferHandle(id);
        self.storage.symbols.symbols[id.0 as usize].data = Some(Box::new(actual_handle));
        actual_handle
    }

    fn acquire_backbuffer(&mut self, window: WindowHandle) -> ImageHandle {
        let name = format!("Window_{}", window.0);
        let id = self.storage.symbols.publish(
            &name,
            Symbol {
                name: name.clone(),
                symbol_type: SymbolType::Image(ImageDesc {
                    width: 1280,
                    height: 720,
                    depth: 1,
                    format: Format::B8G8R8A8_SRGB, // Force SRGB logic match
                    mip_levels: 1,
                    array_layers: 1,
                    usage: ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::TRANSFER_DST,
                    view_type: crate::graph::types::ImageViewType::Type2D,
                    swizzle: crate::graph::types::ComponentMapping::default(),
                }),
                lifetime: SymbolLifetime::External,
                data: None,
            },
        );
        let actual_handle = ImageHandle(id);
        self.storage.symbols.symbols[id.0 as usize].data = Some(Box::new(actual_handle));

        // Record the request
        self.storage
            .swapchain_requests
            .push((actual_handle, window));

        actual_handle
    }

    fn add_node_erased(
        &mut self,
        name: &str,
        setup: Box<
            dyn FnOnce(
                &mut dyn InternalPassBuilder,
            ) -> Box<dyn FnOnce(&mut dyn PassContext) + Send + Sync>,
        >,
    ) {
        let mut child_storage = NodeStorage {
            name: name.to_string(),
            domain: PassDomain::Graphics,
            symbols: SymbolTable::new(),
            children: Vec::new(),
            execute: None,
            pipeline: None,
            image_reads: Vec::new(),
            image_writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: Vec::new(),
            external_images: Vec::new(),
            swapchain_requests: Vec::new(),
            descriptor_sets: Vec::new(),
        };

        {
            let mut child_recorder = PassRecorder {
                storage: &mut child_storage,
                parent_symbols: Some(&self.storage.symbols),
            };

            let execute = setup(&mut child_recorder);
            child_storage.execute = Some(execute);
        }

        self.storage.children.push(child_storage);
    }
}

/// Root of the Frame Graph recording.
pub struct FrameGraph {
    root: NodeStorage,
}

impl FrameGraph {
    pub fn new() -> Self {
        Self {
            root: NodeStorage {
                name: "root".to_string(),
                domain: PassDomain::Cpu,
                symbols: SymbolTable::new(),
                children: Vec::new(),
                execute: None,
                pipeline: None,
                image_reads: Vec::new(),
                image_writes: Vec::new(),
                buffer_reads: Vec::new(),
                buffer_writes: Vec::new(),
                external_images: Vec::new(),
                swapchain_requests: Vec::new(),
                descriptor_sets: Vec::new(),
            },
        }
    }

    pub fn record<F>(&mut self, setup: F)
    where
        F: FnOnce(&mut PassBuilder),
    {
        let mut recorder = PassRecorder {
            storage: &mut self.root,
            parent_symbols: None,
        };

        let mut builder = PassBuilder {
            inner: &mut recorder,
        };
        setup(&mut builder);
    }

    pub fn compile(self) -> CompiledGraph {
        tracing::debug!("Compiling hierarchical frame graph");
        CompiledGraph { _root: self.root }
    }
}

pub struct CompiledGraph {
    _root: NodeStorage,
}

impl CompiledGraph {
    pub fn execute(self, backend: &mut dyn RenderBackendInternal) -> Result<Option<u64>, String> {
        tracing::debug!("Executing hierarchical frame graph");

        // Track transient resources for cleanup
        let mut transient_images = Vec::new();
        let mut transient_buffers = Vec::new();

        // 1. Resource Resolution & Allocation
        self.resolve_resources_recursive(
            &self._root,
            backend,
            &mut transient_images,
            &mut transient_buffers,
        );

        // 2. Execution
        backend.begin_frame();
        // Use a simple Vec for inactive images - usually 0 or 1, much faster than HashSet
        let mut inactive_images = Vec::with_capacity(2);
        let result = Self::execute_node_recursive(self._root, backend, &mut inactive_images);

        // Final Submission (Phase 0: pure submission of recorded commands)
        let _ = backend
            .submit(crate::graph::backend::CommandBatch::default(), &[], &[])
            .map_err(|e| e.to_string())?;

        backend.end_frame();

        // 3. Cleanup Transient Resources
        for image in transient_images {
            backend.release_transient_image(image);
        }
        for buffer in transient_buffers {
            backend.release_transient_buffer(buffer);
        }

        result
    }

    fn resolve_resources_recursive(
        &self,
        node: &NodeStorage,
        backend: &mut dyn RenderBackendInternal,
        transient_images: &mut Vec<BackendImage>,
        transient_buffers: &mut Vec<BackendBuffer>,
    ) {
        // Resolve symbols in current scope
        for symbol in &node.symbols.symbols {
            match symbol.symbol_type {
                SymbolType::Image(ref desc) => {
                    if symbol.lifetime == SymbolLifetime::Transient {
                        let physical = backend.create_transient_image(desc);
                        transient_images.push(physical);
                        let handle = symbol
                            .data
                            .as_ref()
                            .expect("Image without handle")
                            .downcast_ref::<ImageHandle>()
                            .expect("Not a handle")
                            .clone();
                        backend.register_external_image(handle, physical);
                    }
                }
                SymbolType::Buffer(ref desc) => {
                    if symbol.lifetime == SymbolLifetime::Transient {
                        let physical = backend.create_transient_buffer(desc);
                        transient_buffers.push(physical);
                    }
                }
                _ => {}
            }
        }

        // Recurse
        for child in &node.children {
            self.resolve_resources_recursive(child, backend, transient_images, transient_buffers);
        }
    }

    fn execute_node_recursive(
        mut node: NodeStorage,
        backend: &mut dyn RenderBackendInternal,
        inactive_images: &mut Vec<u64>,
    ) -> Result<Option<u64>, String> {
        // 0. Process Swapchain Requests (Automatic Acquire)
        for (handle, window) in node.swapchain_requests {
            match backend
                .acquire_swapchain_image(window)
                .map_err(|e| e.to_string())?
            {
                Some((physical, _sem, _idx)) => {
                    // Register automatically
                    backend.register_external_image(handle, physical);
                }
                None => {
                    // Mark this handle as inactive (minimized)
                    tracing::debug!(window = ?window.0, "Window is minimized, skipping associated passes");
                    inactive_images.push(handle.0.0);
                }
            }
        }

        // Register external resources first
        for (virtual_handle, physical) in node.external_images {
            backend.register_external_image(virtual_handle, physical);
        }

        let mut last_sem = None;

        // If this node has an execute closure, it's a pass
        if let Some(execute) = node.execute.take() {
            // Check if any write target is inactive first to avoid heavy descriptor allocation
            let is_inactive = if inactive_images.is_empty() {
                false
            } else {
                node.image_writes
                    .iter()
                    .any(|(h, _)| inactive_images.contains(&h.0.0))
            };

            if !is_inactive {
                let desc = PassDescriptor {
                    name: &node.name,
                    pipeline: node.pipeline,
                    image_reads: &node.image_reads,
                    image_writes: &node.image_writes,
                    buffer_reads: &node.buffer_reads,
                    buffer_writes: &node.buffer_writes,
                    descriptor_sets: &node.descriptor_sets,
                };
                tracing::debug!(pass = %desc.name, writes = ?desc.image_writes.len(), "Executing pass");
                last_sem = Some(backend.begin_pass(desc, execute));
            } else {
                tracing::debug!(pass = %node.name, "Skipping pass (targets inactive image)");
            }
        }

        // Execute children
        for child in node.children {
            if let Some(sem) = Self::execute_node_recursive(child, backend, inactive_images)? {
                last_sem = Some(sem);
            }
        }

        Ok(last_sem)
    }
}
