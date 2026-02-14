use crate::graph::backend::{PassContext, RenderBackend};
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

    // Captured intents (for Leaf nodes)
    pub image_reads: Vec<(ImageHandle, ResourceUsage)>,
    pub image_writes: Vec<(ImageHandle, ResourceUsage)>,
    pub buffer_reads: Vec<(BufferHandle, ResourceUsage)>,
    pub buffer_writes: Vec<(BufferHandle, ResourceUsage)>,
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

    fn acquire_backbuffer(&mut self, window: WindowHandle) -> ImageHandle {
        let name = format!("Window_{}", window.0);
        let id = self.storage.symbols.publish(
            &name,
            Symbol {
                name: name.clone(),
                symbol_type: SymbolType::Image(ImageDesc {
                    width: 1280,
                    height: 720,
                    format: 0,
                }),
                lifetime: SymbolLifetime::External,
                data: None,
            },
        );
        let actual_handle = ImageHandle(id);
        self.storage.symbols.symbols[id.0 as usize].data = Some(Box::new(actual_handle));
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
            image_reads: Vec::new(),
            image_writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: Vec::new(),
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
                image_reads: Vec::new(),
                image_writes: Vec::new(),
                buffer_reads: Vec::new(),
                buffer_writes: Vec::new(),
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
    pub fn execute(self, backend: &mut dyn RenderBackend) {
        tracing::debug!("Executing hierarchical frame graph");

        // 1. Resource Resolution & Allocation
        // We simple-mindedly walk all symbols and allocate what's needed.
        // In a real engine, this would use a proper allocator and consider lifetimes.
        self.resolve_resources_recursive(&self._root, backend);

        // 2. Execution
        Self::execute_node_recursive(self._root, backend);
    }

    fn resolve_resources_recursive(&self, node: &NodeStorage, backend: &mut dyn RenderBackend) {
        // Resolve symbols in current scope
        for symbol in &node.symbols.symbols {
            match symbol.symbol_type {
                SymbolType::Image(ref desc) => {
                    if symbol.lifetime == SymbolLifetime::Transient
                        || symbol.lifetime == SymbolLifetime::Persistent
                    {
                        let _physical = backend.create_image(desc);
                        // We could store the mapping here if CompiledGraph wasn't read-only,
                        // but the Backend already has the resolve methods for the pass closures.
                        // Actually, the backend needs to KNOW the mapping we just created.
                        // So we "tell" the backend about it if it's internal.
                        // wait, if it's ImageHandle(SymbolId), the SymbolId is stable!
                    }
                }
                SymbolType::Buffer(ref desc) => {
                    if symbol.lifetime == SymbolLifetime::Transient
                        || symbol.lifetime == SymbolLifetime::Persistent
                    {
                        let _physical = backend.create_buffer(desc);
                    }
                }
                _ => {}
            }
        }

        // Recurse
        for child in &node.children {
            self.resolve_resources_recursive(child, backend);
        }
    }

    fn execute_node_recursive(mut node: NodeStorage, backend: &mut dyn RenderBackend) {
        // If this node has an execute closure, it's a pass
        if let Some(execute) = node.execute.take() {
            backend.begin_pass(&node.name, execute);
        }

        // Execute children
        for child in node.children {
            Self::execute_node_recursive(child, backend);
        }
    }
}
