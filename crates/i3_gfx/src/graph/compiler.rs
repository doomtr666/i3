use crate::graph::backend::{
    BackendBuffer, BackendCommandBuffer, BackendImage, BatchStep, CommandBatch, DescriptorWrite,
    DeviceCapabilities, PassDescriptor, RenderBackend, RenderBackendInternal,
};
use crate::graph::pass::{InternalPassBuilder, PassBuilder, RenderPass};
use crate::graph::types::*;
use rayon::prelude::*;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_NODE_ID: AtomicU64 = AtomicU64::new(1);

struct SyncPtr<T>(*mut T);

impl<T> Clone for SyncPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

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

    pub fn publish(&mut self, name: &str, symbol: Symbol, id: SymbolId) -> SymbolId {
        self.name_to_id.insert(name.to_string(), id);
        self.symbols.push(symbol);
        id
    }

    pub fn resolve(&self, name: &str) -> Option<SymbolId> {
        self.name_to_id.get(name).copied()
    }

    pub fn get_data(&self, id: SymbolId) -> Option<&dyn Any> {
        let index = (id.0 & 0xFFFFFFFF) as usize;
        self.symbols
            .get(index)
            .and_then(|s| s.data.as_ref().map(|d| d.as_ref() as &dyn Any))
    }
}

/// Storage for a specific node and its children.
pub struct NodeStorage {
    pub node_id: u64,
    pub name: String,
    pub symbols: SymbolTable,
    pub children: Vec<NodeStorage>,
    pub pass: Option<Box<dyn RenderPass>>,

    pub pipeline: Option<PipelineHandle>,

    // Captured intents (for Leaf nodes)
    pub image_reads: Vec<(ImageHandle, ResourceUsage)>,
    pub image_writes: Vec<(ImageHandle, ResourceUsage)>,
    pub buffer_reads: Vec<(BufferHandle, ResourceUsage)>,
    pub buffer_writes: Vec<(BufferHandle, ResourceUsage)>,

    /// CPU data symbols published by this node (write dependency).
    pub data_writes: Vec<String>,
    /// CPU data symbols consumed by this node (read dependency).
    pub data_reads: Vec<String>,

    pub external_images: Vec<(ImageHandle, BackendImage)>,
    pub external_buffers: Vec<(BufferHandle, BackendBuffer)>,
    pub swapchain_requests: Vec<(ImageHandle, WindowHandle)>,
    pub descriptor_sets: Vec<(u32, Vec<DescriptorWrite>)>,
    pub prefer_async: bool,
    /// Images that must be transitioned to PresentSrc AFTER this pass executes.
    /// Tracked separately from image_writes so the planner can emit a post-transition
    /// rather than merging PRESENT with the pass's main usage (which would lose the
    /// correct target layout/stage via priority ordering in get_image_state).
    pub present_images: Vec<ImageHandle>,
}

impl std::fmt::Debug for NodeStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeStorage")
            .field("name", &self.name)
            .field("symbols", &self.symbols)
            .field("children", &self.children)
            .field("image_reads", &self.image_reads)
            .field("image_writes", &self.image_writes)
            .field("external_images", &self.external_images)
            .field("external_buffers", &self.external_buffers)
            .field("swapchain_requests", &self.swapchain_requests)
            .finish()
    }
}

// NodeStorage no longer implements the old Node trait as RenderPass replaces it.

/// Implementation of the internal PassBuilder trait.
pub struct PassRecorder<'a> {
    storage: &'a mut NodeStorage,
    ancestor_symbols: Vec<&'a SymbolTable>,
    is_setup: bool,
}

impl<'a> InternalPassBuilder for PassRecorder<'a> {
    fn publish_erased(&mut self, _type_id: TypeId, name: &str, data: Box<dyn Any + Send + Sync>) {
        tracing::trace!(name, "Publishing CPU data");
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::CpuData(_type_id),
                lifetime: SymbolLifetime::Transient,
                data: Some(data),
            },
            id,
        );
        // Track as a data dependency for the DAG
        self.storage.data_writes.push(name.to_string());
    }

    fn consume_erased(&mut self, type_id: TypeId, name: &str) -> &dyn Any {
        self.try_consume_erased(type_id, name)
            .unwrap_or_else(|| panic!("Symbol '{}' not found in current or parent scope", name))
    }

    fn try_consume_erased(&mut self, _type_id: TypeId, name: &str) -> Option<&dyn Any> {
        // Track as a data dependency for the DAG
        self.storage.data_reads.push(name.to_string());

        if let Some(id) = self.storage.symbols.resolve(name) {
            tracing::trace!(name, "Consuming CPU data (local)");
            return self.storage.symbols.get_data(id);
        }

        for parent in self.ancestor_symbols.iter().rev() {
            if let Some(id) = parent.resolve(name) {
                tracing::trace!(name, "Consuming CPU data (inherited)");
                return parent.get_data(id);
            }
        }

        None
    }

    fn read_image(&mut self, handle: ImageHandle, usage: ResourceUsage) {
        self.storage.image_reads.push((handle, usage));
    }

    fn write_image(&mut self, handle: ImageHandle, usage: ResourceUsage) {
        self.storage.image_writes.push((handle, usage));
    }

    fn declare_present_image(&mut self, handle: ImageHandle) {
        // Register in present_images for post-transition barrier generation.
        self.storage.present_images.push(handle);
        // Also register as a read dependency so the DAG correctly orders this pass
        // after any pass that writes to the image. PRESENT is filtered out of the
        // normal scratch loop in simulate_pass (handled by the post_transitions block).
        self.storage
            .image_reads
            .push((handle, ResourceUsage::PRESENT));
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

    fn register_external_buffer(&mut self, handle: BufferHandle, physical: BackendBuffer) {
        self.storage.external_buffers.push((handle, physical));
    }

    fn read_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage) {
        self.storage.buffer_reads.push((handle, usage));
    }

    fn write_buffer(&mut self, handle: BufferHandle, usage: ResourceUsage) {
        self.storage.buffer_writes.push((handle, usage));
    }

    fn declare_image(&mut self, name: &str, desc: ImageDesc) -> ImageHandle {
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::Image(desc),
                lifetime: SymbolLifetime::Transient,
                data: None, // Will update below
            },
            id,
        );
        let actual_handle = ImageHandle(id);
        self.storage.symbols.symbols[index as usize].data = Some(Box::new(actual_handle));
        actual_handle
    }

    fn declare_buffer(&mut self, name: &str, desc: BufferDesc) -> BufferHandle {
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::Buffer(desc),
                lifetime: SymbolLifetime::Transient,
                data: None,
            },
            id,
        );
        let actual_handle = BufferHandle(id);
        self.storage.symbols.symbols[index as usize].data = Some(Box::new(actual_handle));
        actual_handle
    }

    fn declare_buffer_history(&mut self, name: &str, desc: BufferDesc) -> BufferHandle {
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::Buffer(desc),
                lifetime: SymbolLifetime::TemporalHistory,
                data: None,
            },
            id,
        );
        let actual_handle = BufferHandle(id);
        self.storage.symbols.symbols[index as usize].data = Some(Box::new(actual_handle));
        actual_handle
    }

    fn read_buffer_history(&mut self, name: &str) -> BufferHandle {
        let history_name = format!("{}_History", name);
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        self.storage.symbols.publish(
            &history_name,
            Symbol {
                name: history_name.clone(),
                symbol_type: SymbolType::Buffer(BufferDesc {
                    size: 0,
                    usage: crate::graph::types::BufferUsageFlags::empty(),
                    memory: crate::graph::types::MemoryType::GpuOnly,
                }), // Size ignored since it refers to an existing external-like buffer
                lifetime: SymbolLifetime::TemporalHistory,
                data: None,
            },
            id,
        );
        let actual_handle = BufferHandle(id);
        self.storage.symbols.symbols[index as usize].data = Some(Box::new(actual_handle));
        actual_handle
    }

    fn import_buffer(&mut self, name: &str, physical: BackendBuffer) -> BufferHandle {
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        self.storage.symbols.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::Buffer(BufferDesc {
                    size: 0,
                    usage: crate::graph::types::BufferUsageFlags::empty(),
                    memory: crate::graph::types::MemoryType::GpuOnly,
                }), // Size/Usage ignored for external buffers
                lifetime: SymbolLifetime::External,
                data: None,
            },
            id,
        );
        let actual_handle = BufferHandle(id);
        self.storage.symbols.symbols[index as usize].data = Some(Box::new(actual_handle));
        self.register_external_buffer(actual_handle, physical);
        actual_handle
    }

    fn acquire_backbuffer(&mut self, window: WindowHandle) -> ImageHandle {
        let name = format!("Window_{}", window.0);
        let index = self.storage.symbols.symbols.len() as u64;
        let id = SymbolId((self.storage.node_id << 32) | index);
        self.storage.symbols.publish(
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
                    clear_value: None,
                }),
                lifetime: SymbolLifetime::External,
                data: None,
            },
            id,
        );
        let actual_handle = ImageHandle(id);
        self.storage.symbols.symbols[index as usize].data = Some(Box::new(actual_handle));

        // Record the request
        self.storage
            .swapchain_requests
            .push((actual_handle, window));

        actual_handle
    }

    fn add_node_erased(&mut self, node: Box<dyn RenderPass>) {
        tracing::trace!(name = node.name(), "Adding sub-node");

        let prefer_async = node.prefer_async();
        let mut child_storage = NodeStorage {
            node_id: NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
            name: node.name().to_string(),
            symbols: SymbolTable::new(),
            children: Vec::new(),
            pass: Some(node),

            pipeline: None,
            image_reads: Vec::new(),
            image_writes: Vec::new(),
            buffer_reads: Vec::new(),
            buffer_writes: Vec::new(),
            data_writes: Vec::new(),
            data_reads: Vec::new(),
            external_images: Vec::new(),
            external_buffers: Vec::new(),
            swapchain_requests: Vec::new(),
            descriptor_sets: Vec::new(),
            prefer_async,
            present_images: Vec::new(),
        };

        // Put it back but we need to record first
        let mut pass = child_storage.pass.take().unwrap();

        {
            let mut ancestors = self.ancestor_symbols.clone();
            ancestors.push(&self.storage.symbols);

            let mut sub_recorder = PassRecorder {
                storage: &mut child_storage,
                ancestor_symbols: ancestors,
                is_setup: self.is_setup,
            };
            let mut builder = PassBuilder {
                inner: &mut sub_recorder,
            };
            pass.declare(&mut builder);
        }

        // Put the real pass back
        child_storage.pass = Some(pass);
        self.storage.children.push(child_storage);
    }

    fn is_setup(&self) -> bool {
        self.is_setup
    }
}

/// Root of the Frame Graph recording.
pub struct FrameGraph {
    /// Persistent global symbols (AssetLoader, Services).
    pub globals: SymbolTable,
    /// Tree root.
    root: NodeStorage,
    /// Explicitly declared graph outputs (Present, Readback, etc.)
    pub outputs: HashMap<SymbolId, OutputKind>,
}

impl FrameGraph {
    pub fn new() -> Self {
        Self {
            globals: SymbolTable::new(),
            root: NodeStorage {
                node_id: 0,
                name: "root".to_string(),
                symbols: SymbolTable::new(),
                children: Vec::new(),
                pass: None,

                pipeline: None,
                image_reads: Vec::new(),
                image_writes: Vec::new(),
                buffer_reads: Vec::new(),
                buffer_writes: Vec::new(),
                data_writes: Vec::new(),
                data_reads: Vec::new(),
                external_images: Vec::new(),
                external_buffers: Vec::new(),
                swapchain_requests: Vec::new(),
                descriptor_sets: Vec::new(),
                prefer_async: false,
                present_images: Vec::new(),
            },
            outputs: HashMap::new(),
        }
    }

    /// Explicitly mark a resource as a final product of the graph.
    /// This drives Dead Pass Culling (DCE) and final layout transitions.
    pub fn mark_output<H: Into<SymbolId>>(&mut self, handle: H, kind: OutputKind) {
        self.outputs.insert(handle.into(), kind);
    }

    pub fn declare<F>(&mut self, setup: F)
    where
        F: FnOnce(&mut PassBuilder),
    {
        let mut recorder = PassRecorder {
            storage: &mut self.root,
            ancestor_symbols: vec![&self.globals],
            is_setup: false,
        };

        let mut builder = PassBuilder {
            inner: &mut recorder,
        };
        setup(&mut builder);
    }

    /// Records the persistent parts of the graph (long-lived passes).
    pub fn setup<F>(&mut self, f: F)
    where
        F: FnOnce(&mut PassBuilder),
    {
        let mut recorder = PassRecorder {
            storage: &mut self.root,
            ancestor_symbols: vec![&self.globals],
            is_setup: true,
        };
        let mut builder = PassBuilder::new(&mut recorder);
        f(&mut builder);
    }

    /// Registers a long-lived service in the global scope.
    pub fn publish<T: 'static + Send + Sync>(&mut self, name: &str, data: T) {
        let index = self.globals.symbols.len() as u64;
        let id = SymbolId((0xFFFF_FFFFu64 << 32) | index); // Use 0xFFFFFFFF as global node tag
        self.globals.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::CpuData(TypeId::of::<T>()),
                lifetime: SymbolLifetime::Persistent,
                data: Some(Box::new(data)),
            },
            id,
        );
    }

    /// Resolve a typed symbol from the global blackboard. Panics if not found.
    pub fn consume<T: 'static + Send + Sync>(&self, name: &str) -> &T {
        let id = self
            .globals
            .resolve(name)
            .unwrap_or_else(|| panic!("Global symbol '{}' not found", name));
        self.globals
            .get_data(id)
            .and_then(|any| any.downcast_ref::<T>())
            .unwrap_or_else(|| panic!("Type mismatch for global symbol '{}'", name))
    }

    /// Resolve a typed symbol from the global blackboard. Returns None if not found.
    pub fn try_consume<T: 'static + Send + Sync>(&self, name: &str) -> Option<&T> {
        let id = self.globals.resolve(name)?;
        self.globals.get_data(id)?.downcast_ref::<T>()
    }

    /// Initializes all registered passes using the Global Scope.
    /// Should be called once after setup.
    pub fn init_all(&mut self, backend: &mut dyn RenderBackend) {
        tracing::debug!("Initializing FrameGraph global scope and passes");
        Self::init_recursive(&mut self.root, backend, &self.globals);
    }

    fn init_recursive(
        node: &mut NodeStorage,
        backend: &mut dyn RenderBackend,
        globals: &SymbolTable,
    ) {
        if let Some(mut pass) = node.pass.take() {
            {
                let mut recorder = PassRecorder {
                    storage: node,
                    ancestor_symbols: vec![globals],
                    is_setup: true,
                };
                let mut builder = PassBuilder::new(&mut recorder);
                pass.init(backend, &mut builder);
            }
            node.pass = Some(pass);
        }

        for child in &mut node.children {
            Self::init_recursive(child, backend, globals);
        }
    }

    pub fn compile(self, capabilities: &DeviceCapabilities) -> CompiledGraph {
        tracing::debug!("Compiling hierarchical frame graph");

        // 1. Flatten the tree into a linear pass list
        let mut flat_passes = Vec::new();
        Self::flatten_recursive(&self.root, &mut flat_passes);

        // 2. Assign Queues
        Self::assign_queues(&mut flat_passes, capabilities);

        // 3. Build dependency DAG from resource read/write overlaps
        let adj = Self::build_dependency_dag(&flat_passes);

        // 3.1. Detect and annotate cross-queue ownership transfers
        Self::detect_cross_queue_transfers(&mut flat_passes, &adj);

        // 4. Topological sort with level assignment (Kahn's algorithm)
        let (order, levels) = Self::topological_sort_levels(&flat_passes, &adj);

        let max_level = levels.iter().copied().max().unwrap_or(0);

        // 5. Group passes by level into ExecutionSteps
        let mut steps = Vec::new();
        let mut last_signaled_value = HashMap::new(); // Queue -> Value
        let mut next_sync_value = HashMap::new(); // Queue -> Value

        for level in 0..=max_level {
            let passes_in_level: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&idx| levels[idx] == level)
                .collect();

            if passes_in_level.is_empty() {
                continue;
            }

            // At the beginning of each level, insert necessary waits
            let mut pending_waits = Vec::new();
            for &pass_idx in &passes_in_level {
                let pass = &flat_passes[pass_idx];
                // For each predecessor i of pass_idx
                for i in 0..pass_idx {
                    if Self::has_dependency(&flat_passes[i], pass) {
                        let src_queue = flat_passes[i].queue;
                        let dst_queue = pass.queue;
                        if src_queue != dst_queue {
                            let value = *next_sync_value.get(&src_queue).unwrap_or(&0);
                            if value
                                > *last_signaled_value
                                    .get(&(src_queue, dst_queue))
                                    .unwrap_or(&0)
                            {
                                pending_waits.push(ExecutionStep::Wait {
                                    queue: dst_queue,
                                    on: src_queue,
                                    value,
                                });
                                last_signaled_value.insert((src_queue, dst_queue), value);
                            }
                        }
                    }
                }
            }
            // Dedup waits
            pending_waits.sort_by_key(|w| match w {
                ExecutionStep::Wait { queue, on, value } => (*queue, *on, *value),
                _ => unreachable!(),
            });
            pending_waits.dedup_by(|a, b| match (a, b) {
                (
                    ExecutionStep::Wait {
                        queue: q1,
                        on: o1,
                        value: v1,
                    },
                    ExecutionStep::Wait {
                        queue: q2,
                        on: o2,
                        value: v2,
                    },
                ) => q1 == q2 && o1 == o2 && v1 == v2,
                _ => false,
            });
            steps.extend(pending_waits);

            // Execute passes in this level
            match passes_in_level.len() {
                1 => {
                    steps.push(ExecutionStep::Barriers(vec![passes_in_level[0]]));
                    steps.push(ExecutionStep::Execute(passes_in_level[0]));
                }
                _ => {
                    steps.push(ExecutionStep::Barriers(passes_in_level.clone()));
                    steps.push(ExecutionStep::ExecuteParallel(passes_in_level.clone()));
                }
            }

            // At the end of each level, if a queue executed something that others might depend on, signal it
            let mut queues_in_level = std::collections::HashSet::new();
            for &pass_idx in &passes_in_level {
                queues_in_level.insert(flat_passes[pass_idx].queue);
            }

            for queue in queues_in_level {
                let value = next_sync_value.entry(queue).or_insert(0);
                *value += 1;
                steps.push(ExecutionStep::Signal {
                    queue,
                    value: *value,
                });
            }
        }

        tracing::debug!(
            passes = flat_passes.len(),
            levels = max_level + 1,
            steps = steps.len(),
            "Compiled graph: {} passes, {} levels, {} steps",
            flat_passes.len(),
            max_level + 1,
            steps.len(),
        );

        CompiledGraph {
            _root: self.root,
            flat_passes,
            steps,
        }
    }

    fn assign_queues(passes: &mut [FlatPass], capabilities: &DeviceCapabilities) {
        tracing::debug!(
            async_compute = capabilities.async_compute,
            async_transfer = capabilities.async_transfer,
            "Assigning queues"
        );
        for pass in passes {
            pass.queue = match pass.domain {
                PassDomain::Graphics => QueueType::Graphics,
                PassDomain::Compute => {
                    if pass.prefer_async && capabilities.async_compute {
                        QueueType::AsyncCompute
                    } else {
                        QueueType::Graphics
                    }
                }
                PassDomain::Transfer => {
                    if pass.prefer_async && capabilities.async_transfer {
                        QueueType::Transfer
                    } else {
                        QueueType::Graphics
                    }
                }
                PassDomain::Cpu => QueueType::Graphics,
            };
            tracing::debug!(
                pass = %pass.name,
                domain = ?pass.domain,
                prefer_async = pass.prefer_async,
                queue = ?pass.queue,
                "Pass assigned"
            );
        }
    }

    /// Recursively flatten the node tree into leaf passes.
    /// Groups contribute their children but are not themselves execution units.
    fn flatten_recursive(node: &NodeStorage, flat_passes: &mut Vec<FlatPass>) {
        // A leaf node: has resource intents, data deps, or a pipeline
        let is_leaf = !node.image_reads.is_empty()
            || !node.image_writes.is_empty()
            || !node.buffer_reads.is_empty()
            || !node.buffer_writes.is_empty()
            || !node.data_reads.is_empty()
            || !node.data_writes.is_empty()
            || node.pipeline.is_some()
            || node.name == "root"; // root is never a leaf

        if node.name != "root" && (is_leaf || node.children.is_empty()) {
            let domain = FlatPass::infer_domain_from_intents(
                &node.image_reads,
                &node.image_writes,
                &node.buffer_reads,
                &node.buffer_writes,
                node.pipeline.is_some(),
            );
            tracing::trace!(pass = %node.name, ?domain, "Flattened pass");
            flat_passes.push(FlatPass {
                node_id: node.node_id,
                name: node.name.clone(),
                domain,
                pipeline: node.pipeline,
                image_reads: node.image_reads.clone(),
                image_writes: node.image_writes.clone(),
                buffer_reads: node.buffer_reads.clone(),
                buffer_writes: node.buffer_writes.clone(),
                data_reads: node.data_reads.clone(),
                data_writes: node.data_writes.clone(),
                prefer_async: node.prefer_async,
                queue: QueueType::Graphics, // Assigned during compilation
                releases: Vec::new(),
                acquires: Vec::new(),
                present_images: node.present_images.clone(),
            });
        }

        for child in &node.children {
            Self::flatten_recursive(child, flat_passes);
        }
    }

    /// Build a dependency DAG. For each pair (i, j) where j > i in declaration order:
    /// - RAW: j reads what i writes
    /// - WAR: j writes what i reads
    /// - WAW: j writes what i writes
    fn build_dependency_dag(passes: &[FlatPass]) -> Vec<Vec<usize>> {
        let n = passes.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for j in 0..n {
            for i in 0..j {
                if Self::has_dependency(&passes[i], &passes[j]) {
                    adj[i].push(j);
                }
            }
        }

        adj
    }

    /// Detect resources that cross queue boundaries and annotate the passes with Release/Acquire transfers.
    fn detect_cross_queue_transfers(passes: &mut [FlatPass], adj: &[Vec<usize>]) {
        let n = passes.len();
        for i in 0..n {
            for j_idx in 0..adj[i].len() {
                let j = adj[i][j_idx];
                let src_queue = passes[i].queue;
                let dst_queue = passes[j].queue;

                if src_queue != dst_queue {
                    // Collect all images causing a dependency between i and j
                    let mut shared_images = Vec::new();

                    // RAW: j reads what i writes
                    for (h, usage_i) in &passes[i].image_writes {
                        if let Some((_, usage_j)) =
                            passes[j].image_reads.iter().find(|(rh, _)| rh.0 == h.0)
                        {
                            shared_images.push((*h, *usage_i, *usage_j));
                        }
                    }
                    // WAW: j writes what i writes
                    for (h, usage_i) in &passes[i].image_writes {
                        if let Some((_, usage_j)) =
                            passes[j].image_writes.iter().find(|(wh, _)| wh.0 == h.0)
                        {
                            shared_images.push((*h, *usage_i, *usage_j));
                        }
                    }
                    // WAR: j writes what i reads
                    for (h, usage_i) in &passes[i].image_reads {
                        if let Some((_, usage_j)) =
                            passes[j].image_writes.iter().find(|(wh, _)| wh.0 == h.0)
                        {
                            shared_images.push((*h, *usage_i, *usage_j));
                        }
                    }

                    for (h, ui, uj) in shared_images {
                        let transfer = CrossQueueTransfer {
                            image: Some(h),
                            buffer: None,
                            src_queue,
                            dst_queue,
                            src_usage: ui,
                            dst_usage: uj,
                        };

                        // Add to source pass if not already there
                        if !passes[i]
                            .releases
                            .iter()
                            .any(|r| r.image == Some(h) && r.dst_queue == dst_queue)
                        {
                            passes[i].releases.push(transfer);
                        }
                        // Add to destination pass if not already there
                        if !passes[j]
                            .acquires
                            .iter()
                            .any(|r| r.image == Some(h) && r.src_queue == src_queue)
                        {
                            passes[j].acquires.push(transfer);
                        }
                    }
                }
            }
        }
    }

    /// Check if pass `b` depends on pass `a` (a must execute before b).
    fn has_dependency(a: &FlatPass, b: &FlatPass) -> bool {
        // RAW: b reads an image/buffer that a writes
        for (h, _) in &a.image_writes {
            if b.image_reads.iter().any(|(rh, _)| rh.0 == h.0) {
                return true;
            }
        }
        for (h, _) in &a.buffer_writes {
            if b.buffer_reads.iter().any(|(rh, _)| rh.0 == h.0) {
                return true;
            }
        }
        // WAW: b writes an image/buffer that a writes
        for (h, _) in &a.image_writes {
            if b.image_writes.iter().any(|(wh, _)| wh.0 == h.0) {
                return true;
            }
        }
        for (h, _) in &a.buffer_writes {
            if b.buffer_writes.iter().any(|(wh, _)| wh.0 == h.0) {
                return true;
            }
        }
        // WAR: b writes an image/buffer that a reads
        for (h, _) in &a.image_reads {
            if b.image_writes.iter().any(|(wh, _)| wh.0 == h.0) {
                return true;
            }
        }
        for (h, _) in &a.buffer_reads {
            if b.buffer_writes.iter().any(|(wh, _)| wh.0 == h.0) {
                return true;
            }
        }
        // CPU data: b reads data that a writes (RAW)
        for name in &a.data_writes {
            if b.data_reads.iter().any(|rn| rn == name) {
                return true;
            }
        }
        // CPU data: b writes data that a reads (WAR)
        for name in &a.data_reads {
            if b.data_writes.iter().any(|wn| wn == name) {
                return true;
            }
        }
        // CPU data: b writes data that a writes (WAW)
        for name in &a.data_writes {
            if b.data_writes.iter().any(|wn| wn == name) {
                return true;
            }
        }
        false
    }

    /// Topological sort with level assignment using Kahn's algorithm.
    /// Returns (execution_order, per_pass_level).
    fn topological_sort_levels(
        passes: &[FlatPass],
        adj: &[Vec<usize>],
    ) -> (Vec<usize>, Vec<usize>) {
        let n = passes.len();
        let mut in_degree = vec![0usize; n];
        for edges in adj {
            for &to in edges {
                in_degree[to] += 1;
            }
        }

        // Seed with zero-indegree nodes
        let mut queue: std::collections::VecDeque<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, d)| **d == 0)
            .map(|(i, _)| i)
            .collect();

        let mut order = Vec::with_capacity(n);
        let mut levels = vec![0usize; n];

        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &next in &adj[node] {
                in_degree[next] -= 1;
                // Level = max(level of all predecessors) + 1
                levels[next] = levels[next].max(levels[node] + 1);
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() != n {
            tracing::error!(
                expected = n,
                got = order.len(),
                "Cycle detected in dependency graph! Some passes will not execute."
            );
        }

        for i in &order {
            tracing::trace!(
                pass = %passes[*i].name,
                level = levels[*i],
                "Pass scheduled"
            );
        }

        (order, levels)
    }
}

/// A discrete step in the compiled execution plan.
#[derive(Debug)]
enum ExecutionStep {
    /// Emit barriers for a batch of passes.
    Barriers(Vec<usize>),
    /// Execute a single pass.
    Execute(usize),
    /// Execute multiple independent passes (parallel-ready, sequential for now).
    ExecuteParallel(Vec<usize>),
    /// Signal a timeline value on a queue.
    Signal { queue: QueueType, value: u64 },
    /// Wait for a timeline value from another queue.
    Wait {
        queue: QueueType,
        on: QueueType,
        value: u64,
    },
}

pub struct CompiledGraph {
    _root: NodeStorage,
    flat_passes: Vec<FlatPass>,
    steps: Vec<ExecutionStep>,
}

impl CompiledGraph {
    pub fn execute<B: RenderBackendInternal>(
        mut self,
        backend: &mut B,
        temporal_registry: Option<&mut crate::graph::temporal::TemporalRegistry>,
    ) -> Result<Option<u64>, GraphError> {
        tracing::debug!(
            passes = self.flat_passes.len(),
            steps = self.steps.len(),
            "Executing compiled frame graph"
        );

        // 0. Clear per-frame virtual→physical maps before any resources are registered.
        // Must happen before resolve_resources_recursive so transient images are included.
        backend.reset_frame_resources();

        // 1. Resource Resolution & Allocation (still tree-based)
        let mut transient_images = Vec::new();
        let mut transient_buffers = Vec::new();
        Self::resolve_resources_recursive(
            &mut self._root,
            backend,
            temporal_registry,
            &mut transient_images,
            &mut transient_buffers,
        );

        // 2. Begin frame (resets per-frame state, waits on timeline)
        backend.begin_frame();

        // 3. Swapchain acquire + external resource registration (must be after begin_frame)
        // IMPORTANT: must happen before analyze_frame so swapchain images are in external_to_physical
        let mut inactive_images: Vec<u64> = Vec::with_capacity(2);
        Self::process_externals_recursive(&mut self._root, backend, &mut inactive_images)?;

        // 3.5 Analyze frame synchronization plan — all resources now registered
        backend.analyze_frame(&self.flat_passes);

        // 4. Build node_id → NodeStorage SyncPtr map for O(1) lookup
        let mut node_map: HashMap<u64, SyncPtr<NodeStorage>> = HashMap::new();
        Self::collect_node_map_sync(&mut self._root, &mut node_map);

        // 4.5 Prepare all active passes sequentially
        let mut prepared_passes: Vec<Option<B::PreparedPass>> =
            Vec::with_capacity(self.flat_passes.len());
        for (idx, flat) in self.flat_passes.iter().enumerate() {
            let is_skipped = !inactive_images.is_empty()
                && flat
                    .image_writes
                    .iter()
                    .any(|(h, _)| inactive_images.contains(&h.0.0));

            if is_skipped {
                tracing::debug!(
                    pass = %flat.name,
                    "Skipping pass preparation (targets inactive image)"
                );
                prepared_passes.push(None);
                continue;
            }

            if let Some(node_ptr) = node_map.get(&flat.node_id) {
                let node = unsafe { &mut *node_ptr.0 };
                let desc = PassDescriptor {
                    name: &node.name,
                    pipeline: flat.pipeline,
                    image_reads: &node.image_reads,
                    image_writes: &node.image_writes,
                    buffer_reads: &node.buffer_reads,
                    buffer_writes: &node.buffer_writes,
                    descriptor_sets: &node.descriptor_sets,
                    queue: flat.queue,
                    releases: &flat.releases,
                    acquires: &flat.acquires,
                };
                prepared_passes.push(Some(backend.prepare_pass(idx, desc)));
            } else {
                tracing::error!(
                    pass = %flat.name,
                    node_id = flat.node_id,
                    "Node not found in tree during preparation!"
                );
                prepared_passes.push(None);
            }
        }

        // 5. Execute steps — build an ORDERED batch so submit() can split
        //    submissions at Signal/Wait boundaries, preventing multi-queue deadlocks.
        let mut batch = CommandBatch::default();

        let push_cb = |batch: &mut CommandBatch, queue: QueueType, cb: BackendCommandBuffer| {
            batch.steps.push(BatchStep::Command { queue, cb });
        };

        for step in &self.steps {
            match step {
                ExecutionStep::Signal { queue, value } => {
                    batch.steps.push(BatchStep::Signal {
                        queue: *queue,
                        value: *value,
                    });
                }
                ExecutionStep::Wait { queue, on, value } => {
                    batch.steps.push(BatchStep::Wait {
                        queue: *queue,
                        on: *on,
                        value: *value,
                    });
                }
                ExecutionStep::Barriers(pass_indices) => {
                    let mut graphics_refs = Vec::new();
                    let mut compute_refs = Vec::new();
                    let mut transfer_refs = Vec::new();

                    for &idx in pass_indices {
                        if let Some(prepared) = &prepared_passes[idx] {
                            match backend.get_prepared_pass_queue(prepared) {
                                QueueType::Graphics => graphics_refs.push(prepared),
                                QueueType::AsyncCompute => compute_refs.push(prepared),
                                QueueType::Transfer => transfer_refs.push(prepared),
                            }
                        }
                    }

                    if !graphics_refs.is_empty() {
                        if let Some(cb) = backend.record_barriers(&graphics_refs) {
                            push_cb(&mut batch, QueueType::Graphics, cb);
                        }
                    }
                    if !compute_refs.is_empty() {
                        if let Some(cb) = backend.record_barriers(&compute_refs) {
                            push_cb(&mut batch, QueueType::AsyncCompute, cb);
                        }
                    }
                    if !transfer_refs.is_empty() {
                        if let Some(cb) = backend.record_barriers(&transfer_refs) {
                            push_cb(&mut batch, QueueType::Transfer, cb);
                        }
                    }
                }
                ExecutionStep::Execute(pass_idx) => {
                    if let Some(prepared) = &prepared_passes[*pass_idx] {
                        let flat = &self.flat_passes[*pass_idx];
                        if let Some(node_ptr) = node_map.get(&flat.node_id) {
                            let node = unsafe { &mut *node_ptr.0 };
                            tracing::debug!(pass = %flat.name, domain = ?flat.domain, queue = ?flat.queue, "Executing pass");
                            let (_sem, cb, _present_req) =
                                backend.record_pass(prepared, node.pass.as_ref().unwrap().as_ref());
                            if let Some(c) = cb {
                                push_cb(&mut batch, flat.queue, c);
                            }
                        }
                    }
                }
                ExecutionStep::ExecuteParallel(pass_indices) => {
                    let parallel_results: Vec<_> = pass_indices
                        .par_iter()
                        .filter_map(|&pass_idx| {
                            if let Some(prepared) = &prepared_passes[pass_idx] {
                                let flat = &self.flat_passes[pass_idx];
                                if let Some(node_ptr) = node_map.get(&flat.node_id) {
                                    let node = unsafe { &mut *node_ptr.0 };
                                    let (_sem, cb, present_req) = backend.record_pass(
                                        prepared,
                                        node.pass.as_ref().unwrap().as_ref(),
                                    );
                                    return Some((cb, present_req, flat.queue));
                                }
                            }
                            None
                        })
                        .collect();

                    for (cb, _present_req, queue) in parallel_results {
                        if let Some(c) = cb {
                            push_cb(&mut batch, queue, c);
                        }
                    }
                }
            }
        }

        // 6. Final submission
        let submit_res = backend
            .submit(batch)
            .map_err(|e| GraphError::BackendError(e))?;

        backend.end_frame();

        // 7. Release transient resources
        for image in transient_images {
            backend.release_transient_image(image);
        }
        for buffer in transient_buffers {
            backend.release_transient_buffer(buffer);
        }

        Ok(Some(submit_res))
    }

    /// Collect all nodes into a map by node_id for O(1) lookup during execution.
    fn collect_node_map_sync(node: &mut NodeStorage, map: &mut HashMap<u64, SyncPtr<NodeStorage>>) {
        map.insert(node.node_id, SyncPtr(node as *mut NodeStorage));
        for child in &mut node.children {
            Self::collect_node_map_sync(child, map);
        }
    }

    /// Process swapchain requests and external resources from the tree.
    fn process_externals_recursive<B: RenderBackendInternal>(
        node: &mut NodeStorage,
        backend: &mut B,
        inactive_images: &mut Vec<u64>,
    ) -> Result<(), GraphError> {
        for (handle, window) in &node.swapchain_requests {
            match backend
                .acquire_swapchain_image(*window)
                .map_err(|e| GraphError::BackendError(e))?
            {
                Some((physical, _sem, _idx)) => {
                    backend.register_external_image(*handle, physical);
                }
                None => {
                    tracing::debug!(
                        window = ?window.0,
                        "Window is minimized, skipping associated passes"
                    );
                    inactive_images.push(handle.0.0);
                }
            }
        }

        for (virtual_handle, physical) in &node.external_images {
            backend.register_external_image(*virtual_handle, *physical);
            // We do not set the debug name here because external resources
            // should already be named where they were created.
        }
        for (virtual_handle, physical) in &node.external_buffers {
            backend.register_external_buffer(*virtual_handle, *physical);
            // We do not set the debug name here because external resources
            // should already be named where they were created.
        }

        for child in &mut node.children {
            Self::process_externals_recursive(child, backend, inactive_images)?;
        }
        Ok(())
    }

    fn resolve_resources_recursive<B: RenderBackendInternal>(
        node: &mut NodeStorage,
        backend: &mut B,
        mut temporal_registry_opt: Option<&mut crate::graph::temporal::TemporalRegistry>,
        transient_images: &mut Vec<BackendImage>,
        transient_buffers: &mut Vec<BackendBuffer>,
    ) {
        // Call to pass.init removed here, now handled by init_all() once.

        // Resolve symbols in current scope
        for symbol in &node.symbols.symbols {
            match symbol.symbol_type {
                SymbolType::Image(ref desc) => {
                    if symbol.lifetime == SymbolLifetime::Transient {
                        let physical = backend.create_transient_image(desc);
                        transient_images.push(physical);
                        let handle = *symbol
                            .data
                            .as_ref()
                            .expect("Image without handle")
                            .downcast_ref::<ImageHandle>()
                            .expect("Not a handle");
                        #[cfg(debug_assertions)]
                        backend.set_image_name(physical, &symbol.name);
                        backend.register_external_image(handle, physical);
                    }
                }
                SymbolType::Buffer(ref desc) => {
                    if symbol.lifetime == SymbolLifetime::Transient {
                        let physical = backend.create_transient_buffer(desc);
                        transient_buffers.push(physical);
                        let handle = *symbol
                            .data
                            .as_ref()
                            .expect("Buffer without handle")
                            .downcast_ref::<BufferHandle>()
                            .expect("Not a handle");
                        #[cfg(debug_assertions)]
                        backend.set_buffer_name(physical, &symbol.name);
                        backend.register_external_buffer(handle, physical);
                    } else if symbol.lifetime == SymbolLifetime::TemporalHistory {
                        if let Some(ref mut temporal) = temporal_registry_opt {
                            let handle = *symbol
                                .data
                                .as_ref()
                                .expect("History Buffer without handle")
                                .downcast_ref::<BufferHandle>()
                                .expect("Not a handle");

                            let physical = if symbol.name.ends_with("_History") {
                                // Extract original name by stripping the _History suffix
                                let base_name = &symbol.name[..symbol.name.len() - 8];
                                temporal.get_or_create_history_buffer(base_name, desc, backend)
                            } else {
                                temporal.get_or_create_buffer(&symbol.name, desc, backend)
                            };

                            backend.register_external_buffer(handle, physical);
                        } else {
                            tracing::warn!(
                                "TemporalHistory symbol '{}' declared but no TemporalRegistry provided!",
                                symbol.name
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        // Recurse children
        for child in &mut node.children {
            let tr_opt = temporal_registry_opt.as_deref_mut();
            Self::resolve_resources_recursive(
                child,
                backend,
                tr_opt,
                transient_images,
                transient_buffers,
            );
        }
    }
}
