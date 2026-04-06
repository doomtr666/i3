use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;

use crate::graph::backend::{DeviceCapabilities, RenderBackend};
use crate::graph::compiled::ExecutionStep;
use crate::graph::node::{NodeStorage, PassRecorder, NEXT_NODE_ID};
use crate::graph::pass::{PassBuilder, RenderPass};
use crate::graph::symbol_table::{Symbol, SymbolTable};
use crate::graph::types::*;

// Re-export so external paths like `i3_gfx::graph::compiler::FrameBlackboard` remain stable.
pub use crate::graph::compiled::CompiledGraph;
pub use crate::graph::symbol_table::FrameBlackboard;

use std::sync::atomic::Ordering;

// ─────────────────────────────────────────────────────────────────────────────
// FrameGraph — root of the frame-graph recording
// ─────────────────────────────────────────────────────────────────────────────

pub struct FrameGraph {
    /// Persistent global symbols (AssetLoader, services).
    pub globals: SymbolTable,
    root: NodeStorage,
    pub outputs: HashMap<SymbolId, OutputKind>,
}

impl FrameGraph {
    pub fn new() -> Self {
        Self {
            globals: SymbolTable::new(),
            root: NodeStorage::new(0, "root", false),
            outputs: HashMap::new(),
        }
    }

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
        };
        let mut builder = PassBuilder { inner: &mut recorder };
        setup(&mut builder);
    }

    /// Registers a long-lived service in the global scope.
    pub fn publish<T: 'static + Send + Sync>(&mut self, name: &str, data: T) {
        let index = self.globals.symbols.len() as u64;
        let id = SymbolId((0xFFFF_FFFFu64 << 32) | index);
        self.globals.publish(
            name,
            Symbol {
                name: name.to_string(),
                symbol_type: SymbolType::CpuData(TypeId::of::<T>()),
                lifetime: SymbolLifetime::Persistent,
                data: Some(Arc::new(data) as Arc<dyn std::any::Any + Send + Sync>),
                is_output: false,
            },
            id,
        );
    }

    /// Resolve a typed symbol from the global blackboard. Panics if not found.
    pub fn consume<T: 'static + Send + Sync>(&self, name: &str) -> &T {
        let id = self.globals.resolve(name)
            .unwrap_or_else(|| panic!("Global symbol '{}' not found", name));
        self.globals.get_data(id)
            .and_then(|any| any.downcast_ref::<T>())
            .unwrap_or_else(|| panic!("Type mismatch for global symbol '{}'", name))
    }

    /// Resolve a typed symbol from the global blackboard. Returns None if not found.
    pub fn try_consume<T: 'static + Send + Sync>(&self, name: &str) -> Option<&T> {
        let id = self.globals.resolve(name)?;
        self.globals.get_data(id)?.downcast_ref::<T>()
    }

    /// Initializes all registered passes using the global scope.
    pub fn init_all(&mut self, backend: &mut dyn RenderBackend) {
        tracing::debug!("Initializing FrameGraph passes");
        Self::init_recursive(&mut self.root, backend, &self.globals);
    }

    /// Initializes a single pass directly using the global scope, without running declare().
    pub fn init_pass_direct(
        &mut self,
        pass: &mut dyn RenderPass,
        backend: &mut dyn RenderBackend,
    ) {
        let mut dummy = NodeStorage::new(
            NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed),
            pass.name(),
            false,
        );
        let mut recorder = PassRecorder {
            storage: &mut dummy,
            ancestor_symbols: vec![&self.globals],
        };
        let mut builder = PassBuilder::new(&mut recorder);
        pass.init(backend, &mut builder);
    }

    fn init_recursive(node: &mut NodeStorage, backend: &mut dyn RenderBackend, globals: &SymbolTable) {
        if let Some(mut pass) = node.pass.take() {
            {
                let mut recorder = PassRecorder {
                    storage: node,
                    ancestor_symbols: vec![globals],
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

    // ── Compilation ──────────────────────────────────────────────────────────

    pub fn compile(self, capabilities: &DeviceCapabilities) -> CompiledGraph {
        tracing::debug!("Compiling frame graph");

        let mut flat_passes = Vec::new();
        Self::flatten_recursive(&self.root, &mut flat_passes);

        Self::assign_queues(&mut flat_passes, capabilities);

        let adj = Self::build_dependency_dag(&flat_passes);
        Self::detect_cross_queue_transfers(&mut flat_passes, &adj);

        let (order, levels) = Self::topological_sort_levels(&flat_passes, &adj);
        let max_level = levels.iter().copied().max().unwrap_or(0);

        let steps = Self::build_execution_steps(&flat_passes, &order, &levels, max_level);

        tracing::debug!(
            passes = flat_passes.len(),
            levels = max_level + 1,
            steps = steps.len(),
            "Compiled: {} passes, {} levels, {} steps",
            flat_passes.len(), max_level + 1, steps.len(),
        );

        CompiledGraph { _root: self.root, flat_passes, steps }
    }

    // ── Queue assignment ─────────────────────────────────────────────────────

    fn assign_queues(passes: &mut [FlatPass], capabilities: &DeviceCapabilities) {
        tracing::debug!(
            async_compute = capabilities.async_compute,
            async_transfer = capabilities.async_transfer,
            "Assigning queues"
        );
        for pass in passes.iter_mut() {
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
                pass = %pass.name, domain = ?pass.domain,
                prefer_async = pass.prefer_async, queue = ?pass.queue,
                "Pass assigned"
            );
        }
    }

    // ── Flatten tree → linear pass list ──────────────────────────────────────

    fn flatten_recursive(node: &NodeStorage, flat_passes: &mut Vec<FlatPass>) {
        let is_leaf = !node.image_reads.is_empty()
            || !node.image_writes.is_empty()
            || !node.buffer_reads.is_empty()
            || !node.buffer_writes.is_empty()
            || !node.data_reads.is_empty()
            || !node.data_writes.is_empty()
            || node.pipeline.is_some()
            || node.name == "root";

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
                queue: QueueType::Graphics,
                releases: Vec::new(),
                acquires: Vec::new(),
                present_images: node.present_images.clone(),
            });
        }

        for child in &node.children {
            Self::flatten_recursive(child, flat_passes);
        }
    }

    // ── Dependency DAG ───────────────────────────────────────────────────────

    fn build_dependency_dag(passes: &[FlatPass]) -> Vec<Vec<usize>> {
        let n = passes.len();
        let mut adj = vec![Vec::new(); n];
        for j in 0..n {
            for i in 0..j {
                if Self::has_dependency(&passes[i], &passes[j]) {
                    adj[i].push(j);
                }
            }
        }
        adj
    }

    fn detect_cross_queue_transfers(passes: &mut [FlatPass], adj: &[Vec<usize>]) {
        let n = passes.len();
        for i in 0..n {
            for j_idx in 0..adj[i].len() {
                let j = adj[i][j_idx];
                let src_queue = passes[i].queue;
                let dst_queue = passes[j].queue;
                if src_queue == dst_queue { continue; }

                let mut shared = Vec::new();
                for (h, ui) in &passes[i].image_writes {
                    if let Some((_, uj)) = passes[j].image_reads.iter().find(|(rh, _)| rh.0 == h.0) {
                        shared.push((*h, *ui, *uj));
                    }
                }
                for (h, ui) in &passes[i].image_writes {
                    if let Some((_, uj)) = passes[j].image_writes.iter().find(|(wh, _)| wh.0 == h.0) {
                        shared.push((*h, *ui, *uj));
                    }
                }
                for (h, ui) in &passes[i].image_reads {
                    if let Some((_, uj)) = passes[j].image_writes.iter().find(|(wh, _)| wh.0 == h.0) {
                        shared.push((*h, *ui, *uj));
                    }
                }

                for (h, ui, uj) in shared {
                    let xfer = CrossQueueTransfer {
                        image: Some(h),
                        buffer: None,
                        src_queue,
                        dst_queue,
                        src_usage: ui,
                        dst_usage: uj,
                    };
                    if !passes[i].releases.iter().any(|r| r.image == Some(h) && r.dst_queue == dst_queue) {
                        passes[i].releases.push(xfer);
                    }
                    if !passes[j].acquires.iter().any(|r| r.image == Some(h) && r.src_queue == src_queue) {
                        passes[j].acquires.push(xfer);
                    }
                }
            }
        }
    }

    fn has_dependency(a: &FlatPass, b: &FlatPass) -> bool {
        // RAW image
        for (h, _) in &a.image_writes {
            if b.image_reads.iter().any(|(rh, _)| rh.0 == h.0) { return true; }
        }
        // RAW buffer
        for (h, _) in &a.buffer_writes {
            if b.buffer_reads.iter().any(|(rh, _)| rh.0 == h.0) { return true; }
        }
        // WAW image
        for (h, _) in &a.image_writes {
            if b.image_writes.iter().any(|(wh, _)| wh.0 == h.0) { return true; }
        }
        // WAW buffer
        for (h, _) in &a.buffer_writes {
            if b.buffer_writes.iter().any(|(wh, _)| wh.0 == h.0) { return true; }
        }
        // WAR image
        for (h, _) in &a.image_reads {
            if b.image_writes.iter().any(|(wh, _)| wh.0 == h.0) { return true; }
        }
        // WAR buffer
        for (h, _) in &a.buffer_reads {
            if b.buffer_writes.iter().any(|(wh, _)| wh.0 == h.0) { return true; }
        }
        // CPU data RAW
        for name in &a.data_writes {
            if b.data_reads.iter().any(|rn| rn == name) { return true; }
        }
        // CPU data WAR
        for name in &a.data_reads {
            if b.data_writes.iter().any(|wn| wn == name) { return true; }
        }
        // CPU data WAW
        for name in &a.data_writes {
            if b.data_writes.iter().any(|wn| wn == name) { return true; }
        }
        false
    }

    // ── Topological sort (Kahn's algorithm) ──────────────────────────────────

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

        let mut queue: std::collections::VecDeque<usize> = in_degree
            .iter().enumerate()
            .filter(|(_, d)| **d == 0)
            .map(|(i, _)| i)
            .collect();

        let mut order = Vec::with_capacity(n);
        let mut levels = vec![0usize; n];

        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &next in &adj[node] {
                in_degree[next] -= 1;
                levels[next] = levels[next].max(levels[node] + 1);
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() != n {
            tracing::error!(expected = n, got = order.len(), "Cycle detected in dependency graph!");
        }

        for i in &order {
            tracing::trace!(pass = %passes[*i].name, level = levels[*i], "Scheduled");
        }

        (order, levels)
    }

    // ── Build execution steps ─────────────────────────────────────────────────

    fn build_execution_steps(
        flat_passes: &[FlatPass],
        order: &[usize],
        levels: &[usize],
        max_level: usize,
    ) -> Vec<ExecutionStep> {
        let mut steps = Vec::new();
        let mut last_signaled: HashMap<(QueueType, QueueType), u64> = HashMap::new();
        let mut next_sync_value: HashMap<QueueType, u64> = HashMap::new();

        for level in 0..=max_level {
            let passes_in_level: Vec<usize> = order.iter().copied()
                .filter(|&idx| levels[idx] == level)
                .collect();

            if passes_in_level.is_empty() { continue; }

            // Insert cross-queue waits.
            let mut pending_waits = Vec::new();
            for &pass_idx in &passes_in_level {
                let pass = &flat_passes[pass_idx];
                for i in 0..pass_idx {
                    if Self::has_dependency(&flat_passes[i], pass) {
                        let src_q = flat_passes[i].queue;
                        let dst_q = pass.queue;
                        if src_q != dst_q {
                            let value = *next_sync_value.get(&src_q).unwrap_or(&0);
                            if value > *last_signaled.get(&(src_q, dst_q)).unwrap_or(&0) {
                                pending_waits.push(ExecutionStep::Wait { queue: dst_q, on: src_q, value });
                                last_signaled.insert((src_q, dst_q), value);
                            }
                        }
                    }
                }
            }
            pending_waits.sort_by_key(|w| match w {
                ExecutionStep::Wait { queue, on, value } => (*queue, *on, *value),
                _ => unreachable!(),
            });
            pending_waits.dedup_by(|a, b| match (a, b) {
                (
                    ExecutionStep::Wait { queue: q1, on: o1, value: v1 },
                    ExecutionStep::Wait { queue: q2, on: o2, value: v2 },
                ) => q1 == q2 && o1 == o2 && v1 == v2,
                _ => false,
            });
            steps.extend(pending_waits);

            // Barriers + execute.
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

            // Signal after each level.
            let mut queues_in_level = std::collections::HashSet::new();
            for &pass_idx in &passes_in_level {
                queues_in_level.insert(flat_passes[pass_idx].queue);
            }
            for queue in queues_in_level {
                let value = next_sync_value.entry(queue).or_insert(0);
                *value += 1;
                steps.push(ExecutionStep::Signal { queue, value: *value });
            }
        }

        steps
    }
}
