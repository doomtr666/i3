use std::collections::HashMap;

use rayon::prelude::*;

use crate::graph::backend::{
    BackendBuffer, BackendImage, BackendCommandBuffer, BatchStep, CommandBatch,
    RenderBackendInternal,
};
use crate::graph::node::NodeStorage;
use crate::graph::symbol_table::FrameBlackboard;
use crate::graph::types::*;

// ─────────────────────────────────────────────────────────────────────────────
// SyncPtr — raw pointer wrapper that is Send + Sync for parallel execute
// ─────────────────────────────────────────────────────────────────────────────

pub(crate) struct SyncPtr<T>(pub(crate) *mut T);

impl<T> Clone for SyncPtr<T> {
    fn clone(&self) -> Self { *self }
}
impl<T> Copy for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

// ─────────────────────────────────────────────────────────────────────────────
// ExecutionStep
// ─────────────────────────────────────────────────────────────────────────────

/// A discrete step in the compiled execution plan.
#[derive(Debug)]
pub(crate) enum ExecutionStep {
    /// Emit barriers for a batch of passes.
    Barriers(Vec<usize>),
    /// Execute a single pass.
    Execute(usize),
    /// Execute multiple independent passes (may run in parallel).
    ExecuteParallel(Vec<usize>),
    /// Signal a timeline value on a queue.
    Signal { queue: QueueType, value: u64 },
    /// Wait for a timeline value from another queue.
    Wait { queue: QueueType, on: QueueType, value: u64 },
}

// ─────────────────────────────────────────────────────────────────────────────
// CompiledGraph
// ─────────────────────────────────────────────────────────────────────────────

pub struct CompiledGraph {
    pub(crate) _root: NodeStorage,
    pub(crate) flat_passes: Vec<FlatPass>,
    pub(crate) steps: Vec<ExecutionStep>,
}

impl CompiledGraph {
    pub fn execute<B: RenderBackendInternal>(
        &mut self,
        backend: &mut B,
        frame_data: &FrameBlackboard,
        temporal_registry: Option<&mut crate::graph::temporal::TemporalRegistry>,
    ) -> Result<Option<u64>, GraphError> {
        tracing::debug!(
            passes = self.flat_passes.len(),
            steps = self.steps.len(),
            "Executing compiled frame graph"
        );

        // 0. Clear per-frame virtual→physical maps.
        backend.reset_frame_resources();

        // 1. Resource resolution & transient allocation (tree-based).
        let mut transient_images = Vec::new();
        let mut transient_buffers = Vec::new();
        Self::resolve_resources_recursive(
            &mut self._root,
            backend,
            temporal_registry,
            &mut transient_images,
            &mut transient_buffers,
        );

        // 2. Begin frame (waits on timelines, resets pools).
        backend.begin_frame();

        // 3. Swapchain acquire + external resource registration.
        let mut inactive_images: Vec<u64> = Vec::with_capacity(2);
        Self::process_externals_recursive(&mut self._root, backend, &mut inactive_images)?;

        // 3.5. Sync plan analysis — all resources now registered.
        backend.analyze_frame(&self.flat_passes);

        // 4. Build node_id → NodeStorage pointer map for O(1) lookup.
        let mut node_map: HashMap<u64, SyncPtr<NodeStorage>> = HashMap::new();
        Self::collect_node_map_sync(&mut self._root, &mut node_map);

        // 4.5. Prepare all active passes.
        let mut prepared_passes: Vec<Option<B::PreparedPass>> =
            Vec::with_capacity(self.flat_passes.len());

        for (idx, flat) in self.flat_passes.iter().enumerate() {
            let is_skipped = !inactive_images.is_empty()
                && flat.image_writes.iter().any(|(h, _)| inactive_images.contains(&h.0.0));

            if is_skipped {
                tracing::debug!(pass = %flat.name, "Skipping pass (targets inactive image)");
                prepared_passes.push(None);
                continue;
            }

            if let Some(node_ptr) = node_map.get(&flat.node_id) {
                let node = unsafe { &mut *node_ptr.0 };
                let desc = crate::graph::backend::PassDescriptor {
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
                tracing::error!(pass = %flat.name, node_id = flat.node_id, "Node not found in tree!");
                prepared_passes.push(None);
            }
        }

        // 5. Execute steps — build an ordered batch for submit().
        let mut batch = CommandBatch::default();

        let push_cb = |batch: &mut CommandBatch, queue: QueueType, cb: BackendCommandBuffer| {
            batch.steps.push(BatchStep::Command { queue, cb });
        };

        for step in &self.steps {
            match step {
                ExecutionStep::Signal { queue, value } => {
                    batch.steps.push(BatchStep::Signal { queue: *queue, value: *value });
                }
                ExecutionStep::Wait { queue, on, value } => {
                    batch.steps.push(BatchStep::Wait { queue: *queue, on: *on, value: *value });
                }
                ExecutionStep::Barriers(pass_indices) => {
                    let mut graphics_refs = Vec::new();
                    let mut compute_refs = Vec::new();
                    let mut transfer_refs = Vec::new();

                    for &idx in pass_indices {
                        if let Some(prepared) = &prepared_passes[idx] {
                            match backend.get_prepared_pass_queue(prepared) {
                                QueueType::Graphics     => graphics_refs.push(prepared),
                                QueueType::AsyncCompute => compute_refs.push(prepared),
                                QueueType::Transfer     => transfer_refs.push(prepared),
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
                                backend.record_pass(prepared, node.pass.as_ref().unwrap().as_ref(), frame_data);
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
                                        frame_data,
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

        // 6. Final submission.
        let submit_res = backend.submit(batch).map_err(GraphError::BackendError)?;
        backend.end_frame();

        // 7. Release transient resources.
        for image in transient_images {
            backend.release_transient_image(image);
        }
        for buffer in transient_buffers {
            backend.release_transient_buffer(buffer);
        }

        Ok(Some(submit_res))
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn collect_node_map_sync(node: &mut NodeStorage, map: &mut HashMap<u64, SyncPtr<NodeStorage>>) {
        map.insert(node.node_id, SyncPtr(node as *mut NodeStorage));
        for child in &mut node.children {
            Self::collect_node_map_sync(child, map);
        }
    }

    fn process_externals_recursive<B: RenderBackendInternal>(
        node: &mut NodeStorage,
        backend: &mut B,
        inactive_images: &mut Vec<u64>,
    ) -> Result<(), GraphError> {
        for (handle, window) in &node.swapchain_requests {
            match backend.acquire_swapchain_image(*window).map_err(GraphError::BackendError)? {
                Some((physical, _sem, _idx)) => {
                    backend.register_external_image(*handle, physical);
                }
                None => {
                    tracing::debug!(window = ?window.0, "Window minimized, skipping passes");
                    inactive_images.push(handle.0.0);
                }
            }
        }

        for (h, p) in &node.external_images {
            backend.register_external_image(*h, *p);
        }
        for (h, p) in &node.external_buffers {
            backend.register_external_buffer(*h, *p);
        }
        for (h, p) in &node.external_accel_structs {
            backend.register_external_accel_struct(*h, *p);
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
        for symbol in &node.symbols.symbols {
            match symbol.symbol_type {
                SymbolType::Image(ref desc) => {
                    if symbol.lifetime == SymbolLifetime::Transient {
                        let physical = backend.create_transient_image(desc);
                        transient_images.push(physical);
                        let handle = *symbol
                            .data.as_ref().expect("Image without handle")
                            .downcast_ref::<ImageHandle>().expect("Not a handle");
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
                            .data.as_ref().expect("Buffer without handle")
                            .downcast_ref::<BufferHandle>().expect("Not a handle");
                        #[cfg(debug_assertions)]
                        backend.set_buffer_name(physical, &symbol.name);
                        backend.register_external_buffer(handle, physical);
                    } else if symbol.lifetime == SymbolLifetime::TemporalHistory {
                        if let Some(ref mut temporal) = temporal_registry_opt {
                            let handle = *symbol
                                .data.as_ref().expect("History buffer without handle")
                                .downcast_ref::<BufferHandle>().expect("Not a handle");
                            let physical = if symbol.name.ends_with("_History") {
                                let base = &symbol.name[..symbol.name.len() - 8];
                                temporal.get_or_create_history_buffer(base, desc, backend)
                            } else {
                                temporal.get_or_create_buffer(&symbol.name, desc, backend)
                            };
                            backend.register_external_buffer(handle, physical);
                        } else {
                            tracing::warn!(
                                "TemporalHistory symbol '{}' declared but no TemporalRegistry!",
                                symbol.name
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        for child in &mut node.children {
            let tr_opt = temporal_registry_opt.as_deref_mut();
            Self::resolve_resources_recursive(child, backend, tr_opt, transient_images, transient_buffers);
        }
    }
}
