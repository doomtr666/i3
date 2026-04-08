use i3_gfx::prelude::*;
use std::collections::HashMap;

/// System for managing acceleration structure handles in the renderer.
/// This persists across frames and caches BLAS objects.
pub struct AccelStructSystem {
    pub blas_cache: HashMap<u32, BackendAccelerationStructure>,
    pub tlas: Option<BackendAccelerationStructure>,
}

impl AccelStructSystem {
    pub fn new() -> Self {
        Self {
            blas_cache: HashMap::new(),
            tlas: None,
        }
    }

    /// Destroys all cached acceleration structures and clears state.
    pub fn reset(&mut self, backend: &mut dyn RenderBackend) {
        for (_, &handle) in &self.blas_cache {
            backend.destroy_blas(handle);
        }
        self.blas_cache.clear();

        if let Some(handle) = self.tlas.take() {
            backend.destroy_tlas(handle);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BlasUpdatePass
// ─────────────────────────────────────────────────────────────────────────────

/// Pass responsible for building newly created BLAS.
///
/// `builds` and `blas_handles` are populated by `DefaultRenderGraph::sync()`
/// before `declare()` is called, so no blackboard access is needed here.
pub struct BlasUpdatePass {
    /// Newly created BLAS to build this frame (populated by sync()).
    pub builds: Vec<BackendAccelerationStructure>,
}

impl BlasUpdatePass {
    pub fn new() -> Self {
        Self { builds: Vec::new() }
    }
}

impl RenderPass for BlasUpdatePass {
    fn name(&self) -> &str {
        "BlasUpdate"
    }

    fn init(&mut self, _backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {}

    fn declare(&mut self, builder: &mut PassBuilder) {
        // Declare write access only for BLAS that are actually being built this frame.
        for &handle in &self.builds {
            builder.write_acceleration_structure(handle, ResourceUsage::ACCEL_STRUCT_WRITE);
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        for &handle in &self.builds {
            ctx.build_blas(handle, false); // Initial build
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TlasRebuildPass
// ─────────────────────────────────────────────────────────────────────────────

/// Pass responsible for rebuilding the TLAS when the instance list changes.
///
/// `tlas` and `instances` are populated by `DefaultRenderGraph::sync()`.
/// A frame-to-frame dirty check avoids the rebuild when nothing changed.
pub struct TlasRebuildPass {
    /// TLAS handle (populated by sync()).
    pub tlas: Option<BackendAccelerationStructure>,
    /// Instance list for this frame (populated by sync()).
    pub instances: Vec<TlasInstanceDesc>,

    // Internal dirty-tracking state.
    instances_cache: Vec<TlasInstanceDesc>,
    tlas_dirty: bool,
}

impl TlasRebuildPass {
    pub fn new() -> Self {
        Self {
            tlas: None,
            instances: Vec::new(),
            instances_cache: Vec::new(),
            tlas_dirty: false,
        }
    }

    /// Clears cached state (e.g. after a scene switch).
    pub fn reset(&mut self) {
        self.tlas = None;
        self.instances.clear();
        self.instances_cache.clear();
        self.tlas_dirty = false;
    }
}

impl RenderPass for TlasRebuildPass {
    fn name(&self) -> &str {
        "TlasRebuild"
    }

    fn init(&mut self, _backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {}

    fn declare(&mut self, builder: &mut PassBuilder) {
        if let Some(tlas) = self.tlas {
            builder.write_acceleration_structure(tlas, ResourceUsage::ACCEL_STRUCT_WRITE);
            // Publish as virtual handle so downstream passes (e.g. DeferredResolve) can consume
            // "TLAS" by name via try_resolve_acceleration_structure — no direct backend handle needed.
            builder.import_acceleration_structure("TLAS", tlas);
        }

        // Dirty check: only rebuild when the instance list changed.
        self.tlas_dirty = self.instances.len() != self.instances_cache.len()
            || self.instances
                .iter()
                .zip(self.instances_cache.iter())
                .any(|(a, b)| a != b);

        if self.tlas_dirty {
            self.instances_cache = self.instances.clone();
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        if !self.tlas_dirty {
            return;
        }
        if let Some(tlas) = self.tlas {
            if !self.instances.is_empty() {
                ctx.build_tlas(tlas, &self.instances, false);
            }
        }
    }
}
