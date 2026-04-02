use i3_gfx::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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

/// Pass responsible for building/updating BLAS for all meshes.
pub struct BlasUpdatePass {
    pub system: Arc<Mutex<AccelStructSystem>>,
    pub builds: Vec<BackendAccelerationStructure>,
}

impl BlasUpdatePass {
    pub fn new(system: Arc<Mutex<AccelStructSystem>>) -> Self {
        Self {
            system,
            builds: Vec::new(),
        }
    }
}

impl RenderPass for BlasUpdatePass {
    fn name(&self) -> &str {
        "BlasUpdate"
    }

    fn init(&mut self, _backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {}

    fn declare(&mut self, builder: &mut PassBuilder) {
        if builder.is_setup() {
            return;
        }

        // Consume pending builds from sync()
        if let Some(builds) = builder.try_consume::<Vec<BackendAccelerationStructure>>("PendingBlasBuilds") {
            self.builds = builds.clone();
        } else {
            self.builds.clear();
        }

        // Declare usage of all active BLAS
        let system = self.system.lock().unwrap();
        for (_, &handle) in &system.blas_cache {
            builder.write_acceleration_structure(handle, ResourceUsage::ACCEL_STRUCT_WRITE);
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        for &handle in &self.builds {
            ctx.build_blas(handle, false); // Initial build
        }
    }
}

/// Pass responsible for rebuilding the TLAS from visible instances.
pub struct TlasRebuildPass {
    pub system: Arc<Mutex<AccelStructSystem>>,
    pub instances: Vec<TlasInstanceDesc>,
}

impl TlasRebuildPass {
    pub fn new(system: Arc<Mutex<AccelStructSystem>>) -> Self {
        Self {
            system,
            instances: Vec::new(),
        }
    }
}

impl RenderPass for TlasRebuildPass {
    fn name(&self) -> &str {
        "TlasRebuild"
    }

    fn init(&mut self, _backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {}

    fn declare(&mut self, builder: &mut PassBuilder) {
        if builder.is_setup() {
            return;
        }

        // Declare TLAS usage
        let system = self.system.lock().unwrap();
        if let Some(tlas) = system.tlas {
            builder.write_acceleration_structure(tlas, ResourceUsage::ACCEL_STRUCT_WRITE);
        }

        // Consume visible instances from culling/scene
        if let Some(instances) = builder.try_consume::<Vec<TlasInstanceDesc>>("TlasInstances") {
            self.instances = instances.clone();
        } else {
            self.instances.clear();
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let system = self.system.lock().unwrap();
        if let Some(tlas) = system.tlas {
            if !self.instances.is_empty() {
                ctx.build_tlas(tlas, &self.instances, false); // Full rebuild for now
            }
        }
    }
}
