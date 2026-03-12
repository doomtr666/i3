use crate::passes::sync::{MaterialSyncPass, ObjectSyncPass};
use i3_gfx::prelude::*;
use std::sync::{Arc, Mutex};

/// Group that manages syncing Scene data (objects and materials) to the GPU.
pub struct SyncGroup {
    pub object_sync: Arc<Mutex<ObjectSyncPass>>,
    pub material_sync: Arc<Mutex<MaterialSyncPass>>,
}

impl SyncGroup {
    pub fn new(max_objects: usize, max_materials: usize) -> Self {
        Self {
            object_sync: Arc::new(Mutex::new(ObjectSyncPass::new(max_objects))),
            material_sync: Arc::new(Mutex::new(MaterialSyncPass::new(max_materials))),
        }
    }
}

impl RenderPass for SyncGroup {
    fn name(&self) -> &str {
        "SyncGroup"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend) {
        self.object_sync.lock().unwrap().init(backend);
        self.material_sync.lock().unwrap().init(backend);
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.add_pass(self.material_sync.clone());
        builder.add_pass(self.object_sync.clone());
    }

    fn execute(&self, _ctx: &mut dyn PassContext) {}
}
