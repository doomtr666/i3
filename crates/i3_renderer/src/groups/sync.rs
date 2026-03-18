use crate::passes::sync::{MaterialSyncPass, ObjectSyncPass};
use i3_gfx::prelude::*;

/// Group that manages syncing Scene data (objects and materials) to the GPU.
pub struct SyncGroup {
    pub object_sync: ObjectSyncPass,
    pub material_sync: MaterialSyncPass,
}

impl SyncGroup {
    pub fn new(max_objects: usize, max_materials: usize) -> Self {
        Self {
            object_sync: ObjectSyncPass::new(max_objects),
            material_sync: MaterialSyncPass::new(max_materials),
        }
    }
}

impl RenderPass for SyncGroup {
    fn name(&self) -> &str {
        "SyncGroup"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend) {
        self.object_sync.init(backend);
        self.material_sync.init(backend);
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.add_pass(&mut self.material_sync);
        builder.add_pass(&mut self.object_sync);
    }

    fn execute(&self, _ctx: &mut dyn PassContext) {}
}
