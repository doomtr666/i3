use crate::passes::sync::{MaterialSyncPass, InstanceSyncPass, MeshRegistrySyncPass};
use i3_gfx::prelude::*;

/// Group that manages syncing Scene data (objects and materials) to the GPU.
pub struct SyncGroup {
    pub mesh_registry_sync: MeshRegistrySyncPass,
    pub instance_sync:      InstanceSyncPass,
    pub material_sync:      MaterialSyncPass,
}

impl SyncGroup {
    pub fn new() -> Self {
        Self {
            mesh_registry_sync: MeshRegistrySyncPass::new(),
            instance_sync:      InstanceSyncPass::new(),
            material_sync:      MaterialSyncPass::new(),
        }
    }
}

impl RenderPass for SyncGroup {
    fn name(&self) -> &str {
        "SyncGroup"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.mesh_registry_sync.init(backend, globals);
        self.instance_sync.init(backend, globals);
        self.material_sync.init(backend, globals);
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.add_pass(&mut self.mesh_registry_sync);
        builder.add_pass(&mut self.instance_sync);
        builder.add_pass(&mut self.material_sync);
    }

    fn execute(&self, _ctx: &mut dyn PassContext, _frame: &i3_gfx::graph::compiler::FrameBlackboard) {}
}
