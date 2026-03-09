use crate::scene::{MaterialData, ObjectData, SceneProvider};
use i3_gfx::prelude::*;

#[derive(Clone, Copy)]
pub struct ScenePointer(pub *const dyn SceneProvider);

// SAFETY: The SceneProvider is only accessed during the single-threaded
// graph recording phase, or synchronized explicitly by the renderer.
unsafe impl Send for ScenePointer {}
unsafe impl Sync for ScenePointer {}

pub struct ObjectSyncPass {
    pub object_buffer: BufferHandle,
    pub max_objects: usize,
    staging_buffer: Option<BufferHandle>,
    scene: Option<ScenePointer>,
}

impl ObjectSyncPass {
    pub fn new(object_buffer: BufferHandle, max_objects: usize) -> Self {
        Self {
            object_buffer,
            max_objects,
            staging_buffer: None,
            scene: None,
        }
    }
}

impl RenderPass for ObjectSyncPass {
    fn name(&self) -> &str {
        "ObjectSyncPass"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        let scene_wrapper = builder.consume::<ScenePointer>("SceneProvider");
        let scene = unsafe { &*scene_wrapper.0 };
        self.scene = Some(*scene_wrapper);

        let dirty_count = scene.iter_dirty_objects().count();

        // 1. Write to the target persistent SSBO
        builder.write_buffer(self.object_buffer, ResourceUsage::WRITE);

        // 2. Allocate a transient staging buffer
        if dirty_count > 0 {
            let staging_size = (dirty_count * std::mem::size_of::<ObjectData>()) as u64;
            let staging_desc = BufferDesc {
                size: staging_size,
                usage: BufferUsageFlags::TRANSFER_SRC,
                memory: MemoryType::CpuToGpu,
            };
            let staging = builder.declare_buffer("ObjectSync_Staging", staging_desc);
            builder.read_buffer(staging, ResourceUsage::READ);
            self.staging_buffer = Some(staging);
        } else {
            self.staging_buffer = None;
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        if let Some(staging) = self.staging_buffer {
            if let Some(scene_ptr) = self.scene {
                let scene = unsafe { &*scene_ptr.0 };

                let dirty_objects: Vec<_> =
                    scene.iter_dirty_objects().map(|(_, d)| d.clone()).collect();
                let count = dirty_objects.len();
                if count == 0 {
                    return;
                }

                let size = (count * std::mem::size_of::<ObjectData>()) as u64;
                let ptr = ctx.map_buffer(staging);
                if !ptr.is_null() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            dirty_objects.as_ptr() as *const u8,
                            ptr,
                            size as usize,
                        );
                    }
                    ctx.unmap_buffer(staging);
                }

                // Copy from staging to main SSBO
                ctx.copy_buffer(staging, self.object_buffer, 0, 0, size);
            }
        }
    }
}

pub struct MaterialSyncPass {
    pub material_buffer: BufferHandle,
    pub max_materials: usize,
    staging_buffer: Option<BufferHandle>,
    scene: Option<ScenePointer>,
}

impl MaterialSyncPass {
    pub fn new(material_buffer: BufferHandle, max_materials: usize) -> Self {
        Self {
            material_buffer,
            max_materials,
            staging_buffer: None,
            scene: None,
        }
    }
}

impl RenderPass for MaterialSyncPass {
    fn name(&self) -> &str {
        "MaterialSyncPass"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        let scene_wrapper = builder.consume::<ScenePointer>("SceneProvider");
        let scene = unsafe { &*scene_wrapper.0 };
        self.scene = Some(*scene_wrapper);

        let material_count = scene.material_count();

        builder.write_buffer(self.material_buffer, ResourceUsage::WRITE);

        if material_count > 0 {
            let staging_size = (material_count * std::mem::size_of::<MaterialData>()) as u64;
            let staging_desc = BufferDesc {
                size: staging_size,
                usage: BufferUsageFlags::TRANSFER_SRC,
                memory: MemoryType::CpuToGpu,
            };
            let staging = builder.declare_buffer("MaterialSync_Staging", staging_desc);
            builder.read_buffer(staging, ResourceUsage::READ);
            self.staging_buffer = Some(staging);
        } else {
            self.staging_buffer = None;
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        if let Some(staging) = self.staging_buffer {
            if let Some(scene_ptr) = self.scene {
                let scene = unsafe { &*scene_ptr.0 };

                let materials: Vec<_> = scene.iter_materials().map(|(_, d)| *d).collect();
                let count = materials.len();
                if count == 0 {
                    return;
                }

                let size = (count * std::mem::size_of::<MaterialData>()) as u64;
                let ptr = ctx.map_buffer(staging);
                if !ptr.is_null() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            materials.as_ptr() as *const u8,
                            ptr,
                            size as usize,
                        );
                    }
                    ctx.unmap_buffer(staging);
                }

                ctx.copy_buffer(staging, self.material_buffer, 0, 0, size);
            }
        }
    }
}
