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

        let count = scene.iter_objects().count();

        // 1. Write to the target persistent SSBO
        builder.write_buffer(self.object_buffer, ResourceUsage::WRITE);

        // 2. Allocate a transient staging buffer
        if count > 0 {
            // Find max object ID to establish array bounds
            let mut max_id = 0;
            for (id, _) in scene.iter_objects() {
                if id.0 as usize > max_id {
                    max_id = id.0 as usize;
                }
            }
            let array_len = max_id + 1;

            let staging_size = (array_len * std::mem::size_of::<ObjectData>()) as u64;
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

                let mut max_id = 0;
                for (id, _) in scene.iter_objects() {
                    if id.0 as usize > max_id {
                        max_id = id.0 as usize;
                    }
                }

                let count = if scene.iter_objects().count() > 0 {
                    max_id + 1
                } else {
                    0
                };

                if count == 0 {
                    return;
                }

                let mut objects = vec![
                    ObjectData {
                        world_transform: nalgebra_glm::Mat4::identity(),
                        prev_transform: nalgebra_glm::Mat4::identity(),
                        material_id: 0,
                        mesh_id: 0,
                        flags: 0,
                        _pad: 0,
                    };
                    count
                ];

                for (id, data) in scene.iter_objects() {
                    objects[id.0 as usize] = data.clone();
                }

                let size = (count * std::mem::size_of::<ObjectData>()) as u64;
                let ptr = ctx.map_buffer(staging);
                if !ptr.is_null() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            objects.as_ptr() as *const u8,
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

                let mut max_id = 0;
                for (id, _) in scene.iter_materials() {
                    if id.0 as usize > max_id {
                        max_id = id.0 as usize;
                    }
                }

                let count = if scene.iter_materials().count() > 0 {
                    max_id + 1
                } else {
                    0
                };

                if count == 0 {
                    return;
                }

                let mut materials = vec![
                    MaterialData {
                        base_color_factor: [1.0, 0.0, 1.0, 1.0], // Magenta default for missing
                        emissive_factor_and_alpha_cutoff: [0.0; 4],
                        metallic_factor: 0.0,
                        roughness_factor: 1.0,
                        _pad_pbr: [0.0; 2],
                        albedo_tex_index: -1,
                        normal_tex_index: -1,
                        rmao_tex_index: -1,
                        emissive_tex_index: -1,
                    };
                    count
                ];

                static LOG_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                let should_log = !LOG_DONE.swap(true, std::sync::atomic::Ordering::Relaxed);

                if should_log {
                    tracing::debug!(
                        "INITIAL MATERIAL DUMP ({} materials, size_of={})", 
                        count, 
                        std::mem::size_of::<MaterialData>()
                    );
                }

                for (id, data) in scene.iter_materials() {
                    let idx = id.0 as usize;
                    materials[idx] = *data;
                    
                    if should_log {
                        tracing::debug!(
                            "  Material[{}]: albedo={}, normal={}, rmao={}, emissive={}, Color={:?}, PBR={:?}",
                            idx,
                            data.albedo_tex_index,
                            data.normal_tex_index,
                            data.rmao_tex_index,
                            data.emissive_tex_index,
                            data.base_color_factor,
                            [data.metallic_factor, data.roughness_factor]
                        );
                    }
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
