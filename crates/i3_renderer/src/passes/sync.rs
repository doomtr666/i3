use crate::scene::{MaterialData, ObjectData};
use i3_gfx::prelude::*;

pub struct ObjectSyncPass {
    pub max_objects: usize,

    // Resolved handles (updated in record)
    object_buffer: BufferHandle,
    staging_buffer: Option<BufferHandle>,
    objects: Vec<(u64, ObjectData)>,
}

impl ObjectSyncPass {
    pub fn new(max_objects: usize) -> Self {
        Self {
            object_buffer: BufferHandle::INVALID,
            max_objects,
            staging_buffer: None,
            objects: Vec::new(),
        }
    }
}

impl RenderPass for ObjectSyncPass {
    fn name(&self) -> &str {
        "ObjectSyncPass"
    }

    fn init(&mut self, _backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {}

    fn record(&mut self, builder: &mut PassBuilder) {
        if builder.is_setup() {
            return;
        }
        self.object_buffer = builder.resolve_buffer("ObjectBuffer");

        let objects = builder.consume::<Vec<(u64, ObjectData)>>("SceneObjects");
        self.objects = objects.clone();
        let count = self.objects.len();

        // 1. Write to the target persistent SSBO
        builder.write_buffer(self.object_buffer, ResourceUsage::WRITE);

        // 2. Allocate a transient staging buffer
        if count > 0 {
            let mut max_id = 0;
            for (id, _) in &self.objects {
                if *id as usize > max_id {
                    max_id = *id as usize;
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
            let mut max_id = 0;
            for (id, _) in &self.objects {
                if *id as usize > max_id {
                    max_id = *id as usize;
                }
            }

            let count = if !self.objects.is_empty() {
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

            for (id, data) in &self.objects {
                objects[*id as usize] = data.clone();
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

pub struct MaterialSyncPass {
    pub max_materials: usize,

    // Resolved handles (updated in record)
    material_buffer: BufferHandle,
    staging_buffer: Option<BufferHandle>,
    materials: Vec<(u32, MaterialData)>,
}

impl MaterialSyncPass {
    pub fn new(max_materials: usize) -> Self {
        Self {
            material_buffer: BufferHandle::INVALID,
            max_materials,
            staging_buffer: None,
            materials: Vec::new(),
        }
    }
}

impl RenderPass for MaterialSyncPass {
    fn name(&self) -> &str {
        "MaterialSyncPass"
    }

    fn init(&mut self, _backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {}

    fn record(&mut self, builder: &mut PassBuilder) {
        if builder.is_setup() {
            return;
        }
        self.material_buffer = builder.resolve_buffer("MaterialBuffer");

        let materials = builder.consume::<Vec<(u32, MaterialData)>>("SceneMaterials");
        self.materials = materials.clone();
        let material_count = self.materials.len();

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
            let mut max_id = 0;
            for (id, _) in &self.materials {
                if *id as usize > max_id {
                    max_id = *id as usize;
                }
            }

            let count = if !self.materials.is_empty() {
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

            for (id, data) in &self.materials {
                let idx = *id as usize;
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
