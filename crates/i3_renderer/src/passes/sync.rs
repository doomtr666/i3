use crate::scene::{MaterialData, GpuMeshDescriptor, GpuInstanceData};
use i3_gfx::prelude::*;

pub struct MeshRegistrySyncPass {
    mesh_descriptors:        Vec<(u32, GpuMeshDescriptor)>,
    mesh_descriptors_cache:  Vec<(u32, GpuMeshDescriptor)>,
    mesh_descriptor_buffer:  BufferHandle,
    physical_buffer:         BackendBuffer,
    staging_buffer:          Option<BufferHandle>,
    is_dirty:                bool,
}

impl MeshRegistrySyncPass {
    pub fn new() -> Self {
        Self {
            mesh_descriptors:       Vec::new(),
            mesh_descriptors_cache: Vec::new(),
            mesh_descriptor_buffer: BufferHandle::INVALID,
            physical_buffer:        BackendBuffer::INVALID,
            staging_buffer:         None,
            is_dirty:               false,
        }
    }
}

impl RenderPass for MeshRegistrySyncPass {
    fn name(&self) -> &str {
        "MeshRegistrySyncPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {
        let max_meshes: u64 = 16384;
        self.physical_buffer = backend.create_buffer(&BufferDesc {
            size: max_meshes * std::mem::size_of::<GpuMeshDescriptor>() as u64,
            usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            memory: MemoryType::CpuToGpu,
        });
        #[cfg(debug_assertions)]
        backend.set_buffer_name(self.physical_buffer, "MeshDescriptorBuffer");
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.mesh_descriptor_buffer = builder.import_buffer("MeshDescriptorBuffer", self.physical_buffer);

        let mesh_descriptors = builder.consume::<Vec<(u32, GpuMeshDescriptor)>>("SceneMeshDescriptors");
        
        // Dirty check: only update if the data has changed
        self.is_dirty = mesh_descriptors.len() != self.mesh_descriptors_cache.len() 
            || mesh_descriptors.iter().zip(self.mesh_descriptors_cache.iter()).any(|(a, b)| a != b);

        if !self.is_dirty {
            self.staging_buffer = None;
            return;
        }

        self.mesh_descriptors = mesh_descriptors.clone();
        self.mesh_descriptors_cache = mesh_descriptors.clone();
        let count = self.mesh_descriptors.len();

        // Use TRANSFER_WRITE for better synchronization in the backend
        builder.write_buffer(self.mesh_descriptor_buffer, ResourceUsage::TRANSFER_WRITE);

        if count > 0 {
            let staging_size = (count * std::mem::size_of::<GpuMeshDescriptor>()) as u64;
            let staging = builder.declare_buffer("MeshRegistrySync_Staging", BufferDesc {
                size: staging_size,
                usage: BufferUsageFlags::TRANSFER_SRC,
                memory: MemoryType::CpuToGpu,
            });
            builder.read_buffer(staging, ResourceUsage::TRANSFER_READ);
            self.staging_buffer = Some(staging);
        } else {
            self.staging_buffer = None;
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        if let Some(staging) = self.staging_buffer {
            let count = self.mesh_descriptors.len();
            if count == 0 { return; }

            let size = (count * std::mem::size_of::<GpuMeshDescriptor>()) as u64;
            let ptr = ctx.map_buffer(staging);
            if !ptr.is_null() {
                let mut flat_descriptors = vec![
                    GpuMeshDescriptor {
                        vertex_buffer_address: 0,
                        index_buffer_address: 0,
                        index_count: 0,
                        vertex_stride: 0,
                        first_index: 0,
                        vertex_offset: 0,
                        aabb_min: [0.0; 3],
                        index_stride: 0,
                        aabb_max: [0.0; 3],
                        _pad1: 0.0,
                    };
                    count
                ];
                for (id, desc) in &self.mesh_descriptors {
                    let idx = *id as usize;
                    if idx < count {
                        flat_descriptors[idx] = *desc;
                    }
                }

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        flat_descriptors.as_ptr() as *const u8,
                        ptr,
                        size as usize,
                    );
                }
                ctx.unmap_buffer(staging);
            }

            ctx.copy_buffer(staging, self.mesh_descriptor_buffer, 0, 0, size);
        }
    }
}

pub struct InstanceSyncPass {
    instances:        Vec<GpuInstanceData>,
    instances_cache:  Vec<GpuInstanceData>,
    instance_buffer:  BufferHandle,
    physical_buffer:  BackendBuffer,
    staging_buffer:   Option<BufferHandle>,
    is_dirty:         bool,
}

impl InstanceSyncPass {
    pub fn new() -> Self {
        Self {
            instances:       Vec::new(),
            instances_cache: Vec::new(),
            instance_buffer: BufferHandle::INVALID,
            physical_buffer: BackendBuffer::INVALID,
            staging_buffer:  None,
            is_dirty:        false,
        }
    }
}

impl RenderPass for InstanceSyncPass {
    fn name(&self) -> &str {
        "InstanceSyncPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {
        let max_instances: u64 = 262144;
        self.physical_buffer = backend.create_buffer(&BufferDesc {
            size: max_instances * std::mem::size_of::<GpuInstanceData>() as u64,
            usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            memory: MemoryType::CpuToGpu,
        });
        #[cfg(debug_assertions)]
        backend.set_buffer_name(self.physical_buffer, "InstanceBuffer");
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.instance_buffer = builder.import_buffer("InstanceBuffer", self.physical_buffer);

        let instances = builder.consume::<Vec<GpuInstanceData>>("SceneInstances");
        
        // Dirty check
        self.is_dirty = instances.len() != self.instances_cache.len() 
            || instances.iter().zip(self.instances_cache.iter()).any(|(a, b)| a != b);

        if !self.is_dirty {
            self.staging_buffer = None;
            return;
        }

        self.instances = instances.clone();
        self.instances_cache = instances.clone();
        let count = self.instances.len();

        builder.write_buffer(self.instance_buffer, ResourceUsage::TRANSFER_WRITE);

        if count > 0 {
            let staging_size = (count * std::mem::size_of::<GpuInstanceData>()) as u64;
            let staging = builder.declare_buffer("InstanceSync_Staging", BufferDesc {
                size: staging_size,
                usage: BufferUsageFlags::TRANSFER_SRC,
                memory: MemoryType::CpuToGpu,
            });
            builder.read_buffer(staging, ResourceUsage::TRANSFER_READ);
            self.staging_buffer = Some(staging);
        } else {
            self.staging_buffer = None;
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        if let Some(staging) = self.staging_buffer {
            let count = self.instances.len();
            if count == 0 { return; }

            let size = (count * std::mem::size_of::<GpuInstanceData>()) as u64;
            let ptr = ctx.map_buffer(staging);
            if !ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.instances.as_ptr() as *const u8,
                        ptr,
                        size as usize,
                    );
                }
                ctx.unmap_buffer(staging);
            }

            ctx.copy_buffer(staging, self.instance_buffer, 0, 0, size);
        }
    }
}

pub struct MaterialSyncPass {
    pub max_materials: usize,

    // Resolved handles (updated in declare)
    material_buffer:  BufferHandle,
    physical_buffer:  BackendBuffer,
    staging_buffer:   Option<BufferHandle>,
    materials:        Vec<(u32, MaterialData)>,
    materials_cache:  Vec<(u32, MaterialData)>,
    is_dirty:         bool,
}

impl MaterialSyncPass {
    pub fn new(max_materials: usize) -> Self {
        Self {
            material_buffer: BufferHandle::INVALID,
            physical_buffer: BackendBuffer::INVALID,
            max_materials,
            staging_buffer:  None,
            materials:       Vec::new(),
            materials_cache: Vec::new(),
            is_dirty:        false,
        }
    }
}

impl RenderPass for MaterialSyncPass {
    fn name(&self) -> &str {
        "MaterialSyncPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {
        self.physical_buffer = backend.create_buffer(&BufferDesc {
            size: 1024 * 1024, // 1MB for ~16k materials
            usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            memory: MemoryType::CpuToGpu,
        });
        #[cfg(debug_assertions)]
        backend.set_buffer_name(self.physical_buffer, "MaterialBuffer");
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.material_buffer = builder.import_buffer("MaterialBuffer", self.physical_buffer);

        let materials = builder.consume::<Vec<(u32, MaterialData)>>("SceneMaterials");
        
        // Dirty check
        self.is_dirty = materials.len() != self.materials_cache.len() 
            || materials.iter().zip(self.materials_cache.iter()).any(|(a, b)| a != b);

        if !self.is_dirty {
            self.staging_buffer = None;
            return;
        }

        self.materials = materials.clone();
        self.materials_cache = materials.clone();
        let material_count = self.materials.len();

        builder.write_buffer(self.material_buffer, ResourceUsage::TRANSFER_WRITE);

        if material_count > 0 {
            let staging_size = (material_count * std::mem::size_of::<MaterialData>()) as u64;
            let staging_desc = BufferDesc {
                size: staging_size,
                usage: BufferUsageFlags::TRANSFER_SRC,
                memory: MemoryType::CpuToGpu,
            };
            let staging = builder.declare_buffer("MaterialSync_Staging", staging_desc);
            builder.read_buffer(staging, ResourceUsage::TRANSFER_READ);
            self.staging_buffer = Some(staging);
        } else {
            self.staging_buffer = None;
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &i3_gfx::graph::compiler::FrameBlackboard) {
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
