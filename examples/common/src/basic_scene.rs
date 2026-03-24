use std::collections::{HashMap, HashSet};
use tracing::debug;

use i3_gfx::prelude::*;
use i3_io::mesh::{IndexFormat, MeshAsset};
use i3_io::scene_asset::{LightType as AssetLightType, SceneAsset};
use i3_renderer::scene::{
    GpuInstanceData, GpuMeshDescriptor, LightData, LightId, LightType, MaterialData, MaterialId,
    Mesh, ObjectData, ObjectId, SceneProvider,
};
use nalgebra_glm as glm;
use uuid::Uuid;

/// A simple in-memory scene for examples and integration tests.
///
/// Owns GPU-resident meshes (VB/IB handles), objects, and lights.
/// Implements `SceneProvider` so it can be passed directly to the renderer.
pub struct BasicScene {
    meshes: Vec<Mesh>,
    /// Mapping from mesh UUID to local mesh index.
    mesh_uuid_to_index: HashMap<Uuid, u32>,
    /// Mapping from mesh UUID to its material UUID.
    mesh_uuid_to_material: HashMap<Uuid, Uuid>,
    /// Mapping from material UUID to local material index.
    material_uuid_to_index: HashMap<Uuid, u32>,
    objects: Vec<(ObjectId, ObjectData)>,
    materials: Vec<(MaterialId, MaterialData)>,
    lights: Vec<(LightId, LightData)>,
    dirty_objects: HashSet<ObjectId>,
    dirty_materials: HashSet<MaterialId>,
    next_object_id: u64,
    next_material_id: u32,
    next_light_id: u64,
    bounds: i3_io::mesh::BoundingBox,
}

impl BasicScene {
    pub fn new() -> Self {
        let mut scene = Self {
            meshes: Vec::new(),
            mesh_uuid_to_index: HashMap::new(),
            mesh_uuid_to_material: HashMap::new(),
            material_uuid_to_index: HashMap::new(),
            objects: Vec::new(),
            materials: Vec::new(),
            lights: Vec::new(),
            dirty_objects: HashSet::new(),
            dirty_materials: HashSet::new(),
            next_object_id: 0,
            next_material_id: 0,
            next_light_id: 0,
            bounds: i3_io::mesh::BoundingBox::empty(),
        };

        // Add default material at index 0
        scene.add_material(MaterialData {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            emissive_factor_and_alpha_cutoff: [0.0, 0.0, 0.0, 0.0],
            metallic_factor: 0.0,
            roughness_factor: 0.5,
            _pad_pbr: [0.0; 2],
            albedo_tex_index: -1,
            normal_tex_index: -1,
            rmao_tex_index: -1,
            emissive_tex_index: -1,
        });

        scene
    }

    pub fn bounds(&self) -> &i3_io::mesh::BoundingBox {
        &self.bounds
    }

    /// Uploads a mesh to the GPU and returns its ID.
    ///
    /// Vertices must be tightly packed `GBufferVertex` structs
    /// (position: [f32;3], normal: [f32;3], color: [f32;3]).
    pub fn add_mesh(
        &mut self,
        backend: &mut dyn RenderBackend,
        vertices: &[u8],
        vertex_count: u32,
        indices: &[u16],
    ) -> u32 {
        let vb = backend.create_buffer(&BufferDesc {
            size: vertices.len() as u64,
            usage: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::DEVICE_ADDRESS,
            memory: MemoryType::CpuToGpu,
        });
        backend
            .upload_buffer(vb, vertices, 0)
            .expect("Failed to upload mesh vertices");

        let ib_bytes = unsafe {
            std::slice::from_raw_parts(
                indices.as_ptr() as *const u8,
                indices.len() * std::mem::size_of::<u16>(),
            )
        };
        let ib = backend.create_buffer(&BufferDesc {
            size: ib_bytes.len() as u64,
            usage: BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::DEVICE_ADDRESS,
            memory: MemoryType::CpuToGpu,
        });
        backend
            .upload_buffer(ib, ib_bytes, 0)
            .expect("Failed to upload mesh indices");

        let id = self.meshes.len() as u32;
        self.meshes.push(Mesh {
            vertex_buffer: vb,
            index_buffer: ib,
            index_count: indices.len() as u32,
            index_type: IndexType::Uint16,
            stride: 48, // Match GBufferVertex [f32; 12]
        });

        let _ = vertex_count; // Reserved for future validation
        id
    }

    /// Adds an object to the scene, returns its ID.
    pub fn add_object(&mut self, data: ObjectData) -> ObjectId {
        let id = ObjectId(self.next_object_id);
        self.next_object_id += 1;
        self.dirty_objects.insert(id);
        self.objects.push((id, data));
        id
    }

    /// Adds a material to the scene, returns its ID.
    pub fn add_material(&mut self, data: MaterialData) -> MaterialId {
        let id = MaterialId(self.next_material_id);
        self.next_material_id += 1;
        self.dirty_materials.insert(id);
        self.materials.push((id, data));
        id
    }

    /// Adds a light to the scene, returns its ID.
    pub fn add_light(&mut self, data: LightData) -> LightId {
        let id = LightId(self.next_light_id);
        self.next_light_id += 1;
        self.lights.push((id, data));
        id
    }

    /// Updates an object's transform, marking it dirty.
    pub fn set_transform(&mut self, id: ObjectId, transform: glm::Mat4) {
        if let Some((_, data)) = self.objects.iter_mut().find(|(oid, _)| *oid == id) {
            data.prev_transform = data.world_transform;
            data.world_transform = transform;
            self.dirty_objects.insert(id);
        }
    }

    /// Clears the dirty set. Call after the renderer has consumed the deltas.
    pub fn clear_dirty(&mut self) {
        self.dirty_objects.clear();
        self.dirty_materials.clear();
    }

    /// Returns the mesh list (for inspection/debugging).
    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }

    /// Convenience: creates a unit cube mesh and returns its MeshId.
    pub fn add_cube_mesh(&mut self, backend: &mut dyn RenderBackend) -> u32 {
        let (vertices, indices) = generate_cube();
        let vb_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * std::mem::size_of::<[f32; 12]>(),
            )
        };
        self.add_mesh(backend, vb_bytes, vertices.len() as u32, &indices)
    }

    pub fn add_white_cube_mesh(&mut self, backend: &mut dyn RenderBackend) -> u32 {
        let (vertices, indices) = generate_white_cube();
        let vb_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * std::mem::size_of::<[f32; 12]>(),
            )
        };
        self.add_mesh(backend, vb_bytes, vertices.len() as u32, &indices)
    }

    /// Convenience: adds default directional lights (key light and backlight).
    pub fn add_default_lights(&mut self) {
        // Main key light (Warm white)
        self.add_light(LightData {
            position: glm::vec3(0.0, 0.0, 0.0),
            direction: glm::normalize(&glm::vec3(-1.0, -1.0, -1.0)),
            color: glm::vec3(1.0, 0.95, 0.9),
            intensity: 1.0,
            radius: 0.0,
            light_type: LightType::Directional,
        });

        // Backlight (Neutral sky-blue / ambient fill)
        self.add_light(LightData {
            position: glm::vec3(0.0, 0.0, 0.0),
            direction: glm::normalize(&glm::vec3(1.0, 0.5, 1.0)),
            color: glm::vec3(0.5, 0.6, 0.8),
            intensity: 0.5,
            radius: 0.0,
            light_type: LightType::Directional,
        });
    }

    /// Uploads a baked mesh asset to the GPU and registers it by UUID.
    ///
    /// Returns the local mesh index. The mesh can later be referenced by its UUID
    /// when loading a baked scene.
    pub fn add_baked_mesh(
        &mut self,
        backend: &mut dyn RenderBackend,
        mesh_asset: &MeshAsset,
        mesh_uuid: Uuid,
    ) -> u32 {
        // Create vertex buffer
        let vb = backend.create_buffer(&BufferDesc {
            size: mesh_asset.vertex_data.len() as u64,
            usage: BufferUsageFlags::VERTEX_BUFFER | BufferUsageFlags::DEVICE_ADDRESS,
            memory: MemoryType::CpuToGpu,
        });
        debug!(
            "Loading baked mesh {:?} with stride {}",
            mesh_uuid, mesh_asset.header.vertex_stride
        );
        backend
            .upload_buffer(vb, &mesh_asset.vertex_data, 0)
            .expect("Failed to upload baked mesh vertices");

        // Create index buffer
        let ib = backend.create_buffer(&BufferDesc {
            size: mesh_asset.index_data.len() as u64,
            usage: BufferUsageFlags::INDEX_BUFFER | BufferUsageFlags::DEVICE_ADDRESS,
            memory: MemoryType::CpuToGpu,
        });
        backend
            .upload_buffer(ib, &mesh_asset.index_data, 0)
            .expect("Failed to upload baked mesh indices");

        let id = self.meshes.len() as u32;
        let index_type = if mesh_asset.header.index_format == IndexFormat::U32 {
            IndexType::Uint32
        } else {
            IndexType::Uint16
        };

        self.meshes.push(Mesh {
            vertex_buffer: vb,
            index_buffer: ib,
            index_count: mesh_asset.header.index_count,
            index_type,
            stride: mesh_asset.header.vertex_stride,
        });

        // Register UUID mapping
        self.mesh_uuid_to_index.insert(mesh_uuid, id);

        let mat_uuid = Uuid::from_bytes(mesh_asset.header.material_id);
        self.mesh_uuid_to_material.insert(mesh_uuid, mat_uuid);

        id
    }

    /// Uploads a baked material and its textures, registering them with the BindlessManager.
    pub fn add_baked_material<T: i3_gfx::graph::backend::RenderBackendInternal + ?Sized>(
        &mut self,
        backend: &mut T,
        bindless: &mut i3_renderer::bindless::BindlessManager,
        material: &i3_io::material::MaterialAsset,
        material_uuid: Uuid,
        texture_loader: &mut dyn FnMut(&Uuid, &mut T) -> Option<ImageHandle>,
    ) -> u32 {
        // Helper to load and register a texture
        let mut process_texture = |tex_uuid: &[u8; 16]| -> i32 {
            let id = Uuid::from_bytes(*tex_uuid);
            if !id.is_nil() {
                if let Some(handle) = texture_loader(&id, backend) {
                    return bindless.register_texture(backend, handle) as i32;
                }
            }
            -1
        };

        if let Some(header) = &material.header {
            let albedo_idx = process_texture(&header.albedo_texture);
            let normal_idx = process_texture(&header.normal_texture);
            let rmao_idx = process_texture(&header.metallic_roughness_texture);
            let emissive_idx = process_texture(&header.emissive_texture);

            let data = MaterialData {
                base_color_factor: header.base_color_factor,
                emissive_factor_and_alpha_cutoff: [
                    header.emissive_factor[0],
                    header.emissive_factor[1],
                    header.emissive_factor[2],
                    header.alpha_cutoff,
                ],
                metallic_factor: header.metallic_factor,
                roughness_factor: header.roughness_factor,
                _pad_pbr: [0.0; 2],
                albedo_tex_index: albedo_idx,
                normal_tex_index: normal_idx,
                rmao_tex_index: rmao_idx,
                emissive_tex_index: emissive_idx,
            };

            let material_id = self.materials.len() as u32;
            self.materials.push((MaterialId(material_id), data));
            self.material_uuid_to_index
                .insert(material_uuid, material_id);

            return material_id;
        }

        0
    }

    /// Loads a baked scene asset, creating objects and lights.
    ///
    /// Meshes must be loaded first via `add_baked_mesh`. Objects reference meshes
    /// by UUID, which is resolved to the local mesh index.
    ///
    /// Returns the number of objects added.
    pub fn load_baked_scene(&mut self, scene: &SceneAsset) -> usize {
        self.bounds = scene.bounds;
        let object_count = scene.objects.len();

        // Add objects
        for obj in &scene.objects {
            // Resolve mesh UUID to local index
            let mesh_uuid = scene.mesh_for_object(obj).unwrap_or(Uuid::nil());
            let mesh_id = self
                .mesh_uuid_to_index
                .get(&mesh_uuid)
                .copied()
                .unwrap_or(0);

            // Objects inherit the material bound to their mesh.
            // Eventually scene assets could override materials per-object, but Assimp usually bounds them per-mesh.
            let material_uuid = self
                .mesh_uuid_to_material
                .get(&mesh_uuid)
                .copied()
                .unwrap_or(Uuid::nil());
            let material_id = self
                .material_uuid_to_index
                .get(&material_uuid)
                .copied()
                .unwrap_or(0);

            // Convert transform from [[f32; 4]; 4] to glm::Mat4
            let transform = glm::Mat4::from(obj.transform);

            let object_id = self.add_object(ObjectData {
                world_transform: transform,
                prev_transform: transform,
                mesh_id,
                material_id, // Resolved material
                flags: 0,
                _pad: 0,
            });

            // Set name if available
            if let Some(name) = scene.object_name(obj) {
                tracing::debug!("Loaded object '{}' (id={:?})", name, object_id);
            }
        }

        // Add lights
        for light in &scene.lights {
            let light_type = match AssetLightType(light.light_type) {
                AssetLightType::POINT => LightType::Point,
                AssetLightType::DIRECTIONAL => LightType::Directional,
                AssetLightType::SPOT => LightType::Spot,
                _ => LightType::Point, // Default fallback
            };

            self.add_light(LightData {
                position: glm::vec3(light.position[0], light.position[1], light.position[2]),
                direction: glm::vec3(light.direction[0], light.direction[1], light.direction[2]),
                color: glm::vec3(light.color[0], light.color[1], light.color[2]),
                intensity: light.intensity,
                radius: light.range,
                light_type,
            });
        }

        object_count
    }

    /// Returns a mutable iterator over all lights.
    pub fn iter_lights_mut(
        &mut self,
    ) -> impl Iterator<
        Item = (
            i3_renderer::scene::LightId,
            &mut i3_renderer::scene::LightData,
        ),
    > {
        self.lights.iter_mut().map(|(id, data)| (*id, data))
    }
}

impl SceneProvider for BasicScene {
    fn object_count(&self) -> usize {
        self.objects.len()
    }

    fn iter_objects(&self) -> Box<dyn Iterator<Item = (ObjectId, &ObjectData)> + '_> {
        Box::new(self.objects.iter().map(|(id, data)| (*id, data)))
    }

    fn iter_dirty_objects(&self) -> Box<dyn Iterator<Item = (ObjectId, &ObjectData)> + '_> {
        Box::new(
            self.objects
                .iter()
                .filter(|(id, _)| self.dirty_objects.contains(id))
                .map(|(id, data)| (*id, data)),
        )
    }

    fn material_count(&self) -> usize {
        self.materials.len()
    }

    fn iter_materials(&self) -> Box<dyn Iterator<Item = (MaterialId, &MaterialData)> + '_> {
        Box::new(self.materials.iter().map(|(id, data)| (*id, data)))
    }

    fn light_count(&self) -> usize {
        self.lights.len()
    }

    fn iter_lights(&self) -> Box<dyn Iterator<Item = (LightId, &LightData)> + '_> {
        Box::new(self.lights.iter().map(|(id, data)| (*id, data)))
    }

    fn mesh(&self, id: u32) -> &Mesh {
        &self.meshes[id as usize]
    }

    fn mesh_descriptor_count(&self) -> usize {
        self.meshes.len()
    }

    fn iter_mesh_descriptors<'a>(
        &'a self,
        backend: &'a dyn RenderBackend,
    ) -> Box<dyn Iterator<Item = (u32, GpuMeshDescriptor)> + 'a> {
        Box::new(self.meshes.iter().enumerate().map(|(i, m)| {
            let desc = GpuMeshDescriptor {
                vertex_buffer_address: backend.get_buffer_device_address(m.vertex_buffer),
                index_buffer_address: backend.get_buffer_device_address(m.index_buffer),
                index_count: m.index_count,
                vertex_stride: m.stride,
                first_index: 0,
                vertex_offset: 0,
                aabb_min: [0.0; 3],
                // 2 bytes per index for U16, 4 bytes for U32 — used by the GBuffer shader
                // to correctly unpack BDA index reads (avoid treating two U16 as one U32).
                index_stride: if m.index_type == IndexType::Uint16 {
                    2
                } else {
                    4
                },
                aabb_max: [0.0; 3],
                _pad1: 0.0,
            };
            (i as u32, desc)
        }))
    }

    fn iter_dirty_mesh_descriptors<'a>(
        &'a self,
        backend: &'a dyn RenderBackend,
    ) -> Box<dyn Iterator<Item = (u32, GpuMeshDescriptor)> + 'a> {
        self.iter_mesh_descriptors(backend)
    }

    fn iter_instances(&self) -> Box<dyn Iterator<Item = GpuInstanceData> + '_> {
        Box::new(self.objects.iter().map(|(_, obj)| GpuInstanceData {
            world_transform: obj.world_transform,
            prev_transform: obj.prev_transform,
            mesh_idx: obj.mesh_id,
            material_id: obj.material_id,
            flags: 0,
            _pad: 0,
            world_aabb_min: [0.0; 3],
            _pad2: 0.0,
            world_aabb_max: [0.0; 3],
            _pad3: 0.0,
        }))
    }
}

/// Generates a white unit cube: 24 vertices (4 per face), 36 indices (CCW winding).
fn generate_white_cube() -> (Vec<[f32; 12]>, Vec<u16>) {
    let (mut vertices, indices) = generate_cube();
    for v in &mut vertices {
        // v[3..6] is normal, v[6..8] is UV, v[8..12] is tangent
        v[6] = 0.0; // U
        v[7] = 0.0; // V (just placeholders for white cube)
    }
    (vertices, indices)
}

/// Generates a unit cube: 24 vertices (4 per face), 36 indices (CCW winding).
fn generate_cube() -> (Vec<[f32; 12]>, Vec<u16>) {
    #[rustfmt::skip]
    let vertices: Vec<[f32; 12]> = vec![
        // Front face (Z+)
        [-0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  0.0, 1.0,  1.0, 0.0, 0.0, 1.0],
        [ 0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 1.0,  1.0, 0.0, 0.0, 1.0],
        [ 0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 0.0,  1.0, 0.0, 0.0, 1.0],
        [-0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  0.0, 0.0,  1.0, 0.0, 0.0, 1.0],
        // Back face (Z-)
        [ 0.5, -0.5, -0.5,  0.0, 0.0,-1.0,  0.0, 1.0, -1.0, 0.0, 0.0, 1.0],
        [-0.5, -0.5, -0.5,  0.0, 0.0,-1.0,  1.0, 1.0, -1.0, 0.0, 0.0, 1.0],
        [-0.5,  0.5, -0.5,  0.0, 0.0,-1.0,  1.0, 0.0, -1.0, 0.0, 0.0, 1.0],
        [ 0.5,  0.5, -0.5,  0.0, 0.0,-1.0,  0.0, 0.0, -1.0, 0.0, 0.0, 1.0],
        // Top face (Y+)
        [-0.5,  0.5,  0.5,  0.0, 1.0, 0.0,  0.0, 1.0,  1.0, 0.0, 0.0, 1.0],
        [ 0.5,  0.5,  0.5,  0.0, 1.0, 0.0,  1.0, 1.0,  1.0, 0.0, 0.0, 1.0],
        [ 0.5,  0.5, -0.5,  0.0, 1.0, 0.0,  1.0, 0.0,  1.0, 0.0, 0.0, 1.0],
        [-0.5,  0.5, -0.5,  0.0, 1.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0, 1.0],
        // Bottom face (Y-)
        [-0.5, -0.5, -0.5,  0.0,-1.0, 0.0,  0.0, 1.0,  1.0, 0.0, 0.0, 1.0],
        [ 0.5, -0.5, -0.5,  0.0,-1.0, 0.0,  1.0, 1.0,  1.0, 0.0, 0.0, 1.0],
        [ 0.5, -0.5,  0.5,  0.0,-1.0, 0.0,  1.0, 0.0,  1.0, 0.0, 0.0, 1.0],
        [-0.5, -0.5,  0.5,  0.0,-1.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0, 1.0],
        // Right face (X+)
        [ 0.5, -0.5,  0.5,  1.0, 0.0, 0.0,  0.0, 1.0,  0.0, 0.0, -1.0, 1.0],
        [ 0.5, -0.5, -0.5,  1.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.0, -1.0, 1.0],
        [ 0.5,  0.5, -0.5,  1.0, 0.0, 0.0,  1.0, 0.0,  0.0, 0.0, -1.0, 1.0],
        [ 0.5,  0.5,  0.5,  1.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, -1.0, 1.0],
        // Left face (X-)
        [-0.5, -0.5, -0.5, -1.0, 0.0, 0.0,  0.0, 1.0,  0.0, 0.0, 1.0, 1.0],
        [-0.5, -0.5,  0.5, -1.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.0, 1.0, 1.0],
        [-0.5,  0.5,  0.5, -1.0, 0.0, 0.0,  1.0, 0.0,  0.0, 0.0, 1.0, 1.0],
        [-0.5,  0.5, -0.5, -1.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 1.0, 1.0],
    ];

    #[rustfmt::skip]
    let indices: Vec<u16> = vec![
         0,  2,  1,  0,  3,  2, // Front
         4,  6,  5,  4,  7,  6, // Back
         8, 10,  9,  8, 11, 10, // Top
        12, 14, 13, 12, 15, 14, // Bottom
        16, 18, 17, 16, 19, 18, // Right
        20, 22, 21, 20, 23, 22, // Left
    ];

    (vertices, indices)
}
