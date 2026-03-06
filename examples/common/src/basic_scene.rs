use std::collections::{HashMap, HashSet};

use i3_gfx::prelude::*;
use i3_io::mesh::MeshAsset;
use i3_io::scene_asset::{LightType as AssetLightType, SceneAsset};
use i3_renderer::scene::{
    LightData, LightId, LightType, Mesh, ObjectData, ObjectId, SceneProvider,
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
    objects: Vec<(ObjectId, ObjectData)>,
    lights: Vec<(LightId, LightData)>,
    dirty: HashSet<ObjectId>,
    next_object_id: u64,
    next_light_id: u64,
    bounds: i3_io::mesh::BoundingBox,
}

impl BasicScene {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            mesh_uuid_to_index: HashMap::new(),
            objects: Vec::new(),
            lights: Vec::new(),
            dirty: HashSet::new(),
            next_object_id: 0,
            next_light_id: 0,
            bounds: i3_io::mesh::BoundingBox::empty(),
        }
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
            usage: BufferUsageFlags::VERTEX_BUFFER,
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
            usage: BufferUsageFlags::INDEX_BUFFER,
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
        });

        let _ = vertex_count; // Reserved for future validation
        id
    }

    /// Adds an object to the scene, returns its ID.
    pub fn add_object(&mut self, data: ObjectData) -> ObjectId {
        let id = ObjectId(self.next_object_id);
        self.next_object_id += 1;
        self.dirty.insert(id);
        self.objects.push((id, data));
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
            self.dirty.insert(id);
        }
    }

    /// Clears the dirty set. Call after the renderer has consumed the deltas.
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
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
                vertices.len() * std::mem::size_of::<[f32; 9]>(),
            )
        };
        self.add_mesh(backend, vb_bytes, vertices.len() as u32, &indices)
    }

    /// Convenience: creates a white unit cube mesh and returns its MeshId.
    pub fn add_white_cube_mesh(&mut self, backend: &mut dyn RenderBackend) -> u32 {
        let (vertices, indices) = generate_white_cube();
        let vb_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * std::mem::size_of::<[f32; 9]>(),
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

        // Backlight (Cool blue / softer)
        self.add_light(LightData {
            position: glm::vec3(0.0, 0.0, 0.0),
            direction: glm::normalize(&glm::vec3(1.0, 0.5, 1.0)),
            color: glm::vec3(0.2, 0.4, 1.0),
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
        mesh: &MeshAsset,
        mesh_uuid: Uuid,
    ) -> u32 {
        // Create vertex buffer
        let vb = backend.create_buffer(&BufferDesc {
            size: mesh.vertex_data.len() as u64,
            usage: BufferUsageFlags::VERTEX_BUFFER,
            memory: MemoryType::CpuToGpu,
        });
        backend
            .upload_buffer(vb, &mesh.vertex_data, 0)
            .expect("Failed to upload baked mesh vertices");

        // Create index buffer
        let ib = backend.create_buffer(&BufferDesc {
            size: mesh.index_data.len() as u64,
            usage: BufferUsageFlags::INDEX_BUFFER,
            memory: MemoryType::CpuToGpu,
        });
        backend
            .upload_buffer(ib, &mesh.index_data, 0)
            .expect("Failed to upload baked mesh indices");

        let id = self.meshes.len() as u32;
        let index_type = match mesh.header.index_format {
            i3_io::mesh::IndexFormat::U16 => IndexType::Uint16,
            i3_io::mesh::IndexFormat::U32 => IndexType::Uint32,
            _ => IndexType::Uint16,
        };

        self.meshes.push(Mesh {
            vertex_buffer: vb,
            index_buffer: ib,
            index_count: mesh.header.index_count,
            index_type,
        });

        // Register UUID mapping
        self.mesh_uuid_to_index.insert(mesh_uuid, id);

        id
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
            let mesh_id = scene
                .mesh_for_object(obj)
                .and_then(|uuid| self.mesh_uuid_to_index.get(&uuid).copied())
                .unwrap_or(0); // Default to first mesh if not found

            // Convert transform from [[f32; 4]; 4] to glm::Mat4
            let transform = glm::Mat4::from(obj.transform);

            let object_id = self.add_object(ObjectData {
                world_transform: transform,
                prev_transform: transform,
                mesh_id,
                material_id: 0, // Default material
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
                .filter(|(id, _)| self.dirty.contains(id))
                .map(|(id, data)| (*id, data)),
        )
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
}

/// Generates a white unit cube: 24 vertices (4 per face), 36 indices (CCW winding).
fn generate_white_cube() -> (Vec<[f32; 9]>, Vec<u16>) {
    let (mut vertices, indices) = generate_cube();
    for v in &mut vertices {
        v[6] = 1.0; // R
        v[7] = 1.0; // G
        v[8] = 1.0; // B
    }
    (vertices, indices)
}

/// Generates a unit cube: 24 vertices (4 per face), 36 indices (CCW winding).
fn generate_cube() -> (Vec<[f32; 9]>, Vec<u16>) {
    #[rustfmt::skip]
    let vertices: Vec<[f32; 9]> = vec![
        // Front face (Z+) — red
        [-0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0],
        [ 0.5, -0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0],
        [ 0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0],
        [-0.5,  0.5,  0.5,  0.0, 0.0, 1.0,  1.0, 0.0, 0.0],
        // Back face (Z-) — green
        [ 0.5, -0.5, -0.5,  0.0, 0.0,-1.0,  0.0, 1.0, 0.0],
        [-0.5, -0.5, -0.5,  0.0, 0.0,-1.0,  0.0, 1.0, 0.0],
        [-0.5,  0.5, -0.5,  0.0, 0.0,-1.0,  0.0, 1.0, 0.0],
        [ 0.5,  0.5, -0.5,  0.0, 0.0,-1.0,  0.0, 1.0, 0.0],
        // Top face (Y+) — blue
        [-0.5,  0.5,  0.5,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0],
        [ 0.5,  0.5,  0.5,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0],
        [ 0.5,  0.5, -0.5,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0],
        [-0.5,  0.5, -0.5,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0],
        // Bottom face (Y-) — yellow
        [-0.5, -0.5, -0.5,  0.0,-1.0, 0.0,  1.0, 1.0, 0.0],
        [ 0.5, -0.5, -0.5,  0.0,-1.0, 0.0,  1.0, 1.0, 0.0],
        [ 0.5, -0.5,  0.5,  0.0,-1.0, 0.0,  1.0, 1.0, 0.0],
        [-0.5, -0.5,  0.5,  0.0,-1.0, 0.0,  1.0, 1.0, 0.0],
        // Right face (X+) — magenta
        [ 0.5, -0.5,  0.5,  1.0, 0.0, 0.0,  1.0, 0.0, 1.0],
        [ 0.5, -0.5, -0.5,  1.0, 0.0, 0.0,  1.0, 0.0, 1.0],
        [ 0.5,  0.5, -0.5,  1.0, 0.0, 0.0,  1.0, 0.0, 1.0],
        [ 0.5,  0.5,  0.5,  1.0, 0.0, 0.0,  1.0, 0.0, 1.0],
        // Left face (X-) — cyan
        [-0.5, -0.5, -0.5, -1.0, 0.0, 0.0,  0.0, 1.0, 1.0],
        [-0.5, -0.5,  0.5, -1.0, 0.0, 0.0,  0.0, 1.0, 1.0],
        [-0.5,  0.5,  0.5, -1.0, 0.0, 0.0,  0.0, 1.0, 1.0],
        [-0.5,  0.5, -0.5, -1.0, 0.0, 0.0,  0.0, 1.0, 1.0],
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
