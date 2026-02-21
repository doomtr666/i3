use std::collections::HashSet;

use i3_gfx::prelude::*;
use i3_renderer::scene::{
    LightData, LightId, LightType, Mesh, ObjectData, ObjectId, SceneProvider,
};
use nalgebra_glm as glm;

/// A simple in-memory scene for examples and integration tests.
///
/// Owns GPU-resident meshes (VB/IB handles), objects, and lights.
/// Implements `SceneProvider` so it can be passed directly to the renderer.
pub struct BasicScene {
    meshes: Vec<Mesh>,
    objects: Vec<(ObjectId, ObjectData)>,
    lights: Vec<(LightId, LightData)>,
    dirty: HashSet<ObjectId>,
    next_object_id: u64,
    next_light_id: u64,
}

impl BasicScene {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            objects: Vec::new(),
            lights: Vec::new(),
            dirty: HashSet::new(),
            next_object_id: 0,
            next_light_id: 0,
        }
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

    /// Convenience: adds a default directional light.
    pub fn add_default_light(&mut self) -> LightId {
        self.add_light(LightData {
            position: glm::vec3(0.0, 0.0, 0.0),
            direction: glm::normalize(&glm::vec3(-1.0, -1.0, -1.0)),
            color: glm::vec3(1.0, 1.0, 1.0),
            intensity: 1.0,
            radius: 0.0,
            light_type: LightType::Directional,
        })
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

/// Generates a unit cube: 24 vertices (4 per face), 36 indices (CCW winding).
///
/// Each vertex is `[f32; 9]` = position(3) + normal(3) + color(3).
/// Each face has a distinct color for debug visualization.
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
