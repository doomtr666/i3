use i3_gfx::prelude::IndexType;
use nalgebra_glm::Mat4;

/// Unique identifier for a renderable object in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectId(pub u64);

/// Unique identifier for a light in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LightId(pub u64);

/// Unique identifier for a material in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

#[derive(Debug, Clone)]
#[repr(C)]
pub struct ObjectData {
    pub world_transform: Mat4,
    pub prev_transform: Mat4,
    pub material_id: u32,
    pub mesh_id: u32,
    pub flags: u32,
    pub _pad: u32,
}

/// GPU-ready data for a material.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct MaterialData {
    pub base_color_factor: [f32; 4],
    pub emissive_factor_and_alpha_cutoff: [f32; 4],
    // PBR factors (metallic, roughness, pad, pad) - 16 bytes
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub _pad_pbr: [f32; 2],
    // Texture indices (albedo, normal, rmao, emissive) - 16 bytes
    pub albedo_tex_index: i32,
    pub normal_tex_index: i32,
    pub rmao_tex_index: i32,
    pub emissive_tex_index: i32,
}

/// Type of light source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Point,
    Directional,
    Spot,
}

/// GPU-ready data for a single light.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LightData {
    pub position: nalgebra_glm::Vec3,
    pub direction: nalgebra_glm::Vec3,
    pub color: nalgebra_glm::Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub light_type: LightType,
}

/// GPU-resident mesh description as expected by the GPU-driven pipeline.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GpuMeshDescriptor {
    pub vertex_buffer_address: u64,
    pub index_buffer_address: u64,
    pub index_count: u32,
    pub vertex_stride: u32,
    pub first_index: u32, // renamed from index_offset for clarity
    pub vertex_offset: i32,
    pub aabb_min: [f32; 3],
    /// Byte stride of one index entry: 2 for IndexFormat::U16, 4 for U32.
    /// Used by the GBuffer shader to read BDA index data correctly.
    pub index_stride: u32,
    pub aabb_max: [f32; 3],
    pub _pad1: f32,
}

/// GPU-resident instance data for the GPU-driven pipeline.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GpuInstanceData {
    pub world_transform: Mat4,
    pub prev_transform: Mat4,
    pub mesh_idx: u32, // index in MeshDescriptorBuffer
    pub material_id: u32,
    pub flags: u32,
    pub _pad: u32,
    pub world_aabb_min: [f32; 3],
    pub _pad2: f32,
    pub world_aabb_max: [f32; 3],
    pub _pad3: f32,
}

/// A GPU-resident mesh. Carries buffer handles only, no CPU vertex data.
///
/// The actual upload mechanism is an implementation detail of the
/// `SceneProvider` — could be CPU upload, Direct Storage, or streaming.
#[derive(Debug, Clone, Copy)]
pub struct Mesh {
    pub vertex_buffer: i3_gfx::prelude::BackendBuffer,
    pub index_buffer: i3_gfx::prelude::BackendBuffer,
    pub index_count: u32,
    pub index_type: IndexType,
    pub stride: u32,
}

/// Trait that the application (or ECS bridge) implements to feed
/// scene data to the renderer's GPU sync passes.
///
/// The renderer never owns the scene — it observes it through this trait.
/// This enables three integration patterns:
/// - **Standalone**: App manages objects directly (examples, tools)
/// - **ECS bridge**: Thin adapter over an ECS world
/// - **Streaming**: AssetLoader surfaces newly-loaded meshes as dirty objects
pub trait SceneProvider {
    /// Total number of active objects.
    fn object_count(&self) -> usize;

    /// Iterate all objects (full upload, used on first frame or reset).
    fn iter_objects(&self) -> Box<dyn Iterator<Item = (ObjectId, &ObjectData)> + '_>;

    fn iter_dirty_objects(&self) -> Box<dyn Iterator<Item = (ObjectId, &ObjectData)> + '_>;

    /// Total number of materials.
    fn material_count(&self) -> usize;

    /// Iterate all materials.
    fn iter_materials(&self) -> Box<dyn Iterator<Item = (MaterialId, &MaterialData)> + '_>;

    /// Total number of active lights.
    fn light_count(&self) -> usize;

    /// Iterate all lights.
    fn iter_lights(&self) -> Box<dyn Iterator<Item = (LightId, &LightData)> + '_>;

    /// Returns the primary directional light (the "sun").
    fn sun(&self) -> LightData {
        self.iter_lights()
            .find(|(_, l)| l.light_type == LightType::Directional)
            .map(|(_, l)| l.clone())
            .unwrap_or(LightData {
                position: nalgebra_glm::vec3(0.0, 0.0, 0.0),
                direction: nalgebra_glm::vec3(0.0, -1.0, 0.0),
                color: nalgebra_glm::vec3(1.0, 0.9, 0.8),
                intensity: 1.0,
                radius: 0.0,
                light_type: LightType::Directional,
            })
    }

    /// Access a GPU-resident mesh by ID.
    fn mesh(&self, id: u32) -> &Mesh;

    // --- GPU-Driven Extensions ---

    /// Number of mesh descriptors registered.
    fn mesh_descriptor_count(&self) -> usize;

    /// Iterate all mesh descriptors (initial upload).
    fn iter_mesh_descriptors<'a>(
        &'a self,
        backend: &'a dyn i3_gfx::graph::backend::RenderBackend,
    ) -> Box<dyn Iterator<Item = (u32, GpuMeshDescriptor)> + 'a>;

    /// Iterate only newly registered mesh descriptors this frame.
    fn iter_dirty_mesh_descriptors<'a>(
        &'a self,
        backend: &'a dyn i3_gfx::graph::backend::RenderBackend,
    ) -> Box<dyn Iterator<Item = (u32, GpuMeshDescriptor)> + 'a>;

    /// Iterate all instances for the scene.
    fn iter_instances(&self) -> Box<dyn Iterator<Item = GpuInstanceData> + '_>;

    /// Instance count total (= object_count alias).
    fn instance_count(&self) -> usize {
        self.object_count()
    }
}
