use nalgebra_glm::Mat4;

/// Unique identifier for a renderable object in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectId(pub u64);

/// Unique identifier for a light in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LightId(pub u64);

/// GPU-ready data for a single renderable object.
#[derive(Debug, Clone)]
pub struct ObjectData {
    pub world_transform: Mat4,
    pub prev_transform: Mat4,
    pub material_id: u32,
    pub mesh_id: u32,
}

/// Type of light source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightType {
    Point,
    Directional,
    Spot,
}

/// GPU-ready data for a single light.
#[derive(Debug, Clone)]
pub struct LightData {
    pub position: nalgebra_glm::Vec3,
    pub direction: nalgebra_glm::Vec3,
    pub color: nalgebra_glm::Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub light_type: LightType,
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

    /// Iterate only objects that changed since last frame (delta upload).
    fn iter_dirty_objects(&self) -> Box<dyn Iterator<Item = (ObjectId, &ObjectData)> + '_>;

    /// Total number of active lights.
    fn light_count(&self) -> usize;

    /// Iterate all lights.
    fn iter_lights(&self) -> Box<dyn Iterator<Item = (LightId, &LightData)> + '_>;
}
