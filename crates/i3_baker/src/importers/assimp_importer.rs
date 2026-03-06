//! Assimp-based importer for 3D model formats.
//!
//! Supports glTF, GLB, FBX, OBJ, Collada, and many other formats
//! via the Assimp library.

use crate::Result;
use crate::pipeline::{BakeContext, BakeOutput, Extractor, ImportedData, Importer};
use bytemuck::bytes_of;
use i3_io::mesh::{BoundingBox, IndexFormat, MeshHeader, VertexFormat};
use i3_io::scene_asset::{ObjectInstance, SceneHeader};
use nalgebra_glm::Mat4;
use std::any::Any;
use std::path::Path;
use uuid::Uuid;

/// UUID type for i3mesh assets.
const MESH_TYPE_UUID: Uuid = i3_io::mesh::MESH_ASSET_TYPE;

/// UUID type for i3scene assets.
const SCENE_TYPE_UUID: Uuid = i3_io::scene_asset::SCENE_ASSET_TYPE;

/// Extracted mesh data (Send + Sync compatible).
#[derive(Debug, Clone)]
pub struct ExtractedMesh {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub has_normals: bool,
    pub has_uvs: bool,
    pub has_colors: bool,
}

/// Intermediate data from Assimp import (Send + Sync compatible).
pub struct AssimpScene {
    /// Source file path.
    pub source_path: std::path::PathBuf,
    /// Extracted meshes.
    pub meshes: Vec<ExtractedMesh>,
    /// Mesh count for scene.
    pub mesh_count: usize,
}

impl ImportedData for AssimpScene {
    fn source_path(&self) -> &Path {
        &self.source_path
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Importer for 3D model formats using Assimp.
pub struct AssimpImporter {
    /// Registered extractors.
    extractors: Vec<Box<dyn Extractor>>,
}

impl AssimpImporter {
    /// Create a new AssimpImporter with default extractors.
    pub fn new() -> Self {
        Self {
            extractors: vec![Box::new(MeshExtractor), Box::new(SceneExtractor)],
        }
    }

    /// Add a custom extractor.
    pub fn add_extractor(&mut self, extractor: Box<dyn Extractor>) {
        self.extractors.push(extractor);
    }
}

impl Default for AssimpImporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Importer for AssimpImporter {
    fn name(&self) -> &str {
        "AssimpImporter"
    }

    fn source_extensions(&self) -> &[&str] {
        &[
            "gltf", "glb",   // glTF
            "fbx",   // FBX
            "obj",   // Wavefront OBJ
            "dae",   // Collada
            "3ds",   // 3DS Max
            "blend", // Blender
            "ply",   // Stanford PLY
            "stl",   // STL
        ]
    }

    fn import(&self, source_path: &Path) -> Result<Box<dyn ImportedData>> {
        use russimp::scene::Scene;

        let path_str = source_path.to_string_lossy();

        let scene = Scene::from_file(
            &path_str,
            vec![
                russimp::scene::PostProcess::CalculateTangentSpace,
                russimp::scene::PostProcess::Triangulate,
                russimp::scene::PostProcess::JoinIdenticalVertices,
                russimp::scene::PostProcess::SortByPrimitiveType,
                russimp::scene::PostProcess::FlipUVs,
            ],
        )
        .map_err(|e| crate::BakerError::Plugin(format!("Assimp import error: {:?}", e)))?;

        // Extract meshes immediately
        let meshes = extract_meshes(&scene);
        let mesh_count = meshes.len();

        Ok(Box::new(AssimpScene {
            source_path: source_path.to_path_buf(),
            meshes,
            mesh_count,
        }))
    }

    fn extract(&self, data: &dyn ImportedData, ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let assimp_data = data
            .as_any()
            .downcast_ref::<AssimpScene>()
            .ok_or_else(|| crate::BakerError::Pipeline("Invalid imported data type".to_string()))?;

        let mut outputs = Vec::new();
        for extractor in &self.extractors {
            let extracted = extractor.extract(assimp_data, ctx)?;
            outputs.extend(extracted);
        }
        Ok(outputs)
    }
}

/// Extract meshes from Assimp scene.
fn extract_meshes(scene: &russimp::scene::Scene) -> Vec<ExtractedMesh> {
    scene
        .meshes
        .iter()
        .map(|mesh| {
            let has_normals = !mesh.normals.is_empty();
            let has_uvs = !mesh.texture_coords.is_empty() && mesh.texture_coords[0].is_some();
            let has_colors = !mesh.colors.is_empty() && mesh.colors[0].is_some();

            // Build vertex data
            let vertex_count = mesh.vertices.len();
            let floats_per_vertex = 3 // position
                + if has_normals { 3 } else { 0 }
                + if has_uvs { 2 } else { 0 }
                + if has_colors { 3 } else { 0 };

            let mut vertices = Vec::with_capacity(vertex_count * floats_per_vertex);

            for i in 0..vertex_count {
                // Position
                vertices.push(mesh.vertices[i].x);
                vertices.push(mesh.vertices[i].y);
                vertices.push(mesh.vertices[i].z);

                // Normal
                if has_normals {
                    if let Some(n) = mesh.normals.get(i) {
                        vertices.push(n.x);
                        vertices.push(n.y);
                        vertices.push(n.z);
                    } else {
                        vertices.extend_from_slice(&[0.0, 0.0, 0.0]);
                    }
                }

                // UV
                if has_uvs {
                    if let Some(ref coords) = mesh.texture_coords[0] {
                        if let Some(uv) = coords.get(i) {
                            vertices.push(uv.x);
                            vertices.push(uv.y);
                        } else {
                            vertices.extend_from_slice(&[0.0, 0.0]);
                        }
                    } else {
                        vertices.extend_from_slice(&[0.0, 0.0]);
                    }
                }

                // Color
                if has_colors {
                    if let Some(ref colors) = mesh.colors[0] {
                        if let Some(c) = colors.get(i) {
                            vertices.push(c.r);
                            vertices.push(c.g);
                            vertices.push(c.b);
                        } else {
                            vertices.extend_from_slice(&[1.0, 1.0, 1.0]);
                        }
                    } else {
                        vertices.extend_from_slice(&[1.0, 1.0, 1.0]);
                    }
                }
            }

            // Build index data
            let mut indices = Vec::new();
            for face in &mesh.faces {
                if face.0.len() == 3 {
                    indices.push(face.0[0] as u32);
                    indices.push(face.0[1] as u32);
                    indices.push(face.0[2] as u32);
                }
            }

            ExtractedMesh {
                vertices,
                indices,
                has_normals,
                has_uvs,
                has_colors,
            }
        })
        .collect()
}

/// Extracts meshes from an Assimp scene.
pub struct MeshExtractor;

impl Extractor for MeshExtractor {
    fn name(&self) -> &str {
        "MeshExtractor"
    }

    fn output_type(&self) -> Uuid {
        MESH_TYPE_UUID
    }

    fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let assimp_data = data
            .as_any()
            .downcast_ref::<AssimpScene>()
            .ok_or_else(|| crate::BakerError::Pipeline("Invalid imported data type".to_string()))?;

        let mut outputs = Vec::new();

        for (mesh_idx, mesh) in assimp_data.meshes.iter().enumerate() {
            let output = build_mesh_output(mesh, mesh_idx, &assimp_data.source_path)?;
            outputs.push(output);
        }

        Ok(outputs)
    }
}

/// Build a mesh output from extracted mesh data.
fn build_mesh_output(
    mesh: &ExtractedMesh,
    mesh_idx: usize,
    source_path: &Path,
) -> Result<BakeOutput> {
    // Determine vertex format
    let (vertex_format, stride) = if mesh.has_normals && mesh.has_uvs {
        (
            VertexFormat::POSITION_NORMAL_UV,
            VertexFormat::POSITION_NORMAL_UV.stride(),
        )
    } else {
        (
            VertexFormat::POSITION_NORMAL_COLOR,
            VertexFormat::POSITION_NORMAL_COLOR.stride(),
        )
    };

    // Convert vertices to bytes
    let vertex_data: Vec<u8> = mesh.vertices.iter().flat_map(|f| f.to_ne_bytes()).collect();

    // Determine index format
    let max_index = mesh.indices.iter().copied().max().unwrap_or(0);
    let (index_format, index_data): (IndexFormat, Vec<u8>) = if max_index > u16::MAX as u32 {
        (
            IndexFormat::U32,
            mesh.indices.iter().flat_map(|i| i.to_ne_bytes()).collect(),
        )
    } else {
        (
            IndexFormat::U16,
            mesh.indices
                .iter()
                .map(|&i| i as u16)
                .flat_map(|i| i.to_ne_bytes())
                .collect(),
        )
    };

    // Calculate bounding box
    let bounds = calculate_bounds(&mesh.vertices, stride as usize / 4);

    let vertex_count = mesh.vertices.len() as u32 / (stride as u32 / 4);
    let index_count = mesh.indices.len() as u32;

    // Build header
    let header = MeshHeader {
        vertex_count,
        index_count,
        vertex_stride: stride,
        index_format,
        vertex_format,
        vertex_offset: std::mem::size_of::<MeshHeader>() as u32,
        index_offset: std::mem::size_of::<MeshHeader>() as u32 + vertex_data.len() as u32,
        bounds_offset: std::mem::size_of::<MeshHeader>() as u32
            + vertex_data.len() as u32
            + index_data.len() as u32,
        skeleton_id: [0u8; 16],
        _reserved: [0u8; 16],
    };

    // Assemble binary
    let mut data = Vec::new();
    data.extend_from_slice(bytes_of(&header));
    data.extend_from_slice(&vertex_data);
    data.extend_from_slice(&index_data);
    data.extend_from_slice(bytes_of(&bounds));

    let asset_id = Uuid::new_v4();
    let name = format!(
        "{}_mesh_{}",
        source_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy(),
        mesh_idx
    );

    Ok(BakeOutput {
        asset_id,
        asset_type: MESH_TYPE_UUID,
        data,
        name,
    })
}

/// Calculate bounding box from vertex floats.
fn calculate_bounds(vertices: &[f32], stride_floats: usize) -> BoundingBox {
    let mut min = [f32::MAX, f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN, f32::MIN];

    let vertex_count = if stride_floats > 0 {
        vertices.len() / stride_floats
    } else {
        0
    };

    for i in 0..vertex_count {
        let offset = i * stride_floats;
        if offset + 3 <= vertices.len() {
            let x = vertices[offset];
            let y = vertices[offset + 1];
            let z = vertices[offset + 2];
            min[0] = min[0].min(x);
            min[1] = min[1].min(y);
            min[2] = min[2].min(z);
            max[0] = max[0].max(x);
            max[1] = max[1].max(y);
            max[2] = max[2].max(z);
        }
    }

    BoundingBox { min, max }
}

/// Extracts scene data from an Assimp scene.
pub struct SceneExtractor;

impl Extractor for SceneExtractor {
    fn name(&self) -> &str {
        "SceneExtractor"
    }

    fn output_type(&self) -> Uuid {
        SCENE_TYPE_UUID
    }

    fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let assimp_data = data
            .as_any()
            .downcast_ref::<AssimpScene>()
            .ok_or_else(|| crate::BakerError::Pipeline("Invalid imported data type".to_string()))?;

        let output = build_scene_output(assimp_data)?;
        Ok(vec![output])
    }
}

/// Build scene output from extracted data.
fn build_scene_output(data: &AssimpScene) -> Result<BakeOutput> {
    let mut objects = Vec::new();
    let mut mesh_refs: Vec<Uuid> = Vec::new();
    let mut string_table = Vec::new();

    // Create one object per mesh with identity transform
    for mesh_idx in 0..data.meshes.len() {
        let mesh_id = Uuid::new_v4();
        let mesh_ref_index = mesh_refs.len() as u32;
        mesh_refs.push(mesh_id);

        let name = format!("mesh_{}", mesh_idx);
        let name_offset = string_table.len() as u32;
        string_table.extend_from_slice(name.as_bytes());
        string_table.push(0);

        objects.push(ObjectInstance {
            transform: glm_to_array(&Mat4::identity()),
            mesh_ref_index,
            skeleton_ref_index: u32::MAX,
            name_offset,
            _reserved: [0u32; 3],
        });
    }

    // Build header
    let header = SceneHeader {
        object_count: objects.len() as u32,
        light_count: 0,
        mesh_ref_count: mesh_refs.len() as u32,
        skeleton_ref_count: 0,
        objects_offset: std::mem::size_of::<SceneHeader>() as u32,
        lights_offset: std::mem::size_of::<SceneHeader>() as u32
            + (objects.len() * std::mem::size_of::<ObjectInstance>()) as u32,
        mesh_refs_offset: std::mem::size_of::<SceneHeader>() as u32
            + (objects.len() * std::mem::size_of::<ObjectInstance>()) as u32,
        skeleton_refs_offset: 0,
        strings_offset: std::mem::size_of::<SceneHeader>() as u32
            + (objects.len() * std::mem::size_of::<ObjectInstance>()) as u32
            + (mesh_refs.len() * 16) as u32,
        strings_size: string_table.len() as u32,
        _reserved: [0u8; 24],
    };

    // Assemble binary
    let mut binary = Vec::new();
    binary.extend_from_slice(bytes_of(&header));

    for obj in &objects {
        binary.extend_from_slice(bytes_of(obj));
    }
    for mesh_id in &mesh_refs {
        binary.extend_from_slice(mesh_id.as_bytes());
    }
    binary.extend_from_slice(&string_table);

    let name = format!(
        "{}_scene",
        data.source_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
    );

    Ok(BakeOutput {
        asset_id: Uuid::new_v4(),
        asset_type: SCENE_TYPE_UUID,
        data: binary,
        name,
    })
}

/// Convert nalgebra Mat4 to array.
fn glm_to_array(m: &Mat4) -> [[f32; 4]; 4] {
    let mut result = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = m[(i, j)];
        }
    }
    result
}
