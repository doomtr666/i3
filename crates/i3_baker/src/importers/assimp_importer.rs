use crate::pipeline::{BakeContext, BakeOutput, Extractor, ImportedData, Importer};
use crate::Result;
use bytemuck::bytes_of;
use i3_io::mesh::{BoundingBox, IndexFormat, MeshHeader, VertexFormat};
use i3_io::scene_asset::{ObjectInstance, SceneHeader};
use i3_io::material::{MaterialHeader, MATERIAL_ASSET_TYPE};
use nalgebra_glm::Mat4;
use std::any::Any;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// UUID type for i3mesh assets.
const MESH_TYPE_UUID: Uuid = i3_io::mesh::MESH_ASSET_TYPE;

/// UUID type for i3scene assets.
const SCENE_TYPE_UUID: Uuid = i3_io::scene_asset::SCENE_ASSET_TYPE;

/// Extracted material data (Send + Sync compatible).
#[derive(Debug, Clone)]
pub struct ExtractedMaterial {
    pub name: String,
    pub albedo_path: Option<PathBuf>,
    pub normal_path: Option<PathBuf>,
    pub metallic_roughness_path: Option<PathBuf>,
    pub emissive_path: Option<PathBuf>,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: [f32; 3],
    pub alpha_cutoff: f32,
}

/// Extracted mesh data (Send + Sync compatible).
#[derive(Debug, Clone)]
pub struct ExtractedMesh {
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
    pub has_normals: bool,
    pub has_uvs: bool,
    pub has_colors: bool,
    pub material_index: Option<usize>,
}

/// Intermediate data from Assimp import (Send + Sync compatible).
pub struct AssimpScene {
    /// Source file path.
    pub source_path: PathBuf,
    /// Extracted meshes.
    pub meshes: Vec<ExtractedMesh>,
    /// Extracted materials.
    pub materials: Vec<ExtractedMaterial>,
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
    extractors: Vec<Box<dyn Extractor>>,
}

impl AssimpImporter {
    pub fn new() -> Self {
        Self {
            extractors: vec![
                Box::new(MeshExtractor),
                Box::new(SceneExtractor),
                Box::new(MaterialExtractor),
            ],
        }
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
            "gltf", "glb", "fbx", "obj", "dae", "3ds", "blend", "ply", "stl",
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
                russimp::scene::PostProcess::FlipWindingOrder,
            ],
        )
        .map_err(|e| crate::BakerError::Plugin(format!("Assimp import error: {:?}", e)))?;

        let materials = extract_materials(&scene, source_path);
        let meshes = extract_meshes(&scene);

        Ok(Box::new(AssimpScene {
            source_path: source_path.to_path_buf(),
            meshes,
            materials,
            mesh_count: scene.meshes.len(),
        }))
    }

    fn extract(&self, data: &dyn ImportedData, ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let assimp_data = data
            .as_any()
            .downcast_ref::<AssimpScene>()
            .ok_or_else(|| crate::BakerError::Pipeline("Invalid imported data type".to_string()))?;

        println!("AssimpImporter: Extracting from {}, meshes={}, materials={}", 
            assimp_data.source_path.display(), assimp_data.meshes.len(), assimp_data.materials.len());

        let mut outputs = Vec::new();
        for extractor in &self.extractors {
            let extracted = extractor.extract(assimp_data, ctx)?;
            println!("Extractor {}: produced {} outputs", extractor.name(), extracted.len());
            outputs.extend(extracted);
        }
        Ok(outputs)
    }
}

fn extract_meshes(scene: &russimp::scene::Scene) -> Vec<ExtractedMesh> {
    scene.meshes.iter().map(|mesh| {
        let has_normals = !mesh.normals.is_empty();
        let has_uvs = !mesh.texture_coords.is_empty() && mesh.texture_coords[0].is_some();
        let has_colors = !mesh.colors.is_empty() && mesh.colors[0].is_some();

        let mut vertices = Vec::with_capacity(mesh.vertices.len() * 9);
        for i in 0..mesh.vertices.len() {
            vertices.push(mesh.vertices[i].x);
            vertices.push(mesh.vertices[i].y);
            vertices.push(mesh.vertices[i].z);

            if has_normals {
                vertices.push(mesh.normals[i].x);
                vertices.push(mesh.normals[i].y);
                vertices.push(mesh.normals[i].z);
            } else {
                vertices.push(0.0); vertices.push(0.0); vertices.push(1.0);
            }

            if has_colors {
                let c = &mesh.colors[0].as_ref().unwrap()[i];
                vertices.push(c.r); vertices.push(c.g); vertices.push(c.b);
            } else {
                vertices.push(1.0); vertices.push(1.0); vertices.push(1.0);
            }
        }

        let mut indices = Vec::new();
        for face in &mesh.faces {
            if face.0.len() == 3 {
                indices.push(face.0[0]);
                indices.push(face.0[1]);
                indices.push(face.0[2]);
            }
        }

        ExtractedMesh {
            vertices,
            indices,
            has_normals,
            has_uvs,
            has_colors,
            material_index: Some(mesh.material_index as usize),
        }
    }).collect()
}

fn extract_materials(scene: &russimp::scene::Scene, _source_path: &Path) -> Vec<ExtractedMaterial> {
    scene.materials.iter().enumerate().map(|(idx, mat)| {
        let name = mat.properties.iter()
            .find(|p| p.key == "?mat.name" || p.key == "name")
            .and_then(|p| match &p.data {
                russimp::material::PropertyTypeInfo::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| format!("material_{}", idx));

        let mut albedo = None;
        let mut normal = None;

        if let Some(tex) = mat.textures.get(&russimp::material::TextureType::Diffuse) {
            albedo = Some(PathBuf::from(&tex.borrow().filename));
        } else if let Some(tex) = mat.textures.get(&russimp::material::TextureType::BaseColor) {
            albedo = Some(PathBuf::from(&tex.borrow().filename));
        }

        if let Some(tex) = mat.textures.get(&russimp::material::TextureType::Normals) {
            normal = Some(PathBuf::from(&tex.borrow().filename));
        }

        ExtractedMaterial {
            name,
            albedo_path: albedo,
            normal_path: normal,
            metallic_roughness_path: None,
            emissive_path: None,
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            emissive_factor: [0.0, 0.0, 0.0],
            alpha_cutoff: 0.5,
        }
    }).collect()
}

pub struct MeshExtractor;
impl Extractor for MeshExtractor {
    fn name(&self) -> &str { "MeshExtractor" }
    fn output_type(&self) -> Uuid { MESH_TYPE_UUID }
    fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let assimp_data = data.as_any().downcast_ref::<AssimpScene>().unwrap();
        let namespace = Uuid::new_v5(&Uuid::NAMESPACE_OID, assimp_data.source_path.to_string_lossy().as_bytes());
        let mut outputs = Vec::new();
        for i in 0..assimp_data.meshes.len() {
            outputs.push(build_mesh_output(assimp_data, i, namespace)?);
        }
        Ok(outputs)
    }
}

fn build_mesh_output(assimp_data: &AssimpScene, mesh_idx: usize, namespace: Uuid) -> Result<BakeOutput> {
    let mesh = &assimp_data.meshes[mesh_idx];
    let file_stem = assimp_data.source_path.file_stem().unwrap().to_string_lossy();
    let name = format!("{}_mesh_{}", file_stem, mesh_idx);
    let asset_id = Uuid::new_v5(&namespace, name.as_bytes());

    let vertex_format = VertexFormat::POSITION_NORMAL_COLOR;
    let stride = vertex_format.stride();
    let vertex_data: Vec<u8> = mesh.vertices.iter().flat_map(|f| f.to_ne_bytes()).collect();

    let max_index = mesh.indices.iter().copied().max().unwrap_or(0);
    let (index_format, index_data): (IndexFormat, Vec<u8>) = if max_index > u16::MAX as u32 {
        (IndexFormat::U32, mesh.indices.iter().flat_map(|i| i.to_ne_bytes()).collect())
    } else {
        (IndexFormat::U16, mesh.indices.iter().map(|&i| i as u16).flat_map(|i| i.to_ne_bytes()).collect())
    };

    let bounds = calculate_bounds(&mesh.vertices, stride as usize / 4);
    let header = MeshHeader {
        vertex_count: (mesh.vertices.len() / (stride as usize / 4)) as u32,
        index_count: mesh.indices.len() as u32,
        vertex_stride: stride,
        index_format,
        vertex_format,
        vertex_offset: std::mem::size_of::<MeshHeader>() as u32,
        index_offset: (std::mem::size_of::<MeshHeader>() + vertex_data.len()) as u32,
        bounds_offset: (std::mem::size_of::<MeshHeader>() + vertex_data.len() + index_data.len()) as u32,
        skeleton_id: [0u8; 16],
        material_id: get_material_id(assimp_data, mesh.material_index).into_bytes(),
    };

    let mut data = Vec::new();
    data.extend_from_slice(bytes_of(&header));
    data.extend_from_slice(&vertex_data);
    data.extend_from_slice(&index_data);
    data.extend_from_slice(bytes_of(&bounds));

    Ok(BakeOutput { asset_id, asset_type: MESH_TYPE_UUID, data, name })
}

fn get_material_id(scene: &AssimpScene, material_idx: Option<usize>) -> Uuid {
    if let Some(idx) = material_idx {
        if idx < scene.materials.len() {
            return Uuid::new_v5(&Uuid::NAMESPACE_OID, scene.materials[idx].name.as_bytes());
        }
    }
    Uuid::nil()
}

fn calculate_bounds(vertices: &[f32], stride_floats: usize) -> BoundingBox {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    let count = if stride_floats > 0 { vertices.len() / stride_floats } else { 0 };
    for i in 0..count {
        let o = i * stride_floats;
        for c in 0..3 {
            min[c] = min[c].min(vertices[o + c]);
            max[c] = max[c].max(vertices[o + c]);
        }
    }
    BoundingBox { min, max }
}

pub struct MaterialExtractor;
impl Extractor for MaterialExtractor {
    fn name(&self) -> &str { "MaterialExtractor" }
    fn output_type(&self) -> Uuid { MATERIAL_ASSET_TYPE }
    fn extract(&self, data: &dyn ImportedData, ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let assimp_data = data.as_any().downcast_ref::<AssimpScene>().unwrap();
        let mut outputs = Vec::new();

        for mat in &assimp_data.materials {
            let asset_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, mat.name.as_bytes());
            
            // Texture UUIDs
            let albedo_id = bake_texture(mat.albedo_path.as_ref(), assimp_data, ctx, true, &mut outputs)?;
            let normal_id = bake_texture(mat.normal_path.as_ref(), assimp_data, ctx, false, &mut outputs)?;

            let header = MaterialHeader {
                albedo_texture: albedo_id.into_bytes(),
                normal_texture: normal_id.into_bytes(),
                metallic_roughness_texture: [0u8; 16],
                emissive_texture: [0u8; 16],
                base_color_factor: mat.base_color_factor,
                metallic_factor: mat.metallic_factor,
                roughness_factor: mat.roughness_factor,
                emissive_factor: mat.emissive_factor,
                alpha_cutoff: mat.alpha_cutoff,
                _padding: [0u8; 20],
            };

            outputs.push(BakeOutput {
                asset_id,
                asset_type: MATERIAL_ASSET_TYPE,
                data: bytes_of(&header).to_vec(),
                name: mat.name.clone(),
            });
        }
        Ok(outputs)
    }
}

fn bake_texture(
    path: Option<&PathBuf>,
    _scene: &AssimpScene,
    ctx: &BakeContext,
    is_srgb: bool,
    outputs: &mut Vec<BakeOutput>
) -> Result<Uuid> {
    let path = match path {
        Some(p) => p,
        None => return Ok(Uuid::nil()),
    };

    let source_dir = ctx.source_path.parent().unwrap();
    let full_path = source_dir.join(path);
    
    if !full_path.exists() {
        println!("AssimpImporter: Texture NOT FOUND at {:?} (tried joining {:?} and {:?})", 
            full_path, source_dir, path);
        return Ok(Uuid::nil());
    }

    let asset_id = Uuid::new_v5(&Uuid::NAMESPACE_URL, full_path.to_string_lossy().as_bytes());
    if outputs.iter().any(|o| o.asset_id == asset_id) {
        return Ok(asset_id);
    }

    println!("AssimpImporter: Baking texture {:?} -> ID={:?}", full_path, asset_id);

    use crate::importers::image_importer::{ImageImporter, TextureImportOptions};
    use i3_io::texture::TextureFormat;

    let mut format = if is_srgb { TextureFormat::BC7_SRGB } else { TextureFormat::BC7_UNORM };
    let path_lc = path.to_string_lossy().to_lowercase();
    if path_lc.contains("normal") || path_lc.contains("norm") {
        format = TextureFormat::BC5_UNORM;
    }

    let importer = ImageImporter::new(TextureImportOptions {
        is_srgb,
        generate_mips: true,
        format,
    });

    let imported = importer.import(&full_path)?;
    let texture_outputs = importer.extract(imported.as_ref(), ctx)?;
    outputs.extend(texture_outputs);

    Ok(asset_id)
}

pub struct SceneExtractor;
impl Extractor for SceneExtractor {
    fn name(&self) -> &str { "SceneExtractor" }
    fn output_type(&self) -> Uuid { SCENE_TYPE_UUID }
    fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let assimp_data = data.as_any().downcast_ref::<AssimpScene>().unwrap();
        let mut objects = Vec::new();
        let mut mesh_refs = Vec::new();
        let mut string_table = Vec::new();

        let namespace = Uuid::new_v5(&Uuid::NAMESPACE_OID, assimp_data.source_path.to_string_lossy().as_bytes());
        let file_stem = assimp_data.source_path.file_stem().unwrap().to_string_lossy();

        let mut scene_bounds = BoundingBox::empty();

        for mesh_idx in 0..assimp_data.meshes.len() {
            let mesh_name = format!("{}_mesh_{}", file_stem, mesh_idx);
            let mesh_id = Uuid::new_v5(&namespace, mesh_name.as_bytes());
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

            let mesh_data = &assimp_data.meshes[mesh_idx];
            let mesh_bounds = calculate_bounds(&mesh_data.vertices, VertexFormat::POSITION_NORMAL_COLOR.stride() as usize / 4);
            scene_bounds.merge(&mesh_bounds);
        }

        let header = SceneHeader {
            object_count: objects.len() as u32,
            light_count: 0,
            mesh_ref_count: mesh_refs.len() as u32,
            skeleton_ref_count: 0,
            objects_offset: std::mem::size_of::<SceneHeader>() as u32,
            lights_offset: 0,
            mesh_refs_offset: std::mem::size_of::<SceneHeader>() as u32 + (objects.len() * std::mem::size_of::<ObjectInstance>()) as u32,
            skeleton_refs_offset: 0,
            strings_offset: std::mem::size_of::<SceneHeader>() as u32 + (objects.len() * std::mem::size_of::<ObjectInstance>()) as u32 + (mesh_refs.len() * 16) as u32,
            strings_size: string_table.len() as u32,
            bounds: scene_bounds,
            _reserved: [0u8; 16],
        };

        let mut binary = Vec::new();
        binary.extend_from_slice(bytes_of(&header));
        for obj in &objects { binary.extend_from_slice(bytes_of(obj)); }
        for id in &mesh_refs { binary.extend_from_slice(id.as_bytes()); }
        binary.extend_from_slice(&string_table);

        Ok(vec![BakeOutput {
            asset_id: Uuid::new_v5(&namespace, format!("{}_scene", file_stem).as_bytes()),
            asset_type: SCENE_TYPE_UUID,
            data: binary,
            name: format!("{}_scene", file_stem),
        }])
    }
}

fn glm_to_array(m: &Mat4) -> [[f32; 4]; 4] {
    let mut r = [[0.0; 4]; 4];
    for i in 0..4 { for j in 0..4 { r[i][j] = m[(i, j)]; } }
    r
}
