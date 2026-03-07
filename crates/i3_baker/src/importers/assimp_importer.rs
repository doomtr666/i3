use crate::Result;
use crate::pipeline::{BakeContext, BakeOutput, Extractor, ImportedData, Importer};
use bytemuck::bytes_of;
use i3_io::material::{MATERIAL_ASSET_TYPE, MaterialHeader};
use i3_io::mesh::{BoundingBox, IndexFormat, MeshHeader, VertexFormat};
use i3_io::scene_asset::{ObjectInstance, SceneHeader};
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
    /// Embedded textures (raw byte buffers from the glTF or FBX).
    pub embedded_textures: Vec<Vec<u8>>,
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
        use asset_importer::postprocess::PostProcessSteps;
        use asset_importer::scene::Scene;

        let clean_path = source_path
            .canonicalize()
            .unwrap_or_else(|_| source_path.to_path_buf());
        let path_str = clean_path.to_string_lossy();

        let scene = Scene::from_file_with_flags(
            path_str.as_ref(),
            PostProcessSteps::CALC_TANGENT_SPACE
                | PostProcessSteps::TRIANGULATE
                | PostProcessSteps::JOIN_IDENTICAL_VERTICES
                | PostProcessSteps::SORT_BY_PTYPE
                | PostProcessSteps::FLIP_UVS
                | PostProcessSteps::FLIP_WINDING_ORDER,
        )
        .map_err(|e| crate::BakerError::Plugin(format!("Assimp import error: {:?}", e)))?;

        let materials = extract_materials(&scene, source_path);
        let meshes = extract_meshes(&scene);

        // Extract embedded textures
        let mut embedded_textures = Vec::new();
        for tex in scene.textures() {
            if let Ok(data) = tex.data() {
                match data {
                    asset_importer::texture::TextureData::Compressed(bytes) => {
                        embedded_textures.push(bytes);
                    }
                    asset_importer::texture::TextureData::Texels(texels) => {
                        let mut bytes = Vec::with_capacity(texels.len() * 4);
                        for t in texels {
                            // asset-importerTexel has b,g,r,a
                            bytes.push(t.b);
                            bytes.push(t.g);
                            bytes.push(t.r);
                            bytes.push(t.a);
                        }
                        embedded_textures.push(bytes);
                    }
                }
            } else {
                embedded_textures.push(Vec::new());
            }
        }

        let mesh_count = scene.num_meshes();

        Ok(Box::new(AssimpScene {
            source_path: source_path.to_path_buf(),
            meshes,
            materials,
            embedded_textures,
            mesh_count,
        }))
    }

    fn extract(&self, data: &dyn ImportedData, ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let assimp_data = data
            .as_any()
            .downcast_ref::<AssimpScene>()
            .ok_or_else(|| crate::BakerError::Pipeline("Invalid imported data type".to_string()))?;

        println!(
            "cargo:warning=AssimpImporter: Extracting from {}, meshes={}, materials={}",
            assimp_data.source_path.display(),
            assimp_data.meshes.len(),
            assimp_data.materials.len()
        );

        let mut outputs = Vec::new();
        println!(
            "cargo:warning=AssimpImporter: Running {} extractors",
            self.extractors.len()
        );
        for extractor in &self.extractors {
            let res = extractor.extract(assimp_data, ctx)?;
            println!(
                "cargo:warning=AssimpImporter: Extractor '{}' produced {} outputs",
                extractor.name(),
                res.len()
            );
            outputs.extend(res);
        }
        println!(
            "cargo:warning=AssimpImporter: TOTAL produced {} outputs",
            outputs.len()
        );
        Ok(outputs)
    }
}

fn extract_meshes(scene: &asset_importer::scene::Scene) -> Vec<ExtractedMesh> {
    let mut extracted = Vec::new();
    for mesh in scene.meshes() {
        let has_normals = mesh.has_normals();
        let has_uvs = mesh.has_texture_coords(0);
        let has_colors = mesh.has_vertex_colors(0);

        let num_vertices = mesh.num_vertices();
        let mut vertices = Vec::with_capacity(num_vertices * 9);

        let pos_iter = mesh.vertices();
        let norm_iter = mesh.normals().unwrap_or_else(|| {
            vec![asset_importer::types::Vector3D::new(0.0, 0.0, 1.0); num_vertices]
        });
        let _uv_iter = mesh.texture_coords(0).unwrap_or_else(|| {
            vec![asset_importer::types::Vector3D::new(0.0, 0.0, 0.0); num_vertices]
        });
        let col_iter = mesh.vertex_colors(0).unwrap_or_else(|| {
            vec![asset_importer::types::Color4D::new(1.0, 1.0, 1.0, 1.0); num_vertices]
        });

        for i in 0..num_vertices {
            let p = pos_iter[i];
            vertices.push(p.x);
            vertices.push(p.y);
            vertices.push(p.z);

            if has_normals {
                let n = norm_iter[i];
                vertices.push(n.x);
                vertices.push(n.y);
                vertices.push(n.z);
            } else {
                vertices.push(0.0);
                vertices.push(0.0);
                vertices.push(1.0);
            }

            if has_colors {
                let c = col_iter[i];
                vertices.push(c.x);
                vertices.push(c.y);
                vertices.push(c.z);
            } else {
                vertices.push(1.0);
                vertices.push(1.0);
                vertices.push(1.0);
            }
        }

        let indices: Vec<u32> = mesh.triangle_indices_iter().collect();

        extracted.push(ExtractedMesh {
            vertices,
            indices,
            has_normals,
            has_uvs,
            has_colors,
            material_index: Some(mesh.material_index()),
        });
    }
    extracted
}

fn extract_materials(
    scene: &asset_importer::scene::Scene,
    _source_path: &Path,
) -> Vec<ExtractedMaterial> {
    scene
        .materials()
        .enumerate()
        .map(|(idx, mat)| {
            use asset_importer::material::TextureType;

            let name = mat.name();
            let name = if name.is_empty() {
                format!("material_{}", idx)
            } else {
                name
            };

            let mut albedo = None;
            let mut normal = None;
            let mut metallic_roughness = None;
            let mut emissive = None;

            if let Some(tex) = mat
                .texture_ref(TextureType::Diffuse, 0)
                .or_else(|| mat.texture_ref(TextureType::BaseColor, 0))
            {
                albedo = Some(PathBuf::from(tex.path_str().as_ref()));
            }
            if let Some(tex) = mat.texture_ref(TextureType::Normals, 0) {
                normal = Some(PathBuf::from(tex.path_str().as_ref()));
            }
            if let Some(tex) = mat.texture_ref(TextureType::Emissive, 0) {
                emissive = Some(PathBuf::from(tex.path_str().as_ref()));
            }
            if let Some(tex) = mat
                .texture_ref(TextureType::Unknown, 0)
                .or_else(|| mat.texture_ref(TextureType::Metalness, 0))
            {
                metallic_roughness = Some(PathBuf::from(tex.path_str().as_ref()));
            }

            let mut base_color_factor = [1.0, 1.0, 1.0, 1.0];
            if let Some(c) = mat.base_color().or_else(|| {
                mat.diffuse_color()
                    .map(|c| asset_importer::types::Color4D::new(c.x, c.y, c.z, 1.0))
            }) {
                base_color_factor = [c.x, c.y, c.z, c.w];
            }

            let metallic_factor = mat.metallic_factor().unwrap_or(1.0);
            let roughness_factor = mat.roughness_factor().unwrap_or(1.0);

            let mut emissive_factor = [0.0, 0.0, 0.0];
            if let Some(c) = mat.emissive_color() {
                emissive_factor = [c.x, c.y, c.z];
            }

            // Fallback for alpha cutoff if needed. 0.5 is default.
            let alpha_cutoff = mat
                .get_float_property_str("$mat.gltf.alphaCutoff")
                .unwrap_or_default()
                .unwrap_or(0.5);

            ExtractedMaterial {
                name,
                albedo_path: albedo,
                normal_path: normal,
                metallic_roughness_path: metallic_roughness,
                emissive_path: emissive,
                base_color_factor,
                metallic_factor,
                roughness_factor,
                emissive_factor,
                alpha_cutoff,
            }
        })
        .collect()
}

pub struct MeshExtractor;
impl Extractor for MeshExtractor {
    fn name(&self) -> &str {
        "MeshExtractor"
    }
    fn output_type(&self) -> Uuid {
        MESH_TYPE_UUID
    }
    fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let assimp_data = data.as_any().downcast_ref::<AssimpScene>().unwrap();
        let namespace = Uuid::new_v5(
            &Uuid::NAMESPACE_OID,
            assimp_data.source_path.to_string_lossy().as_bytes(),
        );
        let mut outputs = Vec::new();
        for i in 0..assimp_data.meshes.len() {
            outputs.push(build_mesh_output(assimp_data, i, namespace)?);
        }
        Ok(outputs)
    }
}

fn build_mesh_output(
    assimp_data: &AssimpScene,
    mesh_idx: usize,
    namespace: Uuid,
) -> Result<BakeOutput> {
    let mesh = &assimp_data.meshes[mesh_idx];
    let file_stem = assimp_data
        .source_path
        .file_stem()
        .unwrap()
        .to_string_lossy();
    let name = format!("{}_mesh_{}", file_stem, mesh_idx);
    let asset_id = Uuid::new_v5(&namespace, name.as_bytes());

    let vertex_format = VertexFormat::POSITION_NORMAL_COLOR;
    let stride = vertex_format.stride();
    let vertex_data: Vec<u8> = mesh.vertices.iter().flat_map(|f| f.to_ne_bytes()).collect();

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

    let bounds = calculate_bounds(&mesh.vertices, stride as usize / 4);
    let header = MeshHeader {
        vertex_count: (mesh.vertices.len() / (stride as usize / 4)) as u32,
        index_count: mesh.indices.len() as u32,
        vertex_stride: stride,
        index_format,
        vertex_format,
        vertex_offset: std::mem::size_of::<MeshHeader>() as u32,
        index_offset: (std::mem::size_of::<MeshHeader>() + vertex_data.len()) as u32,
        bounds_offset: (std::mem::size_of::<MeshHeader>() + vertex_data.len() + index_data.len())
            as u32,
        skeleton_id: [0u8; 16],
        material_id: get_material_id(assimp_data, mesh.material_index).into_bytes(),
    };

    let mut data = Vec::new();
    data.extend_from_slice(bytes_of(&header));
    data.extend_from_slice(&vertex_data);
    data.extend_from_slice(&index_data);
    data.extend_from_slice(bytes_of(&bounds));

    Ok(BakeOutput {
        asset_id,
        asset_type: MESH_TYPE_UUID,
        data,
        name,
    })
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
    let count = if stride_floats > 0 {
        vertices.len() / stride_floats
    } else {
        0
    };
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
    fn name(&self) -> &str {
        "MaterialExtractor"
    }
    fn output_type(&self) -> Uuid {
        MATERIAL_ASSET_TYPE
    }
    fn extract(&self, data: &dyn ImportedData, ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let assimp_data = data.as_any().downcast_ref::<AssimpScene>().unwrap();
        let mut outputs = Vec::new();

        for mat in &assimp_data.materials {
            let asset_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, mat.name.as_bytes());

            // Texture UUIDs
            let albedo_id = bake_texture(
                mat.albedo_path.as_ref(),
                assimp_data,
                ctx,
                true,
                &mut outputs,
            )?;
            let normal_id = bake_texture(
                mat.normal_path.as_ref(),
                assimp_data,
                ctx,
                false,
                &mut outputs,
            )?;
            let metallic_roughness_id = bake_texture(
                mat.metallic_roughness_path.as_ref(),
                assimp_data,
                ctx,
                false,
                &mut outputs,
            )?;
            let emissive_id = bake_texture(
                mat.emissive_path.as_ref(),
                assimp_data,
                ctx,
                true,
                &mut outputs,
            )?;

            let header = MaterialHeader {
                albedo_texture: albedo_id.into_bytes(),
                normal_texture: normal_id.into_bytes(),
                metallic_roughness_texture: metallic_roughness_id.into_bytes(),
                emissive_texture: emissive_id.into_bytes(),
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
    scene: &AssimpScene,
    ctx: &BakeContext,
    is_srgb: bool,
    outputs: &mut Vec<BakeOutput>,
) -> Result<Uuid> {
    let path = match path {
        Some(p) => p,
        None => return Ok(Uuid::nil()),
    };

    println!(
        "cargo:warning=AssimpImporter: bake_texture called with path: {:?}",
        path
    );

    let source_dir = ctx.source_path.parent().unwrap();
    let filename = path.to_string_lossy().trim().to_string();

    if filename.is_empty() {
        return Ok(Uuid::nil());
    }

    use crate::importers::image_importer::{ImageImporter, TextureImportOptions};
    use i3_io::texture::TextureFormat;

    let mut format = if is_srgb {
        TextureFormat::BC7_SRGB
    } else {
        TextureFormat::BC7_UNORM
    };
    let path_lc = path.to_string_lossy().to_lowercase();
    if path_lc.contains("normal") || path_lc.contains("norm") {
        format = TextureFormat::BC5_UNORM;
    }

    let importer = ImageImporter::new(TextureImportOptions {
        is_srgb,
        generate_mips: true,
        format,
    });

    if let Some(embedded_idx_str) = filename.strip_prefix('*') {
        if let Ok(idx) = embedded_idx_str.parse::<usize>() {
            if idx < scene.embedded_textures.len() {
                let asset_id = Uuid::new_v5(
                    &Uuid::NAMESPACE_URL,
                    format!("embedded_{}_{}", scene.source_path.display(), idx).as_bytes(),
                );

                if outputs.iter().any(|o| o.asset_id == asset_id) {
                    return Ok(asset_id);
                }

                println!(
                    "cargo:warning=AssimpImporter: Baking embedded texture '*{}' (ID={:?})",
                    idx, asset_id
                );

                let buffer = &scene.embedded_textures[idx];
                let imported = importer.import_memory(buffer, &scene.source_path)?;
                let texture_outputs = importer.extract(imported.as_ref(), ctx)?;
                outputs.extend(texture_outputs);

                // Override the UUID of the output to match our embedded one
                if let Some(last) = outputs.last_mut() {
                    last.asset_id = asset_id;
                }

                return Ok(asset_id);
            }
        }

        println!(
            "cargo:warning=AssimpImporter: Failed to resolve embedded texture '{}'",
            filename
        );
        return Ok(Uuid::nil());
    }

    let mut full_path = source_dir.join(&filename);

    // Heuristics to find the texture if not in the immediate directory
    if !full_path.exists() {
        let candidates = [
            source_dir.join(&filename),
            source_dir.parent().unwrap_or(source_dir).join(&filename),
            source_dir.join("textures").join(&filename),
            source_dir
                .parent()
                .unwrap_or(source_dir)
                .join("textures")
                .join(&filename),
            // glTF often has a 'textures' folder next to the .gltf
            source_dir.join("..").join("textures").join(&filename),
        ];

        for cand in candidates {
            if cand.exists() {
                full_path = cand;
                break;
            }
        }
    }

    // Final fallback: search recursively in the asset root?
    // For now, let's just log and skip if still not found.
    if !full_path.exists() {
        println!(
            "cargo:warning=AssimpImporter: Texture NOT FOUND: '{}' (searched around {:?})",
            filename, source_dir
        );
        return Ok(Uuid::nil());
    }

    if full_path.is_dir() {
        println!(
            "cargo:warning=AssimpImporter: Skipping directory texture at {:?}",
            full_path
        );
        return Ok(Uuid::nil());
    }

    let full_path = full_path.canonicalize().unwrap_or(full_path);

    let asset_id = Uuid::new_v5(&Uuid::NAMESPACE_URL, full_path.to_string_lossy().as_bytes());
    if outputs.iter().any(|o| o.asset_id == asset_id) {
        return Ok(asset_id);
    }

    println!(
        "cargo:warning=AssimpImporter: Baking texture {:?} (ID={:?})",
        full_path, asset_id
    );

    let imported = importer.import(&full_path)?;
    let texture_outputs = importer.extract(imported.as_ref(), ctx)?;
    outputs.extend(texture_outputs);

    Ok(asset_id)
}

pub struct SceneExtractor;
impl Extractor for SceneExtractor {
    fn name(&self) -> &str {
        "SceneExtractor"
    }
    fn output_type(&self) -> Uuid {
        SCENE_TYPE_UUID
    }
    fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let assimp_data = data.as_any().downcast_ref::<AssimpScene>().unwrap();
        let mut objects = Vec::new();
        let mut mesh_refs = Vec::new();
        let mut string_table = Vec::new();

        let namespace = Uuid::new_v5(
            &Uuid::NAMESPACE_OID,
            assimp_data.source_path.to_string_lossy().as_bytes(),
        );
        let file_stem = assimp_data
            .source_path
            .file_stem()
            .unwrap()
            .to_string_lossy();

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
            let mesh_bounds = calculate_bounds(
                &mesh_data.vertices,
                VertexFormat::POSITION_NORMAL_COLOR.stride() as usize / 4,
            );
            scene_bounds.merge(&mesh_bounds);
        }

        let header = SceneHeader {
            object_count: objects.len() as u32,
            light_count: 0,
            mesh_ref_count: mesh_refs.len() as u32,
            skeleton_ref_count: 0,
            objects_offset: std::mem::size_of::<SceneHeader>() as u32,
            lights_offset: 0,
            mesh_refs_offset: std::mem::size_of::<SceneHeader>() as u32
                + (objects.len() * std::mem::size_of::<ObjectInstance>()) as u32,
            skeleton_refs_offset: 0,
            strings_offset: std::mem::size_of::<SceneHeader>() as u32
                + (objects.len() * std::mem::size_of::<ObjectInstance>()) as u32
                + (mesh_refs.len() * 16) as u32,
            strings_size: string_table.len() as u32,
            bounds: scene_bounds,
            _reserved: [0u8; 16],
        };

        let mut binary = Vec::new();
        binary.extend_from_slice(bytes_of(&header));
        for obj in &objects {
            binary.extend_from_slice(bytes_of(obj));
        }
        for id in &mesh_refs {
            binary.extend_from_slice(id.as_bytes());
        }
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
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] = m[(i, j)];
        }
    }
    r
}
