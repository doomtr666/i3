use crate::Result;
use crate::pipeline::{BakeContext, BakeOutput, Extractor, ImportedData, Importer};
use bytemuck::bytes_of;
use i3_io::material::{MATERIAL_ASSET_TYPE, MaterialHeader};
use i3_io::mesh::{BoundingBox, IndexFormat, MeshHeader, VertexFormat};
use i3_io::scene_asset::{ObjectInstance, SceneHeader};
use nalgebra_glm::Mat4;
use rayon::prelude::*;
use std::any::Any;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::importers::image_importer::{ImageImporter, TextureImportOptions, TextureSemantic};

/// UUID type for i3mesh assets.
const MESH_TYPE_UUID: Uuid = i3_io::mesh::MESH_ASSET_TYPE;

/// UUID type for i3scene assets.
const SCENE_TYPE_UUID: Uuid = i3_io::scene_asset::SCENE_ASSET_TYPE;

// ---------------------------------------------------------------------------
// Texture resolution types (for parallel baking)
// ---------------------------------------------------------------------------

/// Source of a texture to bake (embedded in the model or external file).
#[derive(Clone, Debug)]
enum TextureSource {
    /// Embedded texture at the given index in `AssimpScene::embedded_textures`.
    Embedded { index: usize },
    /// External texture file at the given resolved absolute path.
    File { path: PathBuf },
}

/// A fully resolved texture reference, ready for parallel baking.
#[derive(Clone, Debug)]
struct ResolvedTexture {
    asset_id: Uuid,
    source: TextureSource,
    semantic: TextureSemantic,
}

// ---------------------------------------------------------------------------
// Extracted data types (Send + Sync compatible)
// ---------------------------------------------------------------------------

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
    pub has_tangents: bool,
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

// ---------------------------------------------------------------------------
// AssimpImporter
// ---------------------------------------------------------------------------

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
                            // asset-importer Texel has b,g,r,a
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

        println!(
            "cargo:warning=AssimpImporter: Running {} extractors (parallel)",
            self.extractors.len()
        );

        // Run all extractors in parallel via rayon.
        // Each extractor reads from the shared &AssimpScene (Sync) and &BakeContext (Sync).
        let results: Result<Vec<Vec<BakeOutput>>> = self
            .extractors
            .par_iter()
            .map(|extractor| {
                let res = extractor.extract(assimp_data, ctx)?;
                println!(
                    "cargo:warning=AssimpImporter: Extractor '{}' produced {} outputs",
                    extractor.name(),
                    res.len()
                );
                Ok(res)
            })
            .collect();

        let all_outputs = results?;
        let outputs: Vec<BakeOutput> = all_outputs.into_iter().flatten().collect();

        println!(
            "cargo:warning=AssimpImporter: TOTAL produced {} outputs",
            outputs.len()
        );
        Ok(outputs)
    }
}

// ---------------------------------------------------------------------------
// Assimp scene data extraction (from C++ scene into Rust structs)
// ---------------------------------------------------------------------------

fn extract_meshes(scene: &asset_importer::scene::Scene) -> Vec<ExtractedMesh> {
    let mut extracted = Vec::new();
    for mesh in scene.meshes() {
        let has_normals = mesh.has_normals();
        let has_uvs = mesh.has_texture_coords(0);
        let has_colors = mesh.has_vertex_colors(0);

        let num_vertices = mesh.num_vertices();

        // If it has UVs, we now ALWAYS include tangents (48 bytes stride: P3, N3, UV2, T4)
        // Otherwise, we use P3, N3, C3 (36 bytes stride)
        let has_tangents = mesh.tangents().is_some() && mesh.bitangents().is_some();
        let stride = if has_uvs { 12 } else { 9 };
        let mut vertices = Vec::with_capacity(num_vertices * stride);

        let pos_iter = mesh.vertices();
        let norm_iter = mesh.normals().unwrap_or_else(|| {
            vec![asset_importer::types::Vector3D::new(0.0, 0.0, 1.0); num_vertices]
        });
        let uv_iter = if has_uvs {
            mesh.texture_coords(0).unwrap()
        } else {
            vec![asset_importer::types::Vector3D::new(0.0, 0.0, 0.0); num_vertices]
        };
        let col_iter = if has_colors {
            mesh.vertex_colors(0).unwrap()
        } else {
            vec![asset_importer::types::Color4D::new(1.0, 1.0, 1.0, 1.0); num_vertices]
        };
        let tang_iter = mesh.tangents();
        let bitang_iter = mesh.bitangents();

        for i in 0..num_vertices {
            let p = pos_iter[i];
            vertices.push(p.x);
            vertices.push(p.y);
            vertices.push(p.z);

            let n = norm_iter[i];
            vertices.push(n.x);
            vertices.push(n.y);
            vertices.push(n.z);

            if has_uvs {
                let uv = uv_iter[i];
                vertices.push(uv.x);
                vertices.push(uv.y);

                if let (Some(tangents), Some(bitangents)) = (&tang_iter, &bitang_iter) {
                    let t = tangents[i];
                    let b = bitangents[i];
                    let n = norm_iter[i];

                    // Calculate handness (w component of tangent)
                    // w = dot(cross(n, t), b) < 0 ? -1.0 : 1.0
                    let cross_nt = asset_importer::types::Vector3D::new(
                        n.y * t.z - n.z * t.y,
                        n.z * t.x - n.x * t.z,
                        n.x * t.y - n.y * t.x,
                    );
                    let dot_val = cross_nt.x * b.x + cross_nt.y * b.y + cross_nt.z * b.z;
                    let w = if dot_val < 0.0 { -1.0 } else { 1.0 };

                    vertices.push(t.x);
                    vertices.push(t.y);
                    vertices.push(t.z);
                    vertices.push(w);
                } else {
                    vertices.push(1.0);
                    vertices.push(0.0);
                    vertices.push(0.0);
                    vertices.push(1.0);
                }
            } else {
                let c = col_iter[i];
                vertices.push(c.x);
                vertices.push(c.y);
                vertices.push(c.z);
            }
        }

        let indices: Vec<u32> = mesh.triangle_indices_iter().collect();

        extracted.push(ExtractedMesh {
            vertices,
            indices,
            has_normals,
            has_uvs,
            has_tangents,
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

// ---------------------------------------------------------------------------
// MeshExtractor — parallel mesh baking
// ---------------------------------------------------------------------------

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

        // Build all meshes in parallel via rayon.
        (0..assimp_data.meshes.len())
            .into_par_iter()
            .map(|i| build_mesh_output(assimp_data, i, namespace))
            .collect()
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

    let vertex_format = if mesh.has_uvs {
        VertexFormat::POSITION_NORMAL_UV_TANGENT
    } else {
        VertexFormat::POSITION_NORMAL_COLOR
    };
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

// ---------------------------------------------------------------------------
// MaterialExtractor — three-phase parallel texture baking
// ---------------------------------------------------------------------------

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

        // ---------------------------------------------------------------
        // Phase 1: Collect all unique texture references across materials.
        // Deduplication is by asset_id (deterministic UUID from path).
        // ---------------------------------------------------------------
        let mut unique_textures: HashMap<Uuid, ResolvedTexture> = HashMap::new();

        // Per-material texture UUIDs: [albedo, normal, metallic_roughness, emissive]
        let mut mat_texture_ids: Vec<[Uuid; 4]> = Vec::with_capacity(assimp_data.materials.len());

        for mat in &assimp_data.materials {
            let mut ids = [Uuid::nil(); 4];

            // Albedo
            if let Some(resolved) = resolve_texture_ref(
                mat.albedo_path.as_ref(),
                assimp_data,
                ctx,
                TextureSemantic::Albedo,
            ) {
                ids[0] = resolved.asset_id;
                unique_textures.entry(resolved.asset_id).or_insert(resolved);
            }

            // Normal
            if let Some(resolved) = resolve_texture_ref(
                mat.normal_path.as_ref(),
                assimp_data,
                ctx,
                TextureSemantic::Normal,
            ) {
                ids[1] = resolved.asset_id;
                unique_textures.entry(resolved.asset_id).or_insert(resolved);
            }

            // MetallicRoughness
            if let Some(resolved) = resolve_texture_ref(
                mat.metallic_roughness_path.as_ref(),
                assimp_data,
                ctx,
                TextureSemantic::MetallicRoughness,
            ) {
                ids[2] = resolved.asset_id;
                unique_textures.entry(resolved.asset_id).or_insert(resolved);
            }

            // Emissive
            if let Some(resolved) = resolve_texture_ref(
                mat.emissive_path.as_ref(),
                assimp_data,
                ctx,
                TextureSemantic::Emissive,
            ) {
                ids[3] = resolved.asset_id;
                unique_textures.entry(resolved.asset_id).or_insert(resolved);
            }

            mat_texture_ids.push(ids);
        }

        println!(
            "cargo:warning=MaterialExtractor: {} unique textures to bake (parallel)",
            unique_textures.len()
        );

        // ---------------------------------------------------------------
        // Phase 2: Bake all unique textures in parallel via rayon.
        // This is where the heavy BC7/BC5 compression happens.
        // ---------------------------------------------------------------
        let texture_entries: Vec<(Uuid, ResolvedTexture)> = unique_textures.into_iter().collect();

        let baked_results: Result<Vec<(Uuid, Vec<BakeOutput>)>> = texture_entries
            .par_iter()
            .map(|(id, resolved)| {
                println!(
                    "cargo:warning=MaterialExtractor: Baking texture {:?} ({:?})",
                    resolved.semantic, resolved.source
                );
                let outputs = bake_resolved_texture(resolved, assimp_data, ctx)?;
                Ok((*id, outputs))
            })
            .collect();

        let baked_map: HashMap<Uuid, Vec<BakeOutput>> = baked_results?.into_iter().collect();

        // ---------------------------------------------------------------
        // Phase 3: Assemble material assets + collect texture outputs.
        // ---------------------------------------------------------------
        let mut outputs: Vec<BakeOutput> = Vec::new();

        // First, collect all baked texture outputs.
        for baked_outputs in baked_map.into_values() {
            outputs.extend(baked_outputs);
        }

        // Then, build material assets using the resolved texture UUIDs.
        for (i, mat) in assimp_data.materials.iter().enumerate() {
            let asset_id = Uuid::new_v5(&Uuid::NAMESPACE_OID, mat.name.as_bytes());
            let ids = &mat_texture_ids[i];

            let header = MaterialHeader {
                albedo_texture: ids[0].into_bytes(),
                normal_texture: ids[1].into_bytes(),
                metallic_roughness_texture: ids[2].into_bytes(),
                emissive_texture: ids[3].into_bytes(),
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

// ---------------------------------------------------------------------------
// Texture resolution helpers
// ---------------------------------------------------------------------------

/// Resolve a texture path to a `ResolvedTexture` without performing any I/O-heavy
/// work. Returns `None` if the path is absent, empty, or cannot be resolved.
fn resolve_texture_ref(
    path: Option<&PathBuf>,
    scene: &AssimpScene,
    ctx: &BakeContext,
    semantic: TextureSemantic,
) -> Option<ResolvedTexture> {
    let path = path?;
    let source_dir = ctx.source_path.parent().unwrap();
    let filename = path.to_string_lossy().trim().to_string();

    if filename.is_empty() {
        return None;
    }

    // Embedded texture (path starts with '*')
    if let Some(embedded_idx_str) = filename.strip_prefix('*') {
        if let Ok(idx) = embedded_idx_str.parse::<usize>() {
            if idx < scene.embedded_textures.len() {
                let asset_id = Uuid::new_v5(
                    &Uuid::NAMESPACE_URL,
                    format!("embedded_{}_{}", scene.source_path.display(), idx).as_bytes(),
                );
                return Some(ResolvedTexture {
                    asset_id,
                    source: TextureSource::Embedded { index: idx },
                    semantic,
                });
            }
        }
        println!(
            "cargo:warning=resolve_texture_ref: Failed to resolve embedded texture '{}'",
            filename
        );
        return None;
    }

    // External file texture — resolve with heuristics
    let mut full_path = source_dir.join(&filename);

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

    if !full_path.exists() {
        println!(
            "cargo:warning=resolve_texture_ref: Texture NOT FOUND: '{}' (searched around {:?})",
            filename, source_dir
        );
        return None;
    }

    if full_path.is_dir() {
        println!(
            "cargo:warning=resolve_texture_ref: Skipping directory texture at {:?}",
            full_path
        );
        return None;
    }

    let full_path = full_path.canonicalize().unwrap_or(full_path);
    let asset_id = Uuid::new_v5(&Uuid::NAMESPACE_URL, full_path.to_string_lossy().as_bytes());

    Some(ResolvedTexture {
        asset_id,
        source: TextureSource::File { path: full_path },
        semantic,
    })
}

/// Bake a resolved texture into one or more `BakeOutput`s.
/// This is the CPU-intensive step (image decode + BC7/BC5 compression + mipmap generation).
fn bake_resolved_texture(
    resolved: &ResolvedTexture,
    scene: &AssimpScene,
    ctx: &BakeContext,
) -> Result<Vec<BakeOutput>> {
    let importer = ImageImporter::new(TextureImportOptions {
        semantic: resolved.semantic,
        generate_mips: true,
        format: None,
    });

    let mut outputs = match &resolved.source {
        TextureSource::Embedded { index } => {
            let buffer = &scene.embedded_textures[*index];
            let imported = importer.import_memory(buffer, &scene.source_path)?;
            importer.extract(imported.as_ref(), ctx)?
        }
        TextureSource::File { path } => {
            let imported = importer.import(path)?;
            importer.extract(imported.as_ref(), ctx)?
        }
    };

    // Override the asset_id to match our deterministic UUID from resolve_texture_ref.
    for output in &mut outputs {
        output.asset_id = resolved.asset_id;
    }

    Ok(outputs)
}

// ---------------------------------------------------------------------------
// SceneExtractor
// ---------------------------------------------------------------------------

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
            let mesh_format = if mesh_data.has_uvs {
                VertexFormat::POSITION_NORMAL_UV
            } else {
                VertexFormat::POSITION_NORMAL_COLOR
            };
            let mesh_bounds =
                calculate_bounds(&mesh_data.vertices, mesh_format.stride() as usize / 4);
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
