use std::path::Path;

use i3_gfx::prelude::*;
use i3_renderer::scene::ObjectData;
use nalgebra_glm as glm;
use tracing::info;

use crate::basic_scene::BasicScene;

/// Loads a glTF/GLB file and populates a `BasicScene`.
///
/// Extracts meshes (positions, normals, vertex colors) and node transforms.
/// Each mesh primitive becomes a separate `Mesh`. Each node referencing a mesh
/// becomes an `ObjectData` with the node's world transform.
///
/// Limitations (Phase 2):
/// - Only reads POSITION, NORMAL, and COLOR_0 attributes
/// - Ignores texcoords, materials, textures, animations, skins
/// - Flat hierarchy (no parent-child transform accumulation beyond what glTF provides)
pub fn load_gltf(
    path: &Path,
    backend: &mut dyn RenderBackend,
) -> Result<BasicScene, Box<dyn std::error::Error>> {
    let (document, buffers, _images) = gltf::import(path)?;

    let mut scene = BasicScene::new();
    let mut mesh_id_map: Vec<Vec<u32>> = Vec::new(); // gltf mesh index → vec of our MeshIds (one per primitive)

    // 1. Extract all mesh primitives
    for mesh in document.meshes() {
        let mut primitive_ids = Vec::new();

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            // Positions (required)
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| format!("Mesh {} has no POSITION attribute", mesh.index()))?
                .collect();

            // Normals (default to up if missing)
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|n| n.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

            // Vertex colors (default to white if missing)
            let colors: Vec<[f32; 3]> = reader
                .read_colors(0)
                .map(|c| c.into_rgb_f32().collect())
                .unwrap_or_else(|| vec![[1.0, 1.0, 1.0]; positions.len()]);

            // Interleave into [f32; 9] = pos(3) + normal(3) + color(3)
            let vertices: Vec<[f32; 9]> = positions
                .iter()
                .zip(normals.iter())
                .zip(colors.iter())
                .map(|((p, n), c)| [p[0], p[1], p[2], n[0], n[1], n[2], c[0], c[1], c[2]])
                .collect();

            // Indices
            let indices: Vec<u16> = reader
                .read_indices()
                .ok_or_else(|| {
                    format!(
                        "Mesh {} primitive has no indices (non-indexed not supported)",
                        mesh.index()
                    )
                })?
                .into_u32()
                .map(|i| i as u16)
                .collect();

            let vb_bytes = unsafe {
                std::slice::from_raw_parts(
                    vertices.as_ptr() as *const u8,
                    vertices.len() * std::mem::size_of::<[f32; 9]>(),
                )
            };

            let mesh_id = scene.add_mesh(backend, vb_bytes, vertices.len() as u32, &indices);
            primitive_ids.push(mesh_id);
        }

        mesh_id_map.push(primitive_ids);
    }

    // 2. Extract nodes with meshes
    let mut object_count = 0u32;
    for node in document.nodes() {
        if let Some(mesh) = node.mesh() {
            let transform = node.transform().matrix();
            let world_transform = glm::Mat4::from_column_slice(&[
                transform[0][0],
                transform[0][1],
                transform[0][2],
                transform[0][3],
                transform[1][0],
                transform[1][1],
                transform[1][2],
                transform[1][3],
                transform[2][0],
                transform[2][1],
                transform[2][2],
                transform[2][3],
                transform[3][0],
                transform[3][1],
                transform[3][2],
                transform[3][3],
            ]);

            for &mesh_id in &mesh_id_map[mesh.index()] {
                scene.add_object(ObjectData {
                    world_transform,
                    prev_transform: world_transform,
                    material_id: 0,
                    mesh_id,
                });
                object_count += 1;
            }
        }
    }

    // 3. Add default light if scene has none
    scene.add_default_light();

    info!(
        "Loaded glTF: {} meshes, {} objects from {:?}",
        mesh_id_map.len(),
        object_count,
        path
    );

    Ok(scene)
}
