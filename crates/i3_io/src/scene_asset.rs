//! Baked scene asset format.
//!
//! Contains object instances, lights, and references to meshes/skeletons.

use crate::asset::Asset;
use crate::{AssetHeader, Result};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use uuid::{Uuid, uuid};

/// UUID for scene assets: "i3scene"
pub const SCENE_ASSET_TYPE: Uuid = uuid!("59693373-6365-6e65-0000-000000000000");

/// Scene header (64 bytes, repr C).
/// Describes the counts and offsets for scene data.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SceneHeader {
    /// Number of object instances.
    pub object_count: u32,
    /// Number of light instances.
    pub light_count: u32,
    /// Number of mesh references.
    pub mesh_ref_count: u32,
    /// Number of skeleton references.
    pub skeleton_ref_count: u32,
    /// Offset to object instances array from payload start.
    pub objects_offset: u32,
    /// Offset to light instances array from payload start.
    pub lights_offset: u32,
    /// Offset to mesh reference UUIDs from payload start.
    pub mesh_refs_offset: u32,
    /// Offset to skeleton reference UUIDs from payload start.
    pub skeleton_refs_offset: u32,
    /// Offset to string table from payload start.
    pub strings_offset: u32,
    /// Total string table size in bytes.
    pub strings_size: u32,
    /// Reserved for future use.
    pub _reserved: [u8; 24],
}

/// Object instance in a scene.
/// Maps to ObjectData in the renderer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ObjectInstance {
    /// 4x4 transformation matrix (column-major).
    pub transform: [[f32; 4]; 4],
    /// Index into mesh_refs array.
    pub mesh_ref_index: u32,
    /// Index into skeleton_refs array (0xFFFFFFFF if none).
    pub skeleton_ref_index: u32,
    /// Index into string table for object name.
    pub name_offset: u32,
    /// Reserved for future use.
    pub _reserved: [u32; 3],
}

/// Light instance in a scene.
/// Maps to LightData in the renderer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LightInstance {
    /// Light type: 0 = Point, 1 = Directional, 2 = Spot.
    pub light_type: u32,
    /// Position (for point/spot lights).
    pub position: [f32; 3],
    /// Direction (for directional/spot lights).
    pub direction: [f32; 3],
    /// RGB color intensity.
    pub color: [f32; 3],
    /// Intensity multiplier.
    pub intensity: f32,
    /// Range (for point/spot lights).
    pub range: f32,
    /// Inner cone angle (for spot lights, in radians).
    pub inner_cone_angle: f32,
    /// Outer cone angle (for spot lights, in radians).
    pub outer_cone_angle: f32,
    /// Index into string table for light name.
    pub name_offset: u32,
    /// Reserved for future use.
    pub _reserved: [u32; 1],
}

/// Light type enum matching LightInstance.light_type.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct LightType(pub u32);

impl LightType {
    pub const POINT: LightType = LightType(0);
    pub const DIRECTIONAL: LightType = LightType(1);
    pub const SPOT: LightType = LightType(2);
}

/// Baked scene asset loaded from a bundle.
pub struct SceneAsset {
    pub header: SceneHeader,
    pub objects: Vec<ObjectInstance>,
    pub lights: Vec<LightInstance>,
    pub mesh_refs: Vec<Uuid>,
    pub skeleton_refs: Vec<Uuid>,
    pub string_table: Vec<u8>,
}

impl SceneAsset {
    /// Get the name of an object from the string table.
    pub fn object_name(&self, obj: &ObjectInstance) -> Option<&str> {
        if obj.name_offset == u32::MAX {
            return None;
        }
        let start = obj.name_offset as usize;
        if start >= self.string_table.len() {
            return None;
        }
        // Null-terminated string
        let end = self.string_table[start..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.string_table.len() - start);
        std::str::from_utf8(&self.string_table[start..start + end]).ok()
    }

    /// Get the name of a light from the string table.
    pub fn light_name(&self, light: &LightInstance) -> Option<&str> {
        if light.name_offset == u32::MAX {
            return None;
        }
        let start = light.name_offset as usize;
        if start >= self.string_table.len() {
            return None;
        }
        let end = self.string_table[start..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.string_table.len() - start);
        std::str::from_utf8(&self.string_table[start..start + end]).ok()
    }

    /// Get the mesh UUID for an object.
    pub fn mesh_for_object(&self, obj: &ObjectInstance) -> Option<Uuid> {
        let idx = obj.mesh_ref_index as usize;
        self.mesh_refs.get(idx).copied()
    }

    /// Get the skeleton UUID for an object.
    pub fn skeleton_for_object(&self, obj: &ObjectInstance) -> Option<Uuid> {
        if obj.skeleton_ref_index == u32::MAX {
            return None;
        }
        let idx = obj.skeleton_ref_index as usize;
        self.skeleton_refs.get(idx).copied()
    }

    /// Build a map from mesh UUID to list of object indices.
    pub fn objects_by_mesh(&self) -> HashMap<Uuid, Vec<usize>> {
        let mut map: HashMap<Uuid, Vec<usize>> = HashMap::new();
        for (i, obj) in self.objects.iter().enumerate() {
            if let Some(mesh_id) = self.mesh_for_object(obj) {
                map.entry(mesh_id).or_default().push(i);
            }
        }
        map
    }
}

impl Asset for SceneAsset {
    const ASSET_TYPE_ID: [u8; 16] = *SCENE_ASSET_TYPE.as_bytes();

    fn load(_header: &AssetHeader, data: &[u8]) -> Result<Self> {
        // Parse SceneHeader
        if data.len() < std::mem::size_of::<SceneHeader>() {
            return Err(crate::IoError::InvalidData {
                message: "Scene data too small for header".to_string(),
            });
        }

        let scene_header: SceneHeader =
            bytemuck::pod_read_unaligned(&data[..std::mem::size_of::<SceneHeader>()]);

        // Extract object instances
        let objects = if scene_header.object_count > 0 {
            let start = scene_header.objects_offset as usize;
            let size = scene_header.object_count as usize * std::mem::size_of::<ObjectInstance>();
            let end = start + size;
            if end > data.len() {
                return Err(crate::IoError::InvalidData {
                    message: "Object data exceeds bounds".to_string(),
                });
            }
            bytemuck::cast_slice(&data[start..end]).to_vec()
        } else {
            Vec::new()
        };

        // Extract light instances
        let lights = if scene_header.light_count > 0 {
            let start = scene_header.lights_offset as usize;
            let size = scene_header.light_count as usize * std::mem::size_of::<LightInstance>();
            let end = start + size;
            if end > data.len() {
                return Err(crate::IoError::InvalidData {
                    message: "Light data exceeds bounds".to_string(),
                });
            }
            bytemuck::cast_slice(&data[start..end]).to_vec()
        } else {
            Vec::new()
        };

        // Extract mesh references (UUIDs)
        let mesh_refs = if scene_header.mesh_ref_count > 0 {
            let start = scene_header.mesh_refs_offset as usize;
            let size = scene_header.mesh_ref_count as usize * 16; // UUID = 16 bytes
            let end = start + size;
            if end > data.len() {
                return Err(crate::IoError::InvalidData {
                    message: "Mesh refs data exceeds bounds".to_string(),
                });
            }
            (0..scene_header.mesh_ref_count as usize)
                .map(|i| {
                    let offset = start + i * 16;
                    let bytes: [u8; 16] = data[offset..offset + 16].try_into().unwrap();
                    Uuid::from_bytes(bytes)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Extract skeleton references (UUIDs)
        let skeleton_refs = if scene_header.skeleton_ref_count > 0 {
            let start = scene_header.skeleton_refs_offset as usize;
            let size = scene_header.skeleton_ref_count as usize * 16; // UUID = 16 bytes
            let end = start + size;
            if end > data.len() {
                return Err(crate::IoError::InvalidData {
                    message: "Skeleton refs data exceeds bounds".to_string(),
                });
            }
            (0..scene_header.skeleton_ref_count as usize)
                .map(|i| {
                    let offset = start + i * 16;
                    let bytes: [u8; 16] = data[offset..offset + 16].try_into().unwrap();
                    Uuid::from_bytes(bytes)
                })
                .collect()
        } else {
            Vec::new()
        };

        // Extract string table
        let string_table = if scene_header.strings_size > 0 {
            let start = scene_header.strings_offset as usize;
            let end = start + scene_header.strings_size as usize;
            if end > data.len() {
                return Err(crate::IoError::InvalidData {
                    message: "String table exceeds bounds".to_string(),
                });
            }
            data[start..end].to_vec()
        } else {
            Vec::new()
        };

        Ok(SceneAsset {
            header: scene_header,
            objects,
            lights,
            mesh_refs,
            skeleton_refs,
            string_table,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asset_header_size() {
        assert_eq!(std::mem::size_of::<crate::AssetHeader>(), 64);
    }

    #[test]
    fn test_scene_header_size() {
        assert_eq!(std::mem::size_of::<SceneHeader>(), 64);
    }

    #[test]
    fn test_object_instance_size() {
        assert_eq!(std::mem::size_of::<ObjectInstance>(), 88);
    }

    #[test]
    fn test_light_instance_size() {
        assert_eq!(std::mem::size_of::<LightInstance>(), 64);
    }

    #[test]
    fn test_asset_header_binary_consistency() {
        use uuid::Uuid;
        let id = Uuid::new_v4();
        let header = crate::AssetHeader::new(id, 1234, 5678);
        let bytes = bytemuck::bytes_of(&header);

        assert_eq!(bytes.len(), 64);

        // Magic should be at offset 0
        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(magic, crate::AssetHeader::MAGIC);

        // Version should be at offset 8
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(version, crate::AssetHeader::VERSION);

        // Data offset should be at offset 16
        let offset = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        assert_eq!(offset, 1234);
    }
}
