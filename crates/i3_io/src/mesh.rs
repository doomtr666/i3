//! Baked mesh asset format.
//!
//! GPU-ready vertex and index buffers with zero-copy loading support.

use crate::asset::Asset;
use crate::{AssetHeader, Result};
use bytemuck::{Pod, Zeroable};
use uuid::{Uuid, uuid};

/// UUID for mesh assets
pub const MESH_ASSET_TYPE: Uuid = uuid!("936da01f-9abd-4d9d-80c7-02af85c822a6");

/// Mesh header (64 bytes, repr C).
/// Describes the layout of vertex and index data following the header.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MeshHeader {
    /// Number of vertices.
    pub vertex_count: u32,
    /// Number of indices.
    pub index_count: u32,
    /// Size of one vertex in bytes.
    pub vertex_stride: u32,
    /// Index format: 0 = u16, 1 = u32.
    pub index_format: IndexFormat,
    /// Vertex format enum (see VertexFormat).
    pub vertex_format: VertexFormat,
    /// Offset of vertex data from payload start (after AssetHeader).
    pub vertex_offset: u32,
    /// Offset of index data from payload start.
    pub index_offset: u32,
    /// Offset of bounding box from payload start.
    pub bounds_offset: u32,
    /// UUID of associated skeleton (zero if static mesh).
    pub skeleton_id: [u8; 16],
    /// UUID of associated material.
    pub material_id: [u8; 16],
}

/// Index format for mesh indices.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct IndexFormat(pub u32);

impl IndexFormat {
    pub const U16: IndexFormat = IndexFormat(0);
    pub const U32: IndexFormat = IndexFormat(1);
}

/// Vertex format enum.
/// Defines the layout and stride of vertex data.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct VertexFormat(pub u32);

impl VertexFormat {
    /// Position(3) + Normal(3) + Color(3) = 36 bytes.
    /// Used for basic static meshes.
    pub const POSITION_NORMAL_COLOR: VertexFormat = VertexFormat(0);
    /// Position(3) + Normal(3) + UV(2) = 32 bytes.
    /// Used for textured static meshes.
    pub const POSITION_NORMAL_UV: VertexFormat = VertexFormat(1);
    /// Position(3) + Normal(3) + UV(2) + Tangent(4) = 48 bytes.
    /// Used for normal-mapped meshes.
    pub const POSITION_NORMAL_UV_TANGENT: VertexFormat = VertexFormat(2);
    /// Position(3) + Normal(3) + Color(3) + JointIndices(4) + JointWeights(4) = 52 bytes.
    /// Used for skinned meshes (Phase 2).
    pub const POSITION_NORMAL_COLOR_SKIN: VertexFormat = VertexFormat(3);
    /// Position(3) + Normal(3) + UV(2) + Tangent(4) + JointIndices(4) + JointWeights(4) = 56 bytes.
    /// Used for skinned meshes with normal mapping (Phase 2).
    pub const POSITION_NORMAL_UV_TANGENT_SKIN: VertexFormat = VertexFormat(4);

    /// Returns the stride in bytes for this vertex format.
    pub const fn stride(&self) -> u32 {
        match self.0 {
            0 => 36, // PositionNormalColor
            1 => 32, // PositionNormalUv
            2 => 48, // PositionNormalUvTangent
            3 => 52, // PositionNormalColorSkin
            4 => 56, // PositionNormalUvTangentSkin
            _ => 0,
        }
    }
}

/// Axis-aligned bounding box for frustum culling.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BoundingBox {
    /// Minimum corner (x, y, z).
    pub min: [f32; 3],
    /// Maximum corner (x, y, z).
    pub max: [f32; 3],
}

impl BoundingBox {
    pub fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    pub fn diagonal(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    pub fn diagonal_length(&self) -> f32 {
        let d = self.diagonal();
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    pub fn merge(&mut self, other: &BoundingBox) {
        for i in 0..3 {
            self.min[i] = self.min[i].min(other.min[i]);
            self.max[i] = self.max[i].max(other.max[i]);
        }
    }

    pub fn transform(&self, matrix: &[[f32; 4]; 4]) -> Self {
        let mut new_bounds = Self::empty();
        let corners = [
            [self.min[0], self.min[1], self.min[2]],
            [self.max[0], self.min[1], self.min[2]],
            [self.min[0], self.max[1], self.min[2]],
            [self.max[0], self.max[1], self.min[2]],
            [self.min[0], self.min[1], self.max[2]],
            [self.max[0], self.min[1], self.max[2]],
            [self.min[0], self.max[1], self.max[2]],
            [self.max[0], self.max[1], self.max[2]],
        ];

        for c in corners {
            let x = c[0] * matrix[0][0] + c[1] * matrix[1][0] + c[2] * matrix[2][0] + matrix[3][0];
            let y = c[0] * matrix[0][1] + c[1] * matrix[1][1] + c[2] * matrix[2][1] + matrix[3][1];
            let z = c[0] * matrix[0][2] + c[1] * matrix[1][2] + c[2] * matrix[2][2] + matrix[3][2];
            let w = c[0] * matrix[0][3] + c[1] * matrix[1][3] + c[2] * matrix[2][3] + matrix[3][3];

            let p = [x / w, y / w, z / w];
            for i in 0..3 {
                new_bounds.min[i] = new_bounds.min[i].min(p[i]);
                new_bounds.max[i] = new_bounds.max[i].max(p[i]);
            }
        }
        new_bounds
    }
}

/// Baked mesh asset loaded from a bundle.
pub struct MeshAsset {
    pub header: MeshHeader,
    pub bounds: BoundingBox,
    pub vertex_data: Vec<u8>,
    pub index_data: Vec<u8>,
}

impl MeshAsset {
    /// Returns the skeleton UUID if this mesh is skinned.
    pub fn skeleton_id(&self) -> Option<Uuid> {
        if self.header.skeleton_id == [0u8; 16] {
            None
        } else {
            Some(Uuid::from_bytes(self.header.skeleton_id))
        }
    }

    /// Returns vertex data as a byte slice for GPU upload.
    pub fn vertex_bytes(&self) -> &[u8] {
        &self.vertex_data
    }

    /// Returns index data as a byte slice for GPU upload.
    pub fn index_bytes(&self) -> &[u8] {
        &self.index_data
    }

    /// Returns the number of vertices.
    pub fn vertex_count(&self) -> u32 {
        self.header.vertex_count
    }

    /// Returns the number of indices.
    pub fn index_count(&self) -> u32 {
        self.header.index_count
    }

    /// Returns the index format.
    pub fn index_format(&self) -> IndexFormat {
        self.header.index_format
    }
}

impl Asset for MeshAsset {
    const ASSET_TYPE_ID: [u8; 16] = *MESH_ASSET_TYPE.as_bytes();

    fn load(header: &AssetHeader, data: &[u8]) -> Result<Self> {
        // Parse MeshHeader from the start of data
        if data.len() < std::mem::size_of::<MeshHeader>() {
            return Err(crate::IoError::InvalidData {
                message: format!(
                    "Mesh data too small for header: {} < {}",
                    data.len(),
                    std::mem::size_of::<MeshHeader>()
                ),
            });
        }

        let mesh_header: MeshHeader =
            bytemuck::pod_read_unaligned(&data[..std::mem::size_of::<MeshHeader>()]);

        // Validate vertex format stride
        let expected_stride = mesh_header.vertex_format.stride();
        if mesh_header.vertex_stride != expected_stride {
            return Err(crate::IoError::InvalidData {
                message: format!(
                    "Vertex stride mismatch for format {:?}: expected {}, got {}, header type: {:?}",
                    mesh_header.vertex_format,
                    expected_stride,
                    mesh_header.vertex_stride,
                    header.asset_type
                ),
            });
        }

        // Extract vertex data
        let vertex_size = mesh_header.vertex_count as usize * mesh_header.vertex_stride as usize;
        let vertex_start = mesh_header.vertex_offset as usize;
        let vertex_end = vertex_start + vertex_size;
        if vertex_end > data.len() {
            return Err(crate::IoError::InvalidData {
                message: "Vertex data exceeds bounds".to_string(),
            });
        }
        let vertex_data = data[vertex_start..vertex_end].to_vec();

        // Extract index data
        let index_size = mesh_header.index_count as usize
            * match mesh_header.index_format {
                IndexFormat::U16 => 2,
                _ => 4,
            };
        let index_start = mesh_header.index_offset as usize;
        let index_end = index_start + index_size;
        if index_end > data.len() {
            return Err(crate::IoError::InvalidData {
                message: "Index data exceeds bounds".to_string(),
            });
        }
        let index_data = data[index_start..index_end].to_vec();

        // Extract bounding box
        let bounds_start = mesh_header.bounds_offset as usize;
        let bounds_end = bounds_start + std::mem::size_of::<BoundingBox>();
        if bounds_end > data.len() {
            return Err(crate::IoError::InvalidData {
                message: "Bounds data exceeds bounds".to_string(),
            });
        }
        let bounds: BoundingBox = bytemuck::pod_read_unaligned(&data[bounds_start..bounds_end]);

        Ok(MeshAsset {
            header: mesh_header,
            bounds,
            vertex_data,
            index_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_format_stride() {
        assert_eq!(VertexFormat::POSITION_NORMAL_COLOR.stride(), 36);
        assert_eq!(VertexFormat::POSITION_NORMAL_UV.stride(), 32);
        assert_eq!(VertexFormat::POSITION_NORMAL_UV_TANGENT.stride(), 48);
    }

    #[test]
    fn test_mesh_header_size() {
        assert_eq!(std::mem::size_of::<MeshHeader>(), 64);
    }

    #[test]
    fn test_bounding_box_size() {
        assert_eq!(std::mem::size_of::<BoundingBox>(), 24);
    }
}
