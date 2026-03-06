//! i3_io prelude
//!
//! Commonly used types for the I/O and Asset systems.

pub use crate::AssetHeader;
pub use crate::asset::{Asset, AssetHandle};
pub use crate::error::{IoError, Result};
pub use crate::mesh::{BoundingBox, IndexFormat, MeshAsset, MeshHeader, VertexFormat};
pub use crate::scene_asset::{LightInstance, LightType, ObjectInstance, SceneAsset, SceneHeader};
pub use crate::vfs::{Vfs, VfsFile};
