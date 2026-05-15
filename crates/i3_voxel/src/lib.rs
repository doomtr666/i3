pub mod octree;
pub mod sdf;
pub mod voxel;

pub use i3_math::{AABB, Transform};
pub use octree::{VoxelOctree, VoxelSceneSink};
pub use sdf::{SdfPrimitive, SdfScene};
pub use voxel::{VoxelBlock, VoxelScene, VoxelVertex};

pub mod prelude {
    pub use i3_math::prelude::*;
    pub use crate::octree::{VoxelOctree, VoxelSceneSink};
    pub use crate::sdf::{SdfPrimitive, SdfScene};
    pub use crate::voxel::{VoxelBlock, VoxelScene, VoxelVertex};
}
