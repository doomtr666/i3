// ─── CPU voxel/SDF model ──────────────────────────────────────────────────────
pub mod octree;
pub mod sdf;
pub mod voxel;

// ─── GPU clipmap renderer (was crate i3_sdf — folded in: the voxel feature is one whole) ─
pub mod gpu_scene;
pub mod gpu_buffers;
pub mod clipmap;
pub mod noise_graph;
pub mod passes;

#[cfg(feature = "debug-ui")]
pub mod debug_ui;

pub use i3_math::{AABB, Transform};
pub use octree::{VoxelOctree, VoxelSceneSink};
pub use sdf::{BvhNode, SdfNode, SdfPrimitive, SdfScene};
pub use voxel::{VoxelBlock, VoxelScene, VoxelVertex};

pub use clipmap::{ClipmapState, ClipLevelGpu, NUM_LEVELS, CLIPMAP_GRID, GRID_VOL, MAX_BRICKS_PER_LEVEL};
pub use gpu_buffers::ClipmapBuffers;
pub use gpu_scene::TerrainMatParams;
pub use noise_graph::{NoiseGraph, VmOp};
pub use passes::create_clipmap_passes;

#[cfg(feature = "debug-ui")]
pub use debug_ui::{ClipmapParams, ClipmapDebugUi};

pub mod prelude {
    pub use i3_math::prelude::*;
    pub use crate::octree::{VoxelOctree, VoxelSceneSink};
    pub use crate::sdf::{SdfPrimitive, SdfScene};
    pub use crate::voxel::{VoxelBlock, VoxelScene, VoxelVertex};
}

// ─── Brick layout constants ───────────────────────────────────────────────────
pub const BRICK_SIZE:   u32   = 8;
pub const BRICK_VOXELS: usize = 729;  // (BRICK_SIZE+1)³ = 9³, with +1 overlap

/// Bytes per voxel in the geometry atlas. Layout (little-endian, 8-byte aligned):
///   bytes 0-1  : signed distance      f16 (raw world units — no normalisation)
///   bytes 2-3  : octahedral normal X  snorm16
///   bytes 4-5  : octahedral normal Y  snorm16
///   byte  6    : material id          u8
///   byte  7    : flags                u8  (edited / veg-hint / reserved)
/// Must match the (un)packing in clipmap_bake.slang / clipmap_render.slang.
pub const VOXEL_BYTES: usize = 8;
pub const BRICK_BYTES: usize = BRICK_VOXELS * VOXEL_BYTES;  // 729 * 8 = 5832

/// Pre-mix weight atlas: 4 bytes/voxel (RGBA8 = the 4 terrain layer blend weights),
/// baked ONCE per brick (the slope/height/biome mix is low-frequency and only changes
/// on LOD/edit — like virtual texturing), then read + weighted-sampled per pixel. Half
/// the geom voxel size, so its byte address is exactly `(geom_addr) >> 1`.
pub const WEIGHT_VOXEL_BYTES: usize = 4;
pub const WEIGHT_BRICK_BYTES: usize = BRICK_VOXELS * WEIGHT_VOXEL_BYTES;  // 729 * 4 = 2916

// ─── Atlas capacity ───────────────────────────────────────────────────────────
/// Total brick slots in the geometry/weight atlas. Partitioned across the clipmap
/// cascade as `NUM_LEVELS · MAX_BRICKS_PER_LEVEL` (see `clipmap`).
pub const MAX_SVO_BRICKS: u32 = 49152;
pub const MAX_PRIMS:      u32 = 256;
