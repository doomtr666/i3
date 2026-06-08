pub mod gpu_scene;
pub mod gpu_buffers;
pub mod svo;
pub mod error_metric;
pub mod noise_graph;
pub mod passes;

#[cfg(feature = "debug-ui")]
pub mod debug_ui;

pub use svo::{SvoTree, SvoNode, SvoState};
pub use gpu_buffers::SvoGpuBuffers;
pub use noise_graph::{NoiseGraph, VmOp};
pub use passes::create_svo_passes;

#[cfg(feature = "debug-ui")]
pub use debug_ui::{SvoParams, SvoDebugUi};

// ─── Brick layout constants ───────────────────────────────────────────────────
pub const BRICK_SIZE:   u32   = 8;
pub const BRICK_VOXELS: usize = 729;  // (BRICK_SIZE+1)³ = 9³, with +1 overlap

/// Bytes per voxel in the geometry atlas. Layout (little-endian, 8-byte aligned):
///   bytes 0-1  : signed distance      f16 (raw world units — no normalisation)
///   bytes 2-3  : octahedral normal X  snorm16
///   bytes 4-5  : octahedral normal Y  snorm16
///   byte  6    : material id          u8
///   byte  7    : flags                u8  (edited / veg-hint / reserved)
/// Must match the (un)packing in svo_bake.slang / svo_render.slang.
pub const VOXEL_BYTES: usize = 8;
pub const BRICK_BYTES: usize = BRICK_VOXELS * VOXEL_BYTES;  // 729 * 8 = 5832

// ─── SVO capacity constants ───────────────────────────────────────────────────
pub const MAX_SVO_NODES:  u32 = 262144;
pub const MAX_SVO_BRICKS: u32 = 49152;
pub const MAX_SVO_DEPTH:  u32 = 16;
pub const MAX_PRIMS:      u32 = 256;

/// Node side (world metres) every visible near-surface region is refined down to,
/// regardless of curvature — guarantees small objects (sub-voxel in coarse nodes)
/// are resolved. Below this, only curved regions keep subdividing.
pub const BASE_SIZE: f32 = 4.0;

/// Below this node side (world metres), EMPTY nodes (conservatively near a surface
/// but not actually crossed by one) stop subdividing — only surface-crossing nodes
/// keep refining. Keeps the empty shell coarse so the ray-marcher skips it cheaply
/// (avoids step-budget exhaustion when close). Should be ≲ the smallest feature so
/// `crosses_surface` still catches thin geometry (e.g. the torus tube) by this size.
pub const EMPTY_CAP: f32 = 0.5;
