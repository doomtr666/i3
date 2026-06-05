pub mod gpu_scene;
pub mod gpu_buffers;
pub mod svo;
pub mod error_metric;
pub mod passes;

#[cfg(feature = "debug-ui")]
pub mod debug_ui;

pub use svo::{SvoTree, SvoNode, SvoState};
pub use gpu_buffers::SvoGpuBuffers;
pub use passes::create_svo_passes;

#[cfg(feature = "debug-ui")]
pub use debug_ui::{SvoParams, SvoDebugUi};

// ─── Brick layout constants ───────────────────────────────────────────────────
pub const BRICK_SIZE:   u32   = 8;
pub const BRICK_VOXELS: usize = 729;  // (BRICK_SIZE+1)³ = 9³
pub const BRICK_DWORDS: usize = 183;  // ceil(729/4)

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

/// SDF quantisation half-range, in voxels. The brick stores sdf/(BAND·voxel_size)
/// as u8. Tighter than the brick half-diagonal (~7 voxels) → finer precision near
/// the surface → smoother gradients/normals. Must match `BAND` in svo_render.slang.
pub const BAND_VOXELS: f32 = 3.0;
