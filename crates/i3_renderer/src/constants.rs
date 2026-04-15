/// Maximum number of meshes in the mesh descriptor buffer.
pub const MAX_MESHES: u64 = 16384;

/// Maximum number of instances in the instance buffer and draw call buffer.
pub const MAX_INSTANCES: u64 = 262144;

/// Maximum number of lights processed per frame.
pub const MAX_LIGHTS: u64 = 1024;

/// Number of depth slices in the cluster grid (Z axis).
pub const CLUSTER_GRID_Z: u32 = 16;

/// Tile size in pixels for cluster X/Y grid dimensions.
pub const CLUSTER_TILE_SIZE: u32 = 64;

/// Maximum number of lights that can be assigned to a single cluster.
/// Must match MAX_LIGHTS_PER_CLUSTER in light_cull.slang.
pub const MAX_LIGHTS_PER_CLUSTER: u64 = 512;

/// Size in bytes of a `vkDrawIndexedIndirectCommand` (index_count, instance_count,
/// first_index, vertex_offset, first_instance — 5 × u32).
pub const DRAW_INDIRECT_CMD_SIZE: u64 = 20;

/// Maximum number of materials in the material buffer.
pub const MAX_MATERIALS: u64 = 65536;

/// Fixed resolution of the PreZ depth buffer used to build HiZPreZ.
/// Both dimensions are powers of two → exact halving at every mip level, no edge cases.
/// The PreZ renders the same frustum as the main pass, so HiZ UV [0,1] == screen UV [0,1].
pub const HIZ_PREZ_WIDTH:  u32 = 1024;
pub const HIZ_PREZ_HEIGHT: u32 = 512;
