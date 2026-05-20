use i3_math::nalgebra::Vector3;
use i3_voxel::{BvhNode, SdfPrimitive, SdfScene};

// ─── GpuBrickJob ─────────────────────────────────────────────────────────────
// One entry per brick to bake this frame.  Uploaded to a CpuToGpu buffer,
// consumed by the compute shader (one workgroup = one job).

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuBrickJob {
    pub brick_world_min: [f32; 3],
    pub voxel_size:      f32,
    /// Absolute index into the flat GPU atlas: (lev*MAX_BRICKS + brick_idx) * BRICK_VOXELS
    pub atlas_offset:    u32,
    /// Brick half-diagonal in world space (not used by shader, kept for debug)
    pub half_diag:       f32,
    pub _pad:            [u32; 2],
}

// ─── GpuPrimitive ────────────────────────────────────────────────────────────
// 80-byte packed primitive for the compute shader.
//
// Transform unpacking in the shader:
//   p_centered = p_world - float3(row0.w, row1.w, row2.w)
//   p_rotated  = float3(dot(row0.xyz, p_centered),
//                       dot(row1.xyz, p_centered),
//                       dot(row2.xyz, p_centered))
//   p_local    = p_rotated * inv_scale
//   d_world    = evalLocal(p_local) / inv_scale      (= evalLocal * scale)

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuPrimitive {
    /// Rows of the inverse-rotation matrix packed with world-space translation:
    ///   row_i.xyz = row i of inv_rot,  row_i.w = translation component i
    pub row0:      [f32; 4],
    pub row1:      [f32; 4],
    pub row2:      [f32; 4],
    pub inv_scale: f32,
    pub data0:     f32,   // Sphere: radius | Box: he.x | Capsule/Cylinder: half_height | Torus: major_r
    pub data1:     f32,   // Box: he.y | Capsule/Cylinder: radius | Torus: minor_r
    pub data2:     f32,   // Box: he.z
    pub prim_type:   u32,   // 0=Sphere  1=Box  2=Capsule  3=Cylinder  4=Torus
    pub subtract:    u32,   // 0 or 1
    pub material_id: u32,
    pub _pad:        u32,
}

impl GpuPrimitive {
    pub const SPHERE:   u32 = 0;
    pub const BOX:      u32 = 1;
    pub const CAPSULE:  u32 = 2;
    pub const CYLINDER: u32 = 3;
    pub const TORUS:    u32 = 4;
}

// ─── Packing ─────────────────────────────────────────────────────────────────

/// Pack one SDF node into a `GpuPrimitive`.
/// Returns `None` for `TerrainBox` (function pointer, cannot be serialised).
pub fn pack_node(node: &i3_voxel::SdfNode) -> Option<GpuPrimitive> {
    let tf = node.transform();
    let t  = tf.translation(); // world-space centre

    // Rows of inv_rot extracted via basis-vector application:
    //   inv_transform_normal(e_i) = inv_rot * e_i  = i-th COLUMN of inv_rot
    // Transposing: row_j[i] = col_i[j]
    let col0 = tf.inv_transform_normal(&Vector3::x());
    let col1 = tf.inv_transform_normal(&Vector3::y());
    let col2 = tf.inv_transform_normal(&Vector3::z());
    let row0 = [col0.x, col1.x, col2.x, t.x];
    let row1 = [col0.y, col1.y, col2.y, t.y];
    let row2 = [col0.z, col1.z, col2.z, t.z];
    let inv_scale = tf.inv_scale();
    let subtract    = node.is_subtract() as u32;
    let material_id = node.material_id();

    let prim = match node.primitive() {
        SdfPrimitive::Sphere { radius } => GpuPrimitive {
            row0, row1, row2, inv_scale,
            data0: *radius, data1: 0.0, data2: 0.0,
            prim_type: GpuPrimitive::SPHERE, subtract, material_id, _pad: 0,
        },
        SdfPrimitive::Box { half_extents } => GpuPrimitive {
            row0, row1, row2, inv_scale,
            data0: half_extents.x, data1: half_extents.y, data2: half_extents.z,
            prim_type: GpuPrimitive::BOX, subtract, material_id, _pad: 0,
        },
        SdfPrimitive::Capsule { half_height, radius } => GpuPrimitive {
            row0, row1, row2, inv_scale,
            data0: *half_height, data1: *radius, data2: 0.0,
            prim_type: GpuPrimitive::CAPSULE, subtract, material_id, _pad: 0,
        },
        SdfPrimitive::Cylinder { half_height, radius } => GpuPrimitive {
            row0, row1, row2, inv_scale,
            data0: *half_height, data1: *radius, data2: 0.0,
            prim_type: GpuPrimitive::CYLINDER, subtract, material_id, _pad: 0,
        },
        SdfPrimitive::Torus { major_radius, minor_radius } => GpuPrimitive {
            row0, row1, row2, inv_scale,
            data0: *major_radius, data1: *minor_radius, data2: 0.0,
            prim_type: GpuPrimitive::TORUS, subtract, material_id, _pad: 0,
        },
        SdfPrimitive::TerrainBox { .. } => return None,
    };

    Some(prim)
}

/// Pack every node in a scene (skips TerrainBox).
pub fn pack_scene(scene: &SdfScene) -> Vec<GpuPrimitive> {
    scene.nodes().iter().filter_map(pack_node).collect()
}

// ─── GpuBvhNode ──────────────────────────────────────────────────────────────

/// 32-byte BVH node for GPU traversal.
/// Leaf: left == u32::MAX, right_or_prim = primitive index.
/// Internal: left/right = child indices.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuBvhNode {
    pub aabb_min:      [f32; 3],
    pub left:          u32,
    pub aabb_max:      [f32; 3],
    pub right_or_prim: u32,
}

pub fn pack_bvh(scene: &SdfScene) -> Vec<GpuBvhNode> {
    scene.bvh_nodes().iter().map(|n: &BvhNode| {
        let right_or_prim = if n.left == u32::MAX { n.prim_idx } else { n.right };
        GpuBvhNode {
            aabb_min:      [n.aabb.min.x, n.aabb.min.y, n.aabb.min.z],
            left:          n.left,
            aabb_max:      [n.aabb.max.x, n.aabb.max.y, n.aabb.max.z],
            right_or_prim,
        }
    }).collect()
}
