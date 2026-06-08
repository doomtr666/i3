use i3_math::nalgebra::Vector3;
use i3_voxel::{BvhNode, SdfPrimitive, SdfScene};

// ─── GpuSvoNode ───────────────────────────────────────────────────────────────
// 32 bytes. Mirrors the Slang struct in svo_render.slang.
// first_child == u32::MAX  → leaf node.
// brick_offset == u32::MAX → no baked brick yet (renders as empty).

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuSvoNode {
    pub aabb_min:     [f32; 3],
    pub first_child:  u32,
    pub aabb_max:     [f32; 3],
    pub brick_offset: u32,
}

// ─── GpuBrickJob ─────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuBrickJob {
    pub brick_world_min: [f32; 3],
    pub voxel_size:      f32,
    pub atlas_offset:    u32,      // byte offset into the geometry atlas
    pub _pad:            [u32; 3],
}

// ─── GpuPrimitive ────────────────────────────────────────────────────────────
// 80 bytes. Same layout as in brickmap_bake.slang.

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuPrimitive {
    pub row0:        [f32; 4],
    pub row1:        [f32; 4],
    pub row2:        [f32; 4],
    pub inv_scale:   f32,
    pub data0:       f32,
    pub data1:       f32,
    pub data2:       f32,
    pub prim_type:   u32,
    pub subtract:    u32,
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

// ─── GpuBvhNode ──────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuBvhNode {
    pub aabb_min:      [f32; 3],
    pub left:          u32,
    pub aabb_max:      [f32; 3],
    pub right_or_prim: u32,
}

// ─── TerrainMatParams ──────────────────────────────────────────────────────────
// Terrain texture layer bindless indices + slope/height blend params, pushed to
// svo_render. Layer order: 0=grass, 1=rock, 2=snow, 3=dirt. Index -1 = no texture
// (the shader falls back to a solid colour). ORM = (R=AO, G=Roughness, B=Metallic).
#[derive(Clone, Copy)]
pub struct TerrainMatParams {
    pub albedo:     [i32; 4],
    pub normal:     [i32; 4],
    pub orm:        [i32; 4],
    pub snow_lo:    f32,
    pub snow_hi:    f32,
    pub slope_lo:   f32,   // below this slope (n.y) → fully rock
    pub slope_hi:   f32,   // above this → no rock
    pub dirt_lo:    f32,
    pub dirt_hi:    f32,
    pub tiling:     f32,   // world→uv scale (tiles ≈ 1/tiling metres)
    pub jitter_amp: f32,   // fBm height jitter (m) so bands aren't horizontal
}

impl Default for TerrainMatParams {
    fn default() -> Self {
        Self {
            albedo: [-1; 4], normal: [-1; 4], orm: [-1; 4],
            snow_lo: 9.0, snow_hi: 17.0, slope_lo: 0.45, slope_hi: 0.72,
            dirt_lo: -9.0, dirt_hi: -3.0, tiling: 0.15, jitter_amp: 6.0,
        }
    }
}

// ─── Packing functions ────────────────────────────────────────────────────────

pub fn pack_node(node: &i3_voxel::SdfNode) -> Option<GpuPrimitive> {
    let tf = node.transform();
    let t  = tf.translation();

    let col0 = tf.inv_transform_normal(&Vector3::x());
    let col1 = tf.inv_transform_normal(&Vector3::y());
    let col2 = tf.inv_transform_normal(&Vector3::z());
    let row0 = [col0.x, col1.x, col2.x, t.x];
    let row1 = [col0.y, col1.y, col2.y, t.y];
    let row2 = [col0.z, col1.z, col2.z, t.z];
    let inv_scale    = tf.inv_scale();
    let subtract     = node.is_subtract() as u32;
    let material_id  = node.material_id();

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
        // Terrain is evaluated by the bake's noise library (terrainDensity), not as an
        // analytic primitive — skip here; the setup pass flags terrain separately.
        SdfPrimitive::TerrainBox { .. } | SdfPrimitive::VolumeTerrain { .. } => return None,
    };
    Some(prim)
}

pub fn pack_scene(scene: &SdfScene) -> Vec<GpuPrimitive> {
    scene.nodes().iter().filter_map(pack_node).collect()
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
