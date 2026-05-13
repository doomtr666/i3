use crate::sdf::{SdfNode, SdfScene};
use i3_math::AABB;

use nalgebra::{Point3, Vector3, point};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::Arc;

const VOXEL_BLOCK_WIDTH: i32 = 31;
const VOXEL_BLOCK_PADDED_WIDTH: i32 = VOXEL_BLOCK_WIDTH + 1;
const VOXEL_BLOCK_PADDED_WIDTH2: i32 = VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH;
const VOXEL_BLOCK_PADDED_SIZE: i32 =
    VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH;

// SDF cache covers [-1, BLOCK_WIDTH] per axis = BLOCK_WIDTH+2 points per axis.
const VOXEL_SDF_CACHE_WIDTH: i32 = VOXEL_BLOCK_WIDTH + 2;
const VOXEL_SDF_CACHE_SIZE: usize =
    (VOXEL_SDF_CACHE_WIDTH * VOXEL_SDF_CACHE_WIDTH * VOXEL_SDF_CACHE_WIDTH) as usize;

const VOXEL_SCENE_WIDTH: i32 = 8;

pub const VOXEL_DIST: f32 = 0.05;
const BISECTION_STEPS: usize = 8;

// u32::MAX = absent (sentinel for the flat vertex-map lookup table).
const VERTEX_MAP_EMPTY: u32 = u32::MAX;

thread_local! {
    static SDF_SCRATCH: RefCell<Vec<f32>> =
        RefCell::new(vec![0.0_f32; VOXEL_SDF_CACHE_SIZE]);
    static VERTEX_SCRATCH: RefCell<(Vec<VoxelVertex>, Vec<bool>)> =
        RefCell::new((
            vec![VoxelVertex::default(); VOXEL_BLOCK_PADDED_SIZE as usize],
            vec![false; VOXEL_BLOCK_PADDED_SIZE as usize],
        ));
    // Flat lookup replacing HashMap<[i32;3], u32> — indexed by vertex_idx, no heap allocation.
    static VERTEX_MAP_SCRATCH: RefCell<Vec<u32>> =
        RefCell::new(vec![VERTEX_MAP_EMPTY; VOXEL_BLOCK_PADDED_SIZE as usize]);
}

#[derive(Clone)]
pub struct VoxelVertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
}

impl Default for VoxelVertex {
    fn default() -> Self {
        VoxelVertex {
            position: Point3::origin(),
            normal: Vector3::zeros(),
        }
    }
}

pub struct VoxelBlock {
    sdf: Arc<SdfScene>,
    px: i32,
    py: i32,
    pz: i32,
    packed_vertices: Vec<VoxelVertex>,
    packed_indices: Vec<u32>,
}

#[inline]
fn vertex_idx(x: i32, y: i32, z: i32) -> usize {
    ((x + 1) + (y + 1) * VOXEL_BLOCK_PADDED_WIDTH + (z + 1) * VOXEL_BLOCK_PADDED_WIDTH2) as usize
}

#[inline]
fn sdf_cache_idx(x: i32, y: i32, z: i32) -> usize {
    ((x + 1)
        + (y + 1) * VOXEL_SDF_CACHE_WIDTH
        + (z + 1) * VOXEL_SDF_CACHE_WIDTH * VOXEL_SDF_CACHE_WIDTH) as usize
}

impl VoxelBlock {
    pub fn new(sdf: Arc<SdfScene>, px: i32, py: i32, pz: i32) -> Self {
        Self {
            sdf,
            px,
            py,
            pz,
            packed_vertices: Vec::new(),
            packed_indices: Vec::new(),
        }
    }

    #[inline(always)]
    fn is_solid(sdf_value: f32) -> bool {
        sdf_value < 0.0
    }

    pub fn world_aabb(&self) -> AABB {
        AABB::new(
            Point3::from(self.world_min()),
            Point3::from(self.world_max()),
        )
    }

    pub fn world_min(&self) -> [f32; 3] {
        [
            (self.px * VOXEL_BLOCK_WIDTH) as f32 * VOXEL_DIST,
            (self.py * VOXEL_BLOCK_WIDTH) as f32 * VOXEL_DIST,
            (self.pz * VOXEL_BLOCK_WIDTH) as f32 * VOXEL_DIST,
        ]
    }

    pub fn world_max(&self) -> [f32; 3] {
        [
            ((self.px + 1) * VOXEL_BLOCK_WIDTH) as f32 * VOXEL_DIST,
            ((self.py + 1) * VOXEL_BLOCK_WIDTH) as f32 * VOXEL_DIST,
            ((self.pz + 1) * VOXEL_BLOCK_WIDTH) as f32 * VOXEL_DIST,
        ]
    }

    pub fn get_packed_vertices(&self) -> &[VoxelVertex] {
        &self.packed_vertices
    }

    pub fn get_packed_indices(&self) -> &[u32] {
        &self.packed_indices
    }

    pub fn compute_mesh(&mut self) {
        let sdf = Arc::clone(&self.sdf);
        let local_aabb = self.world_aabb().expand(2.0 * VOXEL_DIST);
        let sdf_nodes = sdf.get_nodes(&local_aabb);
        if sdf_nodes.is_empty() {
            return;
        }

        SDF_SCRATCH.with(|sc| {
            VERTEX_SCRATCH.with(|vc| {
                VERTEX_MAP_SCRATCH.with(|vm| {
                    let mut sdf_cache = sc.borrow_mut();
                    let mut vc_ref = vc.borrow_mut();
                    let (verts, valid) = &mut *vc_ref;
                    let mut vertex_map = vm.borrow_mut();
                    valid.fill(false);
                    vertex_map.fill(VERTEX_MAP_EMPTY);
                    self.fill_sdf_cache(&sdf_nodes, &mut sdf_cache);
                    self.compute_vertices_cached(&sdf_nodes, &sdf_cache, verts, valid);
                    self.compute_indices_cached(&sdf_cache, verts, valid, &mut vertex_map);
                });
            });
        });
    }

    fn fill_sdf_cache(&self, sdf_nodes: &[&SdfNode], cache: &mut Vec<f32>) {
        for z in -1..=VOXEL_BLOCK_WIDTH {
            for y in -1..=VOXEL_BLOCK_WIDTH {
                for x in -1..=VOXEL_BLOCK_WIDTH {
                    cache[sdf_cache_idx(x, y, z)] =
                        SdfScene::value(sdf_nodes, &self.world_pos(x, y, z));
                }
            }
        }
    }

    fn compute_vertices_cached(
        &self,
        sdf_nodes: &[&SdfNode],
        sdf_cache: &[f32],
        verts: &mut Vec<VoxelVertex>,
        valid: &mut Vec<bool>,
    ) {
        for z in -1..=(VOXEL_BLOCK_WIDTH - 1) {
            for y in -1..=(VOXEL_BLOCK_WIDTH - 1) {
                for x in -1..=(VOXEL_BLOCK_WIDTH - 1) {
                    let mut vertices = [Point3::origin(); 8];
                    let mut values = [0.0f32; 8];
                    for i in 0..8 {
                        let ix = (i & 1) as i32;
                        let iy = ((i >> 1) & 1) as i32;
                        let iz = ((i >> 2) & 1) as i32;
                        vertices[i] = self.world_pos(x + ix, y + iy, z + iz);
                        values[i] = sdf_cache[sdf_cache_idx(x + ix, y + iy, z + iz)];
                    }

                    if !Self::check_intersection(&values) {
                        continue;
                    }

                    let edge_indices = [
                        (0, 1),
                        (1, 3),
                        (3, 2),
                        (2, 0),
                        (4, 5),
                        (5, 7),
                        (7, 6),
                        (6, 4),
                        (0, 4),
                        (1, 5),
                        (2, 6),
                        (3, 7),
                    ];

                    let mut center_of_mass = Vector3::<f32>::zeros();
                    let mut intersect_count = 0.0f32;
                    let mut accumulated_normal = Vector3::<f32>::zeros();

                    for (a, b) in edge_indices {
                        let v1 = vertices[a];
                        let v2 = vertices[b];
                        let fa = values[a];
                        let fb = values[b];

                        if Self::is_solid(fa) != Self::is_solid(fb) {
                            let q = {
                                let (mut lo, mut hi) = (0.0_f32, 1.0_f32);
                                let mut flo = fa;
                                for _ in 0..BISECTION_STEPS {
                                    let mid = (lo + hi) * 0.5;
                                    let p = v1 + mid * (v2 - v1);
                                    let fmid = SdfScene::value(sdf_nodes, &p);
                                    if flo * fmid <= 0.0 {
                                        hi = mid;
                                    } else {
                                        lo = mid;
                                        flo = fmid;
                                    }
                                }
                                v1 + ((lo + hi) * 0.5) * (v2 - v1)
                            };
                            let n = SdfScene::normal(sdf_nodes, &q);
                            accumulated_normal += n;
                            center_of_mass += q.coords;
                            intersect_count += 1.0;
                        }
                    }

                    if intersect_count == 0.0 {
                        continue;
                    }

                    let position = Point3::from(center_of_mass / intersect_count);
                    let final_normal = accumulated_normal
                        .try_normalize(1e-6)
                        .unwrap_or_else(Vector3::y);

                    let idx = vertex_idx(x, y, z);
                    verts[idx] = VoxelVertex {
                        position,
                        normal: final_normal,
                    };
                    valid[idx] = true;
                }
            }
        }
    }

    fn compute_indices_cached(
        &mut self,
        sdf_cache: &[f32],
        verts: &[VoxelVertex],
        valid: &[bool],
        vertex_map: &mut Vec<u32>,
    ) {
        for z in 0..VOXEL_BLOCK_WIDTH {
            for y in 0..VOXEL_BLOCK_WIDTH {
                for x in 0..VOXEL_BLOCK_WIDTH {
                    let sv = sdf_cache[sdf_cache_idx(x, y, z)];
                    let xp_v = sdf_cache[sdf_cache_idx(x + 1, y, z)];
                    let yp_v = sdf_cache[sdf_cache_idx(x, y + 1, z)];
                    let zp_v = sdf_cache[sdf_cache_idx(x, y, z + 1)];
                    let ccw = !Self::is_solid(sv);

                    Self::emit_quad(
                        verts,
                        valid,
                        vertex_map,
                        &mut self.packed_vertices,
                        &mut self.packed_indices,
                        sv,
                        xp_v,
                        ccw,
                        [x, y, z],
                        [x, y, z - 1],
                        [x, y - 1, z - 1],
                        [x, y - 1, z],
                    );
                    Self::emit_quad(
                        verts,
                        valid,
                        vertex_map,
                        &mut self.packed_vertices,
                        &mut self.packed_indices,
                        sv,
                        yp_v,
                        ccw,
                        [x, y, z],
                        [x - 1, y, z],
                        [x - 1, y, z - 1],
                        [x, y, z - 1],
                    );
                    Self::emit_quad(
                        verts,
                        valid,
                        vertex_map,
                        &mut self.packed_vertices,
                        &mut self.packed_indices,
                        sv,
                        zp_v,
                        ccw,
                        [x, y, z],
                        [x, y - 1, z],
                        [x - 1, y - 1, z],
                        [x - 1, y, z],
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_quad(
        verts: &[VoxelVertex],
        valid: &[bool],
        vertex_map: &mut Vec<u32>,
        packed_vertices: &mut Vec<VoxelVertex>,
        packed_indices: &mut Vec<u32>,
        start_v: f32,
        neighbor_v: f32,
        ccw: bool,
        v0: [i32; 3],
        v1: [i32; 3],
        v2: [i32; 3],
        v3: [i32; 3],
    ) {
        if Self::is_solid(start_v) == Self::is_solid(neighbor_v) {
            return;
        }

        let indices = {
            let mut get_or_add = |coords: [i32; 3]| -> Option<u32> {
                let [x, y, z] = coords;
                let vi = vertex_idx(x, y, z);
                if !valid[vi] {
                    return None;
                }
                let slot = &mut vertex_map[vi];
                if *slot != VERTEX_MAP_EMPTY {
                    return Some(*slot);
                }
                let idx = packed_vertices.len() as u32;
                packed_vertices.push(verts[vi].clone());
                *slot = idx;
                Some(idx)
            };
            (
                get_or_add(v0),
                get_or_add(v1),
                get_or_add(v2),
                get_or_add(v3),
            )
        };

        if let (Some(i0), Some(i1), Some(i2), Some(i3)) = indices {
            let p = |i: u32| packed_vertices[i as usize].position;
            let d02 = (p(i2) - p(i0)).norm_squared();
            let d13 = (p(i3) - p(i1)).norm_squared();

            if d02 <= d13 {
                if ccw {
                    packed_indices.extend([i0, i3, i2, i0, i2, i1]);
                } else {
                    packed_indices.extend([i0, i1, i2, i0, i2, i3]);
                }
            } else if ccw {
                packed_indices.extend([i0, i3, i1, i3, i2, i1]);
            } else {
                packed_indices.extend([i0, i1, i3, i1, i2, i3]);
            }
        }
    }

    fn check_intersection(v: &[f32; 8]) -> bool {
        let base_solid = Self::is_solid(v[0]);
        (1..8).any(|i| Self::is_solid(v[i]) != base_solid)
    }

    fn world_pos(&self, x: i32, y: i32, z: i32) -> Point3<f32> {
        point![
            (self.px * VOXEL_BLOCK_WIDTH + x) as f32 * VOXEL_DIST,
            (self.py * VOXEL_BLOCK_WIDTH + y) as f32 * VOXEL_DIST,
            (self.pz * VOXEL_BLOCK_WIDTH + z) as f32 * VOXEL_DIST,
        ]
    }
}

pub struct VoxelScene {
    blocks: Vec<VoxelBlock>,
}

impl VoxelScene {
    pub fn new(sdf: Arc<SdfScene>) -> VoxelScene {
        let mut blocks = Vec::new();

        for z in 0..VOXEL_SCENE_WIDTH {
            for y in 0..VOXEL_SCENE_WIDTH {
                for x in 0..VOXEL_SCENE_WIDTH {
                    blocks.push(VoxelBlock::new(Arc::clone(&sdf), x, y, z));
                }
            }
        }

        VoxelScene { blocks }
    }

    pub fn compute_meshes(&mut self) {
        self.blocks.par_iter_mut().for_each(|b| b.compute_mesh());
    }

    pub fn iter(&self) -> std::slice::Iter<'_, VoxelBlock> {
        self.blocks.iter()
    }

    pub fn voxel_dist(&self) -> f32 {
        VOXEL_DIST
    }

    pub fn voxel_width(&self) -> i32 {
        VOXEL_BLOCK_WIDTH
    }

    pub fn world_min(&self) -> [f32; 3] {
        [0.0, 0.0, 0.0]
    }

    pub fn world_max(&self) -> [f32; 3] {
        [
            (VOXEL_SCENE_WIDTH * VOXEL_BLOCK_WIDTH) as f32 * VOXEL_DIST,
            (VOXEL_SCENE_WIDTH * VOXEL_BLOCK_WIDTH) as f32 * VOXEL_DIST,
            (VOXEL_SCENE_WIDTH * VOXEL_BLOCK_WIDTH) as f32 * VOXEL_DIST,
        ]
    }
}

impl<'a> IntoIterator for &'a VoxelScene {
    type Item = &'a VoxelBlock;
    type IntoIter = std::slice::Iter<'a, VoxelBlock>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
