use crate::sdf::{SdfNode, SdfScene};
use i3_math::AABB;

use nalgebra::{Matrix3, Point3, Vector3, point};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

const VOXEL_BLOCK_WIDTH: i32 = 31;
const VOXEL_BLOCK_PADDED_WIDTH: i32 = VOXEL_BLOCK_WIDTH + 1;
const VOXEL_BLOCK_PADDED_WIDTH2: i32 = VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH;
const VOXEL_BLOCK_PADDED_SIZE: i32 =
    VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH;

const VOXEL_SCENE_WIDTH: i32 = 8;

pub const VOXEL_DIST: f32 = 0.05;
//const TOLERANCE_SVD: f32 = 1e-4;
const BISECTION_STEPS: usize = 8;
//const NEWTON_STEPS: usize = 2;

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
    vertices: Vec<Option<VoxelVertex>>,
    packed_vertices: Vec<VoxelVertex>,
    packed_indices: Vec<u32>,
}

#[inline]
fn vertex_idx(x: i32, y: i32, z: i32) -> usize {
    ((x + 1) + (y + 1) * VOXEL_BLOCK_PADDED_WIDTH + (z + 1) * VOXEL_BLOCK_PADDED_WIDTH2) as usize
}

impl VoxelBlock {
    pub fn new(sdf: Arc<SdfScene>, px: i32, py: i32, pz: i32) -> Self {
        Self {
            sdf,
            px,
            py,
            pz,
            vertices: vec![None; VOXEL_BLOCK_PADDED_SIZE as usize],
            packed_vertices: Vec::new(),
            packed_indices: Vec::new(),
        }
    }

    #[inline(always)]
    fn is_solid(sdf_value: f32) -> bool {
        // Defines the absolute boundary.
        // < 0.0 is solid matter, >= 0.0 is empty air.
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

    pub fn get_vertices(&self) -> &[Option<VoxelVertex>] {
        &self.vertices
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

        self.compute_vertices(&sdf_nodes);
        self.compute_indices(&sdf_nodes);
    }

    fn compute_vertices(&mut self, sdf_nodes: &[&SdfNode]) {
        for z in -1..=(VOXEL_BLOCK_WIDTH as i32 - 1) {
            for y in -1..=(VOXEL_BLOCK_WIDTH as i32 - 1) {
                for x in -1..=(VOXEL_BLOCK_WIDTH as i32 - 1) {
                    let mut vertices = [Point3::origin(); 8];
                    let mut values = [0.0f32; 8];
                    for i in 0..8 {
                        let p = self.world_pos(x + (i & 1), y + ((i >> 1) & 1), z + ((i >> 2) & 1));
                        vertices[i as usize] = p;
                        values[i as usize] = SdfScene::value(sdf_nodes, &p);
                    }

                    if !self.check_intersection(&values) {
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

                    let mut mass_mat = Matrix3::<f32>::zeros();
                    let mut mass_vec = Vector3::<f32>::zeros();
                    let mut center_of_mass = Vector3::<f32>::zeros();
                    let mut intersect_count = 0.0f32;

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
                            mass_mat += n * n.transpose();
                            mass_vec += n * n.dot(&q.coords);
                            center_of_mass += q.coords;
                            intersect_count += 1.0;
                        }
                    }

                    if intersect_count == 0.0 {
                        continue;
                    }

                    center_of_mass /= intersect_count;

                    /*

                    let rebase = center_of_mass;
                    let mass_vec_r = mass_vec - mass_mat * rebase;

                    let svd = mass_mat.svd(true, true);
                    let qef_result = match svd.pseudo_inverse(TOLERANCE_SVD) {
                        Ok(pseudo_inv) => Point3::from(rebase + pseudo_inv * mass_vec_r),
                        Err(_) => Point3::from(center_of_mass),
                    };

                    let mut projected = qef_result;

                    let voxel_min = self.world_pos(x, y, z);
                    let voxel_max = self.world_pos(x + 1, y + 1, z + 1);

                    let voxel_aabb = AABB::new(
                        Point3::from(voxel_min.coords),
                        Point3::from(voxel_max.coords),
                    );

                    if !voxel_aabb.contains(&qef_result) {
                        projected = Point3::from(center_of_mass);
                    }

                    for _ in 0..NEWTON_STEPS {
                        let f = SdfScene::value(sdf_nodes, &projected);
                        let n = SdfScene::normal(sdf_nodes, &projected);
                        projected = Point3::from(projected.coords - f * n * 0.5);
                    }

                    let position = voxel_aabb.clamp(&projected);
                    let final_normal = SdfScene::normal(sdf_nodes, &position);

                    self.vertices[vertex_idx(x, y, z)] = Some(VoxelVertex {
                        position,
                        normal: final_normal,
                    });
                    */

                    let position = center_of_mass.into();
                    let final_normal = SdfScene::normal(sdf_nodes, &position);

                    self.vertices[vertex_idx(x, y, z)] = Some(VoxelVertex {
                        position,
                        normal: final_normal,
                    });
                }
            }
        }
    }

    fn compute_indices(&mut self, sdf_nodes: &[&SdfNode]) {
        let mut vertex_map = HashMap::<[i32; 3], u32>::new();

        for z in 0..VOXEL_BLOCK_WIDTH as i32 {
            for y in 0..VOXEL_BLOCK_WIDTH as i32 {
                for x in 0..VOXEL_BLOCK_WIDTH as i32 {
                    let sv = SdfScene::value(sdf_nodes, &self.world_pos(x, y, z));
                    let ccw = !Self::is_solid(sv);

                    let xp_v = SdfScene::value(sdf_nodes, &self.world_pos(x + 1, y, z));
                    let yp_v = SdfScene::value(sdf_nodes, &self.world_pos(x, y + 1, z));
                    let zp_v = SdfScene::value(sdf_nodes, &self.world_pos(x, y, z + 1));

                    Self::emit_quad(
                        &self.vertices,
                        &mut self.packed_vertices,
                        &mut self.packed_indices,
                        &mut vertex_map,
                        sv,
                        xp_v,
                        ccw,
                        [x, y, z],
                        [x, y, z - 1],
                        [x, y - 1, z - 1],
                        [x, y - 1, z],
                    );
                    Self::emit_quad(
                        &self.vertices,
                        &mut self.packed_vertices,
                        &mut self.packed_indices,
                        &mut vertex_map,
                        sv,
                        yp_v,
                        ccw,
                        [x, y, z],
                        [x - 1, y, z],
                        [x - 1, y, z - 1],
                        [x, y, z - 1],
                    );
                    Self::emit_quad(
                        &self.vertices,
                        &mut self.packed_vertices,
                        &mut self.packed_indices,
                        &mut vertex_map,
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

    fn emit_quad(
        vertices: &[Option<VoxelVertex>],
        packed_vertices: &mut Vec<VoxelVertex>,
        packed_indices: &mut Vec<u32>,
        vertex_map: &mut HashMap<[i32; 3], u32>,
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

        // Phase 1: resolve indices — closure is dropped at block end, releasing the &mut borrow
        let indices = {
            let mut get_or_add = |coords: [i32; 3]| -> Option<u32> {
                if let Some(&idx) = vertex_map.get(&coords) {
                    return Some(idx);
                }
                let [x, y, z] = coords;
                let vertex = vertices[vertex_idx(x, y, z)].as_ref()?;
                let idx = packed_vertices.len() as u32;
                packed_vertices.push(vertex.clone());
                vertex_map.insert(coords, idx);
                Some(idx)
            };
            (
                get_or_add(v0),
                get_or_add(v1),
                get_or_add(v2),
                get_or_add(v3),
            )
        };

        // Phase 2: choose shorter diagonal, then emit two triangles with correct winding
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
            } else {
                if ccw {
                    packed_indices.extend([i0, i3, i1, i3, i2, i1]);
                } else {
                    packed_indices.extend([i0, i1, i3, i1, i2, i3]);
                }
            }
        }
    }

    fn check_intersection(&self, v: &[f32; 8]) -> bool {
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
