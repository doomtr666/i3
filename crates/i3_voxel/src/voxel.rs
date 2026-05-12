use crate::sdf::{SdfNode, SdfScene};
use i3_math::AABB;

use nalgebra::{Matrix3, Point3, Vector3, point};
use std::collections::HashMap;
use std::sync::Arc;

const VOXEL_BLOCK_WIDTH: i32 = 31;
const VOXEL_BLOCK_PADDED_WIDTH: i32 = VOXEL_BLOCK_WIDTH + 1;
const VOXEL_BLOCK_PADDED_WIDTH2: i32 = VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH;
const VOXEL_BLOCK_PADDED_SIZE: i32 =
    VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH;

const VOXEL_SCENE_WIDTH: i32 = 8;

pub const VOXEL_DIST: f32 = 0.05;
const TOLERANCE_SVD: f32 = 1e-4;

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

    pub fn get_vertices(&self) -> &Vec<Option<VoxelVertex>> {
        &self.vertices
    }

    pub fn get_packed_vertices(&self) -> &Vec<VoxelVertex> {
        &self.packed_vertices
    }

    pub fn get_packed_indices(&self) -> &Vec<u32> {
        &self.packed_indices
    }

    pub fn compute_mesh(&mut self) {
        let sdf = self.sdf.clone();
        let sdf_nodes = sdf.get_nodes(&self.world_aabb());
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

                        if (fa >= 0.0) != (fb >= 0.0) {
                            let q = {
                                let (mut lo, mut hi) = (0.0_f32, 1.0_f32);
                                let mut flo = fa;
                                for _ in 0..8 {
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

                    // 2. Créer l'AABB de la cellule avec une petite tolérance pour les angles vifs
                    //let margin = VOXEL_DIST * 0.25;
                    let voxel_aabb = AABB::new(
                        Point3::from(voxel_min.coords),
                        Point3::from(voxel_max.coords),
                    );
                    //.expand(margin);

                    if !voxel_aabb.contains(&qef_result) {
                        projected = Point3::from(center_of_mass);
                    }

                    for _ in 0..2 {
                        let f = SdfScene::value(sdf_nodes, &projected);
                        let n = SdfScene::normal(sdf_nodes, &projected);
                        projected = Point3::from(projected.coords - f * n * 0.5);
                    }

                    let position = voxel_aabb.clamp(&projected);

                    let final_normal = SdfScene::normal(sdf_nodes, &position);

                    self.vertices[((x + 1)
                        + (y + 1) * VOXEL_BLOCK_PADDED_WIDTH
                        + (z + 1) * VOXEL_BLOCK_PADDED_WIDTH2)
                        as usize] = Some(VoxelVertex {
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
                    let start_pos = self.world_pos(x, y, z);
                    let start_v = SdfScene::value(sdf_nodes, &start_pos);
                    let ccw = start_v >= 0.0;

                    // x+ edge
                    let xp_pos = self.world_pos(x + 1, y, z);
                    let xp_v = SdfScene::value(sdf_nodes, &xp_pos);

                    if (start_v >= 0.0) != (xp_v >= 0.0) {
                        if let (Some(i0), Some(i1), Some(i2), Some(i3)) = (
                            self.create_or_add_vertex(&mut vertex_map, x, y, z),
                            self.create_or_add_vertex(&mut vertex_map, x, y, z - 1),
                            self.create_or_add_vertex(&mut vertex_map, x, y - 1, z - 1),
                            self.create_or_add_vertex(&mut vertex_map, x, y - 1, z),
                        ) {
                            if ccw {
                                self.packed_indices.extend([i0, i3, i2, i0, i2, i1]);
                            } else {
                                self.packed_indices.extend([i0, i1, i2, i0, i2, i3]);
                            }
                        }
                    }

                    // y+ edge
                    let yp_pos = self.world_pos(x, y + 1, z);
                    let yp_v = SdfScene::value(sdf_nodes, &yp_pos);

                    if (start_v >= 0.0) != (yp_v >= 0.0) {
                        if let (Some(i0), Some(i1), Some(i2), Some(i3)) = (
                            self.create_or_add_vertex(&mut vertex_map, x, y, z),
                            self.create_or_add_vertex(&mut vertex_map, x - 1, y, z),
                            self.create_or_add_vertex(&mut vertex_map, x - 1, y, z - 1),
                            self.create_or_add_vertex(&mut vertex_map, x, y, z - 1),
                        ) {
                            if ccw {
                                self.packed_indices.extend([i0, i3, i2, i0, i2, i1]);
                            } else {
                                self.packed_indices.extend([i0, i1, i2, i0, i2, i3]);
                            }
                        }
                    }

                    // z+ edge
                    let zp_pos = self.world_pos(x, y, z + 1);
                    let zp_v = SdfScene::value(sdf_nodes, &zp_pos);

                    if (start_v >= 0.0) != (zp_v >= 0.0) {
                        if let (Some(i0), Some(i1), Some(i2), Some(i3)) = (
                            self.create_or_add_vertex(&mut vertex_map, x, y, z),
                            self.create_or_add_vertex(&mut vertex_map, x, y - 1, z),
                            self.create_or_add_vertex(&mut vertex_map, x - 1, y - 1, z),
                            self.create_or_add_vertex(&mut vertex_map, x - 1, y, z),
                        ) {
                            if ccw {
                                self.packed_indices.extend([i0, i3, i2, i0, i2, i1]);
                            } else {
                                self.packed_indices.extend([i0, i1, i2, i0, i2, i3]);
                            }
                        }
                    }
                }
            }
        }
    }

    fn create_or_add_vertex(
        &mut self,
        vertex_map: &mut HashMap<[i32; 3], u32>,
        x: i32,
        y: i32,
        z: i32,
    ) -> Option<u32> {
        if let Some(&index) = vertex_map.get(&[x, y, z]) {
            return Some(index);
        }
        let vertex = self.vertices[((x + 1)
            + (y + 1) * VOXEL_BLOCK_PADDED_WIDTH
            + (z + 1) * VOXEL_BLOCK_PADDED_WIDTH2) as usize]
            .as_ref()?;
        let index = self.packed_vertices.len() as u32;
        self.packed_vertices.push(vertex.clone());
        vertex_map.insert([x, y, z], index);
        Some(index)
    }

    fn check_intersection(&self, v: &[f32; 8]) -> bool {
        let sign = v[0] > 0.0;
        (1..8).any(|i| (v[i] > 0.0) != sign)
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
    #[allow(dead_code)]
    sdf: Arc<SdfScene>,
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

        VoxelScene { sdf, blocks }
    }

    pub fn compute_meshes(&mut self) {
        for block in &mut self.blocks {
            block.compute_mesh();
        }
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
