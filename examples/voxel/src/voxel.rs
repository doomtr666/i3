extern crate nalgebra as na;
use crate::Sdf;
use na::{Matrix3, Point3, Vector3, point};
use std::collections::HashMap;
use std::sync::Arc;

const VOXEL_BLOCK_WIDTH: i32 = 31;
const VOXEL_BLOCK_PADDED_WIDTH: i32 = VOXEL_BLOCK_WIDTH + 1;
const VOXEL_BLOCK_PADDED_WIDTH2: i32 = VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH;
const VOXEL_BLOCK_PADDED_SIZE: i32 =
    VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH * VOXEL_BLOCK_PADDED_WIDTH;

const VOXEL_SCENE_WIDTH: i32 = 4;

const VOXEL_DIST: f32 = 0.1;
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
    sdf: Arc<dyn Sdf>,
    px: i32,
    py: i32,
    pz: i32,
    vertices: Vec<Option<VoxelVertex>>,
    packed_vertices: Vec<VoxelVertex>,
    packed_indices: Vec<u32>,
}

impl VoxelBlock {
    pub fn new(sdf: Arc<dyn Sdf>, px: i32, py: i32, pz: i32) -> Self {
        Self {
            sdf,
            px,
            py,
            pz,
            vertices: vec![None; VOXEL_BLOCK_PADDED_SIZE as usize],
            packed_vertices: vec![VoxelVertex::default(); 0],
            packed_indices: vec![0; 0],
        }
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
        self.compute_vertices();
        self.compute_indices();
    }

    fn compute_vertices(&mut self) {
        for z in -1..=(VOXEL_BLOCK_WIDTH as i32 - 1) {
            for y in -1..=(VOXEL_BLOCK_WIDTH as i32 - 1) {
                for x in -1..=(VOXEL_BLOCK_WIDTH as i32 - 1) {
                    let mut vertices = [Point3::origin(); 8];
                    let mut values = [0.0f32; 8];
                    for i in 0..8 {
                        let p = self.world_pos(x + (i & 1), y + ((i >> 1) & 1), z + ((i >> 2) & 1));
                        vertices[i as usize] = p;
                        values[i as usize] = self.sdf.value(&p);
                    }

                    if !self.check_intersection(&values) {
                        continue;
                    }

                    // solve intersection point
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
                            // Bisection refinement — linear lerp is inaccurate for
                            // curved SDFs; 8 steps give sub-millimetre precision here.
                            let q = {
                                let (mut lo, mut hi) = (0.0_f32, 1.0_f32);
                                let mut flo = fa;
                                for _ in 0..8 {
                                    let mid = (lo + hi) * 0.5;
                                    let p = v1 + mid * (v2 - v1);
                                    let fmid = self.sdf.value(&p);
                                    if flo * fmid <= 0.0 {
                                        hi = mid;
                                    } else {
                                        lo = mid;
                                        flo = fmid;
                                    }
                                }
                                v1 + ((lo + hi) * 0.5) * (v2 - v1)
                            };
                            let n = self.sdf.normal(&q);
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

                    // add a small random value to stabilize the matrix
                    let epsilon_matrice = 1e-6;
                    mass_mat[(0, 0)] += epsilon_matrice;
                    mass_mat[(1, 1)] += epsilon_matrice;
                    mass_mat[(2, 2)] += epsilon_matrice;

                    let svd = mass_mat.svd(true, true);
                    let qef_result = match svd.pseudo_inverse(TOLERANCE_SVD) {
                        Ok(pseudo_inv) => Point3::from(pseudo_inv * mass_vec),
                        Err(_) => Point3::from(center_of_mass),
                    };

                    let mut projected = qef_result;

                    // Clamp to cell bounds (DC topology requirement).
                    let local_min = self.world_pos(x, y, z);
                    let local_max = self.world_pos(x + 1, y + 1, z + 1);

                    if qef_result.x < local_min.x
                        || qef_result.x > local_max.x
                        || qef_result.y < local_min.y
                        || qef_result.y > local_max.y
                        || qef_result.z < local_min.z
                        || qef_result.z > local_max.z
                    {
                        // If the QEF pushes the point outside the voxel, applying a component-wise
                        // clamp crushes the geometry. Falling back to the center of mass of the
                        // intersections is much more reliable.
                        projected = Point3::from(center_of_mass);
                    }

                    // Newton-project onto the isosurface: for non-linear SDFs the QEF
                    // minimizer satisfies the linearised constraints but may sit slightly
                    // off the actual zero-set.  Each step: p -= f(p)·∇f(p)/|∇f|²
                    // For a unit-gradient SDF (sphere, etc.) this converges in one step.
                    for _ in 0..3 {
                        let f = self.sdf.value(&projected);
                        let n = self.sdf.normal(&projected);
                        projected = Point3::from(projected.coords - f * n);
                    }

                    // final clamp
                    let position = Point3::from([
                        projected.x.clamp(local_min.x, local_max.x),
                        projected.y.clamp(local_min.y, local_max.y),
                        projected.z.clamp(local_min.z, local_max.z),
                    ]);

                    let normal = self.sdf.normal(&position);

                    self.vertices[((x + 1)
                        + (y + 1) * VOXEL_BLOCK_PADDED_WIDTH
                        + (z + 1) * VOXEL_BLOCK_PADDED_WIDTH2)
                        as usize] = Some(VoxelVertex {
                        position: position,
                        normal: normal,
                    });
                }
            }
        }
    }

    fn compute_indices(&mut self) {
        let mut vertex_map = HashMap::<[i32; 3], u32>::new();

        for z in 0..VOXEL_BLOCK_WIDTH as i32 {
            for y in 0..VOXEL_BLOCK_WIDTH as i32 {
                for x in 0..VOXEL_BLOCK_WIDTH as i32 {
                    let start_pos = self.world_pos(x, y, z);
                    let start_v = self.sdf.value(&start_pos);

                    // x+ edge
                    let xp_pos = self.world_pos(x + 1, y, z);
                    let xp_v = self.sdf.value(&xp_pos);

                    if (start_v >= 0.0) != (xp_v >= 0.0) {
                        if let (Some(i0), Some(i1), Some(i2), Some(i3)) = (
                            self.create_or_add_vertex(&mut vertex_map, x, y, z),
                            self.create_or_add_vertex(&mut vertex_map, x, y, z - 1),
                            self.create_or_add_vertex(&mut vertex_map, x, y - 1, z - 1),
                            self.create_or_add_vertex(&mut vertex_map, x, y - 1, z),
                        ) {
                            self.packed_indices.extend([i0, i1, i2, i3]);
                        }
                    }

                    // y+ edge
                    let yp_pos = self.world_pos(x, y + 1, z);
                    let yp_v = self.sdf.value(&yp_pos);

                    if (start_v >= 0.0) != (yp_v >= 0.0) {
                        if let (Some(i0), Some(i1), Some(i2), Some(i3)) = (
                            self.create_or_add_vertex(&mut vertex_map, x, y, z),
                            self.create_or_add_vertex(&mut vertex_map, x - 1, y, z),
                            self.create_or_add_vertex(&mut vertex_map, x - 1, y, z - 1),
                            self.create_or_add_vertex(&mut vertex_map, x, y, z - 1),
                        ) {
                            self.packed_indices.extend([i0, i1, i2, i3]);
                        }
                    }

                    // z+ edge
                    let zp_pos = self.world_pos(x, y, z + 1);
                    let zp_v = self.sdf.value(&zp_pos);

                    if (start_v >= 0.0) != (zp_v >= 0.0) {
                        if let (Some(i0), Some(i1), Some(i2), Some(i3)) = (
                            self.create_or_add_vertex(&mut vertex_map, x, y, z),
                            self.create_or_add_vertex(&mut vertex_map, x, y - 1, z),
                            self.create_or_add_vertex(&mut vertex_map, x - 1, y - 1, z),
                            self.create_or_add_vertex(&mut vertex_map, x - 1, y, z),
                        ) {
                            self.packed_indices.extend([i0, i1, i2, i3]);
                        }
                    }
                }
            }
        }

        println!(
            "vertices: {}, indices: {}",
            self.packed_vertices.len(),
            self.packed_indices.len()
        );
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
            (self.px as i32 * VOXEL_BLOCK_WIDTH as i32 + x) as f32 * VOXEL_DIST,
            (self.py as i32 * VOXEL_BLOCK_WIDTH as i32 + y) as f32 * VOXEL_DIST,
            (self.pz as i32 * VOXEL_BLOCK_WIDTH as i32 + z) as f32 * VOXEL_DIST,
        ]
    }
}

pub struct VoxelScene {
    sdf: Arc<dyn Sdf>,
    blocks: Vec<VoxelBlock>,
}

impl VoxelScene {
    pub fn new(sdf: Arc<dyn Sdf>) -> VoxelScene {
        let mut blocks = Vec::<VoxelBlock>::new();

        for z in 0..VOXEL_SCENE_WIDTH {
            for y in 0..VOXEL_SCENE_WIDTH {
                for x in 0..VOXEL_SCENE_WIDTH {
                    let block = VoxelBlock::new(Arc::clone(&sdf), x, y, z);
                    blocks.push(block);
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
