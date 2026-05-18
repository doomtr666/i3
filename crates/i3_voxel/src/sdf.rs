use i3_math::{AABB, Transform};
use libnoise::Generator;
use nalgebra::{Point3, Vector2, Vector3};
use std::sync::Arc;

const SDF_NORMAL_EPS: f32 = 0.001;

#[derive(Clone)]
pub enum SdfPrimitive {
    Sphere {
        radius: f32,
    },
    Box {
        half_extents: Vector3<f32>,
    },
    Capsule {
        half_height: f32,
        radius: f32,
    },
    Cylinder {
        half_height: f32,
        radius: f32,
    },
    Torus {
        major_radius: f32,
        minor_radius: f32,
    },
    TerrainBox {
        half_extents: Vector3<f32>,
        amplitude: f32,
        sampler: Arc<dyn Fn(f32, f32) -> f32 + Send + Sync>,
    },
}

impl SdfPrimitive {
    fn local_aabb(&self) -> AABB {
        match self {
            SdfPrimitive::Sphere { radius } => AABB::new(
                Point3::new(-radius, -radius, -radius),
                Point3::new(*radius, *radius, *radius),
            ),
            SdfPrimitive::Box { half_extents } => AABB::new(
                Point3::new(-half_extents.x, -half_extents.y, -half_extents.z),
                Point3::new(half_extents.x, half_extents.y, half_extents.z),
            ),
            SdfPrimitive::Capsule {
                half_height,
                radius,
            } => AABB::new(
                Point3::new(-radius, -half_height - radius, -radius),
                Point3::new(*radius, *half_height + *radius, *radius),
            ),
            SdfPrimitive::Cylinder {
                half_height,
                radius,
            } => AABB::new(
                Point3::new(-radius, -half_height, -radius),
                Point3::new(*radius, *half_height, *radius),
            ),
            SdfPrimitive::Torus {
                major_radius,
                minor_radius,
            } => AABB::new(
                Point3::new(
                    -major_radius - minor_radius,
                    -minor_radius,
                    -major_radius - minor_radius,
                ),
                Point3::new(
                    major_radius + minor_radius,
                    *minor_radius,
                    major_radius + minor_radius,
                ),
            ),
            SdfPrimitive::TerrainBox { half_extents, .. } => AABB::new(
                Point3::new(-half_extents.x, -half_extents.y, -half_extents.z),
                Point3::new(half_extents.x, half_extents.y, half_extents.z),
            ),
        }
    }

    fn local_value(&self, position: &Point3<f32>) -> f32 {
        match self {
            SdfPrimitive::Sphere { radius } => position.coords.norm() - radius,
            SdfPrimitive::Box { half_extents } => {
                let local_p = position.coords;

                let q = Vector3::new(
                    local_p.x.abs() - half_extents.x,
                    local_p.y.abs() - half_extents.y,
                    local_p.z.abs() - half_extents.z,
                );

                let outside_dist = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).norm();
                let inside_dist = q.x.max(q.y).max(q.z).min(0.0);

                outside_dist + inside_dist
            }
            SdfPrimitive::Capsule {
                half_height,
                radius,
            } => {
                let clamped_y = position.y.clamp(-*half_height, *half_height);
                let closest_point = Point3::new(0.0, clamped_y, 0.0);

                (position - closest_point).norm() - radius
            }

            SdfPrimitive::Cylinder {
                half_height,
                radius,
            } => {
                let d_xz = Vector3::new(position.x, 0.0, position.z).norm() - radius;
                let d_y = position.y.abs() - half_height;

                let outside_dist = Vector2::new(d_xz.max(0.0), d_y.max(0.0)).norm();
                let inside_dist = d_xz.max(d_y).min(0.0);

                outside_dist + inside_dist
            }

            SdfPrimitive::Torus {
                major_radius,
                minor_radius,
            } => {
                let q_x = Vector3::new(position.x, 0.0, position.z).norm() - major_radius;
                let q = Vector2::new(q_x, position.y);

                q.norm() - minor_radius
            }

            SdfPrimitive::TerrainBox {
                half_extents,
                amplitude,
                sampler,
            } => {
                let q = Vector3::new(
                    position.x.abs() - half_extents.x,
                    position.y.abs() - half_extents.y,
                    position.z.abs() - half_extents.z,
                );
                let box_d = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).norm()
                    + q.x.max(q.y).max(q.z).min(0.0);

                let nx = (position.x / half_extents.x + 1.0) * 0.5;
                let nz = (position.z / half_extents.z + 1.0) * 0.5;

                // Divide by |∇f| = sqrt(1 + (∂h/∂x)² + (∂h/∂z)²) to keep the SDF
                // Lipschitz-1 on steep slopes, which is required for correct DC output.
                const G: f32 = 1e-3;
                let h   = sampler(nx,     nz    ) * amplitude;
                let h_x = sampler(nx + G, nz    ) * amplitude;
                let h_z = sampler(nx,     nz + G) * amplitude;
                let dh_dx = (h_x - h) / (G * half_extents.x * 2.0);
                let dh_dz = (h_z - h) / (G * half_extents.z * 2.0);
                let terrain_d = (position.y - h)
                    / (1.0_f32 + dh_dx * dh_dx + dh_dz * dh_dz).sqrt();

                terrain_d.max(box_d)
            }
        }
    }
}

impl SdfPrimitive {
    pub fn terrain_box(
        half_extents: Vector3<f32>,
        amplitude: f32,
        generator: impl Generator<2> + Send + Sync + 'static,
    ) -> Self {
        SdfPrimitive::TerrainBox {
            half_extents,
            amplitude,
            sampler: Arc::new(move |x: f32, z: f32| generator.sample([x as f64, z as f64]) as f32),
        }
    }
}

pub struct SdfNode {
    pub(crate) primitive:   SdfPrimitive,
    pub(crate) world_aabb:  AABB,
    pub(crate) transform:   Transform,
    pub(crate) subtract:    bool,
    pub(crate) material_id: u32,
}

impl SdfNode {
    pub fn new(transform: &Transform, primitive: &SdfPrimitive, subtract: bool, material_id: u32) -> SdfNode {
        SdfNode {
            primitive: primitive.clone(),
            world_aabb: AABB::transform(&primitive.local_aabb(), transform),
            transform: *transform,
            subtract,
            material_id,
        }
    }

    pub fn primitive(&self)   -> &SdfPrimitive { &self.primitive }
    pub fn transform(&self)   -> &Transform    { &self.transform }
    pub fn is_subtract(&self) -> bool          { self.subtract }
    pub fn material_id(&self) -> u32           { self.material_id }
}

// ─── BVH ─────────────────────────────────────────────────────────────────────

pub struct BvhNode {
    pub aabb:     AABB,
    pub left:     u32,     // index in bvh_nodes[]; u32::MAX = leaf
    pub right:    u32,
    pub prim_idx: u32,     // index in nodes[]; valid only when left == u32::MAX
}

// ─── SdfScene ─────────────────────────────────────────────────────────────────

pub struct SdfScene {
    nodes:     Vec<SdfNode>,
    bvh_nodes: Vec<BvhNode>,
}

impl SdfScene {
    pub fn new() -> SdfScene {
        SdfScene { nodes: Vec::new(), bvh_nodes: Vec::new() }
    }

    pub fn add(&mut self, transform: &Transform, primitive: &SdfPrimitive) {
        self.nodes.push(SdfNode::new(transform, primitive, false, 0));
        self.bvh_nodes.clear();
    }

    pub fn sub(&mut self, transform: &Transform, primitive: &SdfPrimitive) {
        self.nodes.push(SdfNode::new(transform, primitive, true, 0));
        self.bvh_nodes.clear();
    }

    pub fn add_mat(&mut self, transform: &Transform, primitive: &SdfPrimitive, material_id: u32) {
        self.nodes.push(SdfNode::new(transform, primitive, false, material_id));
        self.bvh_nodes.clear();
    }

    pub fn sub_mat(&mut self, transform: &Transform, primitive: &SdfPrimitive, material_id: u32) {
        self.nodes.push(SdfNode::new(transform, primitive, true, material_id));
        self.bvh_nodes.clear();
    }

    /// Build (or rebuild) the BVH. Call once after all primitives are added.
    pub fn build_bvh(&mut self) {
        self.bvh_nodes.clear();
        if self.nodes.is_empty() {
            return;
        }
        let mut indices: Vec<u32> = (0..self.nodes.len() as u32).collect();
        Self::bvh_build(&self.nodes, &mut indices, &mut self.bvh_nodes);
    }

    /// Query nodes whose world AABB intersects `query`. Uses BVH if built, else linear scan.
    pub fn get_nodes(&self, aabb: &AABB) -> Vec<&SdfNode> {
        if !self.bvh_nodes.is_empty() {
            self.bvh_query(aabb)
        } else {
            self.nodes
                .iter()
                .filter(|n| n.world_aabb.intersects(aabb))
                .collect()
        }
    }

    // ── BVH internals ────────────────────────────────────────────────────────

    fn bvh_build(nodes: &[SdfNode], indices: &mut [u32], out: &mut Vec<BvhNode>) -> u32 {
        let idx = out.len() as u32;

        // Compute union AABB of all primitives in this subset
        let mut union_aabb = nodes[indices[0] as usize].world_aabb;
        for &i in &indices[1..] {
            union_aabb = union_aabb.union(&nodes[i as usize].world_aabb);
        }

        if indices.len() == 1 {
            out.push(BvhNode { aabb: union_aabb, left: u32::MAX, right: 0, prim_idx: indices[0] });
            return idx;
        }

        // Split along the longest axis by centroid median
        let ext = union_aabb.extent();
        let axis = if ext.x >= ext.y && ext.x >= ext.z { 0usize }
                   else if ext.y >= ext.z { 1 }
                   else { 2 };

        indices.sort_unstable_by(|&a, &b| {
            let ca = nodes[a as usize].world_aabb.center().coords[axis];
            let cb = nodes[b as usize].world_aabb.center().coords[axis];
            ca.partial_cmp(&cb).unwrap()
        });

        let mid = indices.len() / 2;
        let (left_idx, right_idx) = indices.split_at_mut(mid);

        // Reserve slot for this internal node before recursing
        out.push(BvhNode { aabb: union_aabb, left: 0, right: 0, prim_idx: u32::MAX });
        let left  = Self::bvh_build(nodes, left_idx, out);
        let right = Self::bvh_build(nodes, right_idx, out);
        out[idx as usize].left  = left;
        out[idx as usize].right = right;
        idx
    }

    fn bvh_query<'a>(&'a self, query: &AABB) -> Vec<&'a SdfNode> {
        let mut result = Vec::new();
        if self.bvh_nodes.is_empty() {
            return result;
        }
        let mut stack = [0u32; 64];
        let mut top = 0usize;
        stack[top] = 0;
        top += 1;

        while top > 0 {
            top -= 1;
            let node = &self.bvh_nodes[stack[top] as usize];
            if !node.aabb.intersects(query) {
                continue;
            }
            if node.left == u32::MAX {
                // Leaf
                result.push(&self.nodes[node.prim_idx as usize]);
            } else {
                stack[top] = node.left;
                top += 1;
                stack[top] = node.right;
                top += 1;
            }
        }
        result
    }

    pub fn nodes(&self) -> &[SdfNode] { &self.nodes }
    pub fn bvh_nodes(&self) -> &[BvhNode] { &self.bvh_nodes }

    pub fn nodes_mut(&mut self) -> &mut Vec<SdfNode> {
        self.bvh_nodes.clear();
        &mut self.nodes
    }

    pub fn value(nodes: &[&SdfNode], position: &Point3<f32>) -> f32 {
        let mut solid_dist = f32::MAX;
        let mut empty_dist = f32::MAX;

        for node in nodes {
            let local_p = node.transform.inv_transform_point(position);
            let dist = node.primitive.local_value(&local_p) * node.transform.scale();

            if node.subtract {
                empty_dist = empty_dist.min(dist);
            } else {
                solid_dist = solid_dist.min(dist);
            }
        }

        solid_dist.max(-empty_dist)
    }

    pub fn normal(nodes: &[&SdfNode], position: &Point3<f32>) -> Vector3<f32> {
        let k0 = Vector3::new(1.0, -1.0, -1.0);
        let k1 = Vector3::new(-1.0, -1.0, 1.0);
        let k2 = Vector3::new(-1.0, 1.0, -1.0);
        let k3 = Vector3::new(1.0, 1.0, 1.0);

        let p = position.coords;

        let f0 = Self::value(nodes, &Point3::from(p + k0 * SDF_NORMAL_EPS));
        let f1 = Self::value(nodes, &Point3::from(p + k1 * SDF_NORMAL_EPS));
        let f2 = Self::value(nodes, &Point3::from(p + k2 * SDF_NORMAL_EPS));
        let f3 = Self::value(nodes, &Point3::from(p + k3 * SDF_NORMAL_EPS));

        let n = k0 * f0 + k1 * f1 + k2 * f2 + k3 * f3;

        n.try_normalize(1e-6).unwrap_or_else(|| Vector3::y())
    }
}

impl Default for SdfScene {
    fn default() -> Self {
        Self::new()
    }
}
