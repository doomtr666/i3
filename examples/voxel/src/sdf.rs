extern crate nalgebra;
use nalgebra::{Isometry3, Point3, UnitQuaternion, Vector3};

#[derive(Clone, Copy)]
pub struct Transform {
    isometry: Isometry3<f32>,
    inv_isometry: Isometry3<f32>,
    scale: f32,
    inv_scale: f32, // Precalculated to avoid divisions in the hot loop
}

impl Transform {
    /// Creates a new Transform and precalculates all inverses
    pub fn new(translation: Vector3<f32>, rotation: UnitQuaternion<f32>, scale: f32) -> Self {
        let isometry = Isometry3::from_parts(translation.into(), rotation);

        Self {
            isometry,
            inv_isometry: isometry.inverse(), // Extremely fast (quaternion conjugation)
            scale,
            inv_scale: 1.0 / scale, // Precalculated multiplication factor
        }
    }

    pub fn identity() -> Self {
        Self {
            isometry: Isometry3::identity(),
            inv_isometry: Isometry3::identity(),
            scale: 1.0,
            inv_scale: 1.0,
        }
    }

    /// World to Local (Domain Warping for SDF evaluation)
    #[inline]
    pub fn inv_transform_point(&self, p: &Point3<f32>) -> Point3<f32> {
        // 1. Un-translate and Un-rotate
        let local_unscaled = self.inv_isometry * p;
        // 2. Un-scale
        Point3::from(local_unscaled.coords * self.inv_scale)
    }

    /// Local to World (Restoring normal orientation)
    #[inline]
    pub fn transform_normal(&self, n: &Vector3<f32>) -> Vector3<f32> {
        // Uniform scale doesn't affect normal direction, only rotation does.
        self.isometry.rotation * n
    }

    /// Local to World (Useful for instantiating mesh vertices)
    #[inline]
    pub fn transform_point(&self, p: &Point3<f32>) -> Point3<f32> {
        // 1. Scale
        let scaled_p = Point3::from(p.coords * self.scale);
        // 2. Rotate and Translate
        self.isometry * scaled_p
    }

    /// World to Local (Useful for projecting world normals back to primitives if needed)
    #[inline]
    pub fn inv_transform_normal(&self, n: &Vector3<f32>) -> Vector3<f32> {
        self.inv_isometry.rotation * n
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }
}

pub struct AABB {
    min: Point3<f32>,
    max: Point3<f32>,
}

impl AABB {
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> AABB {
        AABB { min, max }
    }

    pub fn infinite() -> AABB {
        AABB {
            min: Point3::from([f32::MIN, f32::MIN, f32::MIN]),
            max: Point3::from([f32::MAX, f32::MAX, f32::MAX]),
        }
    }

    pub fn invalid() -> AABB {
        AABB {
            min: Point3::from([f32::MAX, f32::MAX, f32::MAX]),
            max: Point3::from([f32::MIN, f32::MIN, f32::MIN]),
        }
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        if other.min.x > self.max.x {
            return false;
        }

        if other.min.y > self.max.y {
            return false;
        }

        if other.min.z > self.max.z {
            return false;
        }

        if other.max.x < self.min.x {
            return false;
        }

        if other.max.y < self.min.y {
            return false;
        }

        if other.max.z < self.min.z {
            return false;
        }

        true
    }

    pub fn transform(&self, transform: &Transform) -> AABB {
        // Handle edge cases immediately to avoid math errors (NaN/Inf)
        if self.min.x > self.max.x {
            return AABB::invalid();
        }

        // Extract the 8 corners of the original AABB
        let corners = [
            Point3::new(self.min.x, self.min.y, self.min.z),
            Point3::new(self.max.x, self.min.y, self.min.z),
            Point3::new(self.min.x, self.max.y, self.min.z),
            Point3::new(self.max.x, self.max.y, self.min.z),
            Point3::new(self.min.x, self.min.y, self.max.z),
            Point3::new(self.max.x, self.min.y, self.max.z),
            Point3::new(self.min.x, self.max.y, self.max.z),
            Point3::new(self.max.x, self.max.y, self.max.z),
        ];

        let mut new_min = Point3::from([f32::MAX, f32::MAX, f32::MAX]);
        let mut new_max = Point3::from([f32::MIN, f32::MIN, f32::MIN]);

        for corner in &corners {
            // 1. Apply uniform scale to the local coordinates
            let scaled_coords = corner.coords * transform.scale;
            let scaled_corner = Point3::from(scaled_coords);

            // 2. Apply rotation and translation (Isometry)
            let transformed_corner = transform.isometry * scaled_corner;

            // 3. Expand the new bounds
            new_min.x = new_min.x.min(transformed_corner.x);
            new_min.y = new_min.y.min(transformed_corner.y);
            new_min.z = new_min.z.min(transformed_corner.z);

            new_max.x = new_max.x.max(transformed_corner.x);
            new_max.y = new_max.y.max(transformed_corner.y);
            new_max.z = new_max.z.max(transformed_corner.z);
        }

        AABB {
            min: new_min,
            max: new_max,
        }
    }
}

#[derive(Clone)]
pub enum SdfPrimitive {
    Sphere { radius: f32 },
    Box { half_extents: Vector3<f32> },
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
        }
    }

    fn local_normal(&self, position: &Point3<f32>) -> Vector3<f32> {
        match self {
            SdfPrimitive::Sphere { radius } => position.coords.normalize(),
            &SdfPrimitive::Box { half_extents } => {
                let local_p = position.coords;
                let d = Vector3::new(
                    local_p.x.abs() - half_extents.x,
                    local_p.y.abs() - half_extents.y,
                    local_p.z.abs() - half_extents.z,
                );

                let sign = Vector3::new(local_p.x.signum(), local_p.y.signum(), local_p.z.signum());

                if d.x > 0.0 || d.y > 0.0 || d.z > 0.0 {
                    let mut n = Vector3::zeros();
                    if d.x > 0.0 {
                        n.x = d.x * sign.x;
                    }
                    if d.y > 0.0 {
                        n.y = d.y * sign.y;
                    }
                    if d.z > 0.0 {
                        n.z = d.z * sign.z;
                    }
                    n.normalize()
                } else {
                    let mut n = Vector3::zeros();
                    if d.x > d.y && d.x > d.z {
                        n.x = sign.x;
                    } else if d.y > d.z {
                        n.y = sign.y;
                    } else {
                        n.z = sign.z;
                    }
                    n
                }
            }
        }
    }
}

pub struct SdfNode {
    primitive: SdfPrimitive,
    world_aabb: AABB,
    transform: Transform,
    substract: bool,
}

impl SdfNode {
    pub fn new(transform: &Transform, primitive: &SdfPrimitive, substract: bool) -> SdfNode {
        SdfNode {
            primitive: primitive.clone(),
            world_aabb: AABB::transform(&primitive.local_aabb(), transform),
            transform: transform.clone(),
            substract,
        }
    }
}

pub struct SdfScene {
    nodes: Vec<SdfNode>,
}

impl SdfScene {
    pub fn new() -> SdfScene {
        SdfScene {
            nodes: Vec::<SdfNode>::new(),
        }
    }

    pub fn add(&mut self, transform: &Transform, primitive: &SdfPrimitive) {
        self.nodes.push(SdfNode::new(transform, primitive, false));
    }

    pub fn sub(&mut self, transform: &Transform, primitive: &SdfPrimitive) {
        self.nodes.push(SdfNode::new(transform, primitive, true));
    }

    pub fn get_nodes(&self, aabb: &AABB) -> Vec<&SdfNode> {
        let mut result = Vec::<&SdfNode>::new();

        for node in &self.nodes {
            if node.world_aabb.intersects(aabb) {
                result.push(&node);
            }
        }

        result
    }

    pub fn value(nodes: &[&SdfNode], position: &Point3<f32>) -> f32 {
        let mut solid_dist = f32::MAX;
        let mut empty_dist = f32::MAX;

        for node in nodes {
            // 1. Warp point to local space
            let local_p = node.transform.inv_transform_point(position);

            // 2. Evaluate and un-warp distance metric
            let dist = node.primitive.local_value(&local_p) * node.transform.scale;

            if node.substract {
                empty_dist = empty_dist.min(dist);
            } else {
                solid_dist = solid_dist.min(dist);
            }
        }

        solid_dist.max(-empty_dist)
    }

    pub fn normal(nodes: &[&SdfNode], position: &Point3<f32>) -> Vector3<f32> {
        let mut solid_dist = f32::MAX;
        let mut solid_normal = Vector3::zeros();

        let mut empty_dist = f32::MAX;
        let mut empty_normal = Vector3::zeros();

        for node in nodes {
            let local_p = node.transform.inv_transform_point(position);
            let dist = node.primitive.local_value(&local_p) * node.transform.scale();

            if node.substract {
                if dist < empty_dist {
                    empty_dist = dist;
                    // Rotate the local normal back to world space
                    empty_normal = node
                        .transform
                        .transform_normal(&node.primitive.local_normal(&local_p));
                }
            } else {
                if dist < solid_dist {
                    solid_dist = dist;
                    solid_normal = node
                        .transform
                        .transform_normal(&node.primitive.local_normal(&local_p));
                }
            }
        }

        // Inside the hole: invert the normal of the subtracting primitive
        if -empty_dist > solid_dist {
            -empty_normal
        } else {
            solid_normal
        }
    }
}
