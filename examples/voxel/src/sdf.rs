extern crate nalgebra as na;
use na::{Isometry3, Matrix4, Point3, Vector3};

pub struct Transform {
    isometry: Isometry3<f32>,
    scale: f32,
}

impl From<&Transform> for Matrix4<f32> {
    fn from(value: &Transform) -> Self {
        // to_homogeneous() converts the Isometry (T * R) into a 4x4 matrix.
        // Appending the scale results in the standard T * R * S matrix.
        value
            .isometry
            .to_homogeneous()
            .append_nonuniform_scaling(&Vector3::new(value.scale, value.scale, value.scale))
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
        unimplemented!()
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
    inv_transform: Matrix4<f32>,
    substract: bool,
}

impl SdfNode {
    pub fn new(transform: &Transform, primitive: &SdfPrimitive, substract: bool) -> SdfNode {
        SdfNode {
            primitive: primitive.clone(),
            world_aabb: AABB::transform(&primitive.local_aabb(), transform),
            inv_transform: Matrix4::<f32>::from(transform).try_inverse().unwrap(),
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

    pub fn value(nodes: &Vec<&SdfNode>, position: &Point3<f32>) -> f32 {
        unimplemented!()
    }

    pub fn normal(nodes: &Vec<&SdfNode>, position: &Point3<f32>) -> Vector3<f32> {
        unimplemented!()
    }
}

pub trait Sdf {
    fn value(&self, p: &Point3<f32>) -> f32;
    fn normal(&self, p: &Point3<f32>) -> Vector3<f32>;
}

pub struct SphereSdf {
    center: Point3<f32>,
    radius: f32,
}

impl SphereSdf {
    pub fn new(center: Point3<f32>, radius: f32) -> Self {
        Self { center, radius }
    }
}

impl Sdf for SphereSdf {
    fn value(&self, p: &Point3<f32>) -> f32 {
        (p - self.center).norm() - self.radius
    }

    fn normal(&self, p: &Point3<f32>) -> Vector3<f32> {
        (p - self.center).normalize()
    }
}

pub struct BoxSdf {
    center: Point3<f32>,
    half_extents: Vector3<f32>,
}

impl BoxSdf {
    pub fn new(center: Point3<f32>, half_extents: Vector3<f32>) -> Self {
        Self {
            center: center - Vector3::from([1e-5, 1e-5, 1e-5]),
            half_extents,
        }
    }
}

impl Sdf for BoxSdf {
    fn value(&self, p: &Point3<f32>) -> f32 {
        let local_p = p - self.center;

        let q = Vector3::new(
            local_p.x.abs() - self.half_extents.x,
            local_p.y.abs() - self.half_extents.y,
            local_p.z.abs() - self.half_extents.z,
        );

        let outside_dist = Vector3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).norm();
        let inside_dist = q.x.max(q.y).max(q.z).min(0.0);

        outside_dist + inside_dist
    }

    fn normal(&self, p: &Point3<f32>) -> Vector3<f32> {
        let local_p = p - self.center;
        let d = Vector3::new(
            local_p.x.abs() - self.half_extents.x,
            local_p.y.abs() - self.half_extents.y,
            local_p.z.abs() - self.half_extents.z,
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
