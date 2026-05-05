extern crate nalgebra as na;
use na::{Isometry3, Point3, Vector3};

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
}

pub struct Transform {
    isometry: Isometry3<f32>,
    scale: f32,
}

pub enum SdfPrimitive {
    Sphere { radius: f32 },
    Box { half_extents: Vector3<f32> },
}

pub struct SdfNode {
    primitive: SdfPrimitive,
    aaab: AABB,
}

pub struct SdfScene {
    primitives: Vec<SdfNode>,
}

impl SdfScene {
    pub fn new() -> SdfScene {
        SdfScene {
            primitives: Vec::<SdfNode>::new(),
        }
    }

    pub fn add(&mut self, transform: &Transform, primitive: &SdfPrimitive) {}
    pub fn sub(&mut self, transform: &Transform, primitive: &SdfPrimitive) {}
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
    half_extents: Vector3<f32>, // Dimensions depuis le centre (ex: [1.0, 1.0, 1.0] pour une boîte de 2x2x2)
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
