use crate::Transform;
use nalgebra::Point3;

pub struct AABB {
    pub(crate) min: Point3<f32>,
    pub(crate) max: Point3<f32>,
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

    pub fn contains(&self, p: &Point3<f32>) -> bool {
        if p.x > self.max.x {
            return false;
        }
        if p.y > self.max.y {
            return false;
        }
        if p.z > self.max.z {
            return false;
        }
        if p.x < self.min.x {
            return false;
        }
        if p.y < self.min.y {
            return false;
        }
        if p.z < self.min.z {
            return false;
        }
        true
    }

    pub fn clamp(&self, p: &Point3<f32>) -> Point3<f32> {
        Point3::<f32>::new(
            p.x.clamp(self.min.x, self.max.x),
            p.y.clamp(self.min.y, self.max.y),
            p.z.clamp(self.min.z, self.max.z),
        )
    }

    pub fn expand(&self, r: f32) -> AABB {
        AABB {
            min: Point3::from([self.min.x - r, self.min.y - r, self.min.z - r]),
            max: Point3::from([self.max.x + r, self.max.y + r, self.max.z + r]),
        }
    }

    pub fn transform(&self, transform: &Transform) -> AABB {
        if self.min.x > self.max.x {
            return AABB::invalid();
        }

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
            let t = transform.transform_point(corner);
            new_min.x = new_min.x.min(t.x);
            new_min.y = new_min.y.min(t.y);
            new_min.z = new_min.z.min(t.z);
            new_max.x = new_max.x.max(t.x);
            new_max.y = new_max.y.max(t.y);
            new_max.z = new_max.z.max(t.z);
        }

        AABB {
            min: new_min,
            max: new_max,
        }
    }
}
