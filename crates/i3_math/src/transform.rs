use nalgebra::{Isometry3, Point3, UnitQuaternion, Vector3};

#[derive(Clone, Copy)]
pub struct Transform {
    isometry: Isometry3<f32>,
    inv_isometry: Isometry3<f32>,
    scale: f32,
    inv_scale: f32,
}

impl Transform {
    pub fn new(translation: Vector3<f32>, rotation: UnitQuaternion<f32>, scale: f32) -> Self {
        let isometry = Isometry3::from_parts(translation.into(), rotation);

        Self {
            isometry,
            inv_isometry: isometry.inverse(),
            scale,
            inv_scale: 1.0 / scale,
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

    pub fn with_translation_offset(&self, offset: Vector3<f32>) -> Self {
        let new_translation = self.isometry.translation.vector + offset;
        let new_isometry = Isometry3::from_parts(new_translation.into(), self.isometry.rotation);

        Self {
            isometry: new_isometry,
            inv_isometry: new_isometry.inverse(),
            scale: self.scale,
            inv_scale: self.inv_scale,
        }
    }

    #[inline]
    pub fn inv_transform_point(&self, p: &Point3<f32>) -> Point3<f32> {
        let local_unscaled = self.inv_isometry * p;
        Point3::from(local_unscaled.coords * self.inv_scale)
    }

    #[inline]
    pub fn transform_normal(&self, n: &Vector3<f32>) -> Vector3<f32> {
        self.isometry.rotation * n
    }

    #[inline]
    pub fn transform_point(&self, p: &Point3<f32>) -> Point3<f32> {
        let scaled_p = Point3::from(p.coords * self.scale);
        self.isometry * scaled_p
    }

    #[inline]
    pub fn inv_transform_normal(&self, n: &Vector3<f32>) -> Vector3<f32> {
        self.inv_isometry.rotation * n
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }
}
