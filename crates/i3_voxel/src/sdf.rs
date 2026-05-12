#![allow(dead_code)]

use i3_math::{AABB, Transform};
use nalgebra::{Point3, Vector2, Vector3};

const SDF_EPSILON: f32 = 1e-4;
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
                // On projette le point sur le segment central de l'axe Y
                let clamped_y = position.y.clamp(-*half_height, *half_height);
                let closest_point = Point3::new(0.0, clamped_y, 0.0);

                (position - closest_point).norm() - radius
            }

            SdfPrimitive::Cylinder {
                half_height,
                radius,
            } => {
                // Distance sur le plan XZ (le disque)
                let d_xz = Vector3::new(position.x, 0.0, position.z).norm() - radius;
                // Distance sur l'axe Y (la hauteur)
                let d_y = position.y.abs() - half_height;

                // Combinaison euclidienne exacte pour les coins extérieurs et intérieurs
                let outside_dist = Vector2::new(d_xz.max(0.0), d_y.max(0.0)).norm();
                let inside_dist = d_xz.max(d_y).min(0.0);

                outside_dist + inside_dist
            }

            SdfPrimitive::Torus {
                major_radius,
                minor_radius,
            } => {
                // 1. Distance au grand cercle sur le plan XZ
                let q_x = Vector3::new(position.x, 0.0, position.z).norm() - major_radius;
                // 2. Distance euclidienne finale en combinant avec l'axe Y
                let q = Vector2::new(q_x, position.y);

                q.norm() - minor_radius
            }
        }
    }
}

pub struct SdfNode {
    pub(crate) primitive: SdfPrimitive,
    pub(crate) world_aabb: AABB,
    pub(crate) transform: Transform,
    pub(crate) substract: bool,
}

impl SdfNode {
    pub fn new(transform: &Transform, primitive: &SdfPrimitive, substract: bool) -> SdfNode {
        let jittered_transform =
            transform.with_translation_offset(Vector3::new(SDF_EPSILON, SDF_EPSILON, SDF_EPSILON));

        SdfNode {
            primitive: primitive.clone(),
            world_aabb: AABB::transform(&primitive.local_aabb(), &jittered_transform),
            transform: jittered_transform,
            substract,
        }
    }
}

pub struct SdfScene {
    nodes: Vec<SdfNode>,
}

impl SdfScene {
    pub fn new() -> SdfScene {
        SdfScene { nodes: Vec::new() }
    }

    pub fn add(&mut self, transform: &Transform, primitive: &SdfPrimitive) {
        self.nodes.push(SdfNode::new(transform, primitive, false));
    }

    pub fn sub(&mut self, transform: &Transform, primitive: &SdfPrimitive) {
        self.nodes.push(SdfNode::new(transform, primitive, true));
    }

    pub fn get_nodes(&self, aabb: &AABB) -> Vec<&SdfNode> {
        self.nodes
            .iter()
            .filter(|n| n.world_aabb.intersects(&aabb))
            .collect()
    }

    // Polynomial Smooth Minimum (CORRIGÉ)
    #[inline]
    pub fn smin(a: f32, b: f32, k: f32) -> f32 {
        let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
        // C'est b qui prend (1.0 - h) pour garantir le minimum
        b * (1.0 - h) + a * h - k * h * (1.0 - h)
    }

    // Polynomial Smooth Maximum
    #[inline]
    pub fn smax(a: f32, b: f32, k: f32) -> f32 {
        let h = (0.5 + 0.5 * (a - b) / k).clamp(0.0, 1.0);
        a * h + b * (1.0 - h) + k * h * (1.0 - h)
    }

    /*
        pub fn value(nodes: &[&SdfNode], position: &Point3<f32>) -> f32 {
            let mut solid_dist = f32::MAX;
            let mut empty_dist = f32::MAX;
            let k = SDF_SMOOTH_RADIUS; // Smooth radius

            for node in nodes {
                let local_p = node.transform.inv_transform_point(position);
                let dist = node.primitive.local_value(&local_p) * node.transform.scale();

                if node.substract {
                    if empty_dist == f32::MAX {
                        empty_dist = dist;
                    } else {
                        empty_dist = Self::smin(empty_dist, dist, k);
                    }
                } else {
                    if solid_dist == f32::MAX {
                        solid_dist = dist;
                    } else {
                        solid_dist = Self::smin(solid_dist, dist, k);
                    }
                }
            }

            if empty_dist != f32::MAX {
                Self::smax(solid_dist, -empty_dist, k)
            } else {
                solid_dist
            }
        }
    */

    pub fn value(nodes: &[&SdfNode], position: &Point3<f32>) -> f32 {
        let mut solid_dist = f32::MAX;
        let mut empty_dist = f32::MAX;

        for node in nodes {
            let local_p = node.transform.inv_transform_point(position);
            let dist = node.primitive.local_value(&local_p) * node.transform.scale();

            if node.substract {
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
