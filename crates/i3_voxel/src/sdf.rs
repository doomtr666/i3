#![allow(dead_code)]

use i3_math::{AABB, Transform};
use nalgebra::{Point3, Vector2, Vector3};

use crate::voxel::VOXEL_DIST;

const SDF_EPSILON: f32 = 1e-4;

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
    pub(crate) fn local_normal(&self, position: &Point3<f32>) -> Vector3<f32> {
        match self {
            SdfPrimitive::Sphere { .. } => position.coords.normalize(),
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
            SdfPrimitive::Capsule {
                half_height,
                radius: _,
            } => {
                // Le vecteur normalisé depuis le point le plus proche sur le segment Y
                let clamped_y = position.y.clamp(-*half_height, *half_height);
                let closest_point = Point3::new(0.0, clamped_y, 0.0);

                // Gérer le cas extrême où p est exactement sur le centre
                let diff = position - closest_point;
                if diff.norm_squared() < 1e-6 {
                    Vector3::y()
                } else {
                    diff.normalize()
                }
            }

            SdfPrimitive::Cylinder {
                half_height,
                radius,
            } => {
                let p_xz = Vector3::new(position.x, 0.0, position.z);
                let length_xz = p_xz.norm();

                let d_x = length_xz - radius;
                let d_y = position.y.abs() - half_height;

                let sign_y = position.y.signum();
                let dir_xz = if length_xz > 1e-6 {
                    p_xz / length_xz
                } else {
                    Vector3::x()
                };

                if d_x > 0.0 && d_y > 0.0 {
                    // À l'extérieur, dans la diagonale du coin (arête vive)
                    let corner_vec = Vector3::new(dir_xz.x * d_x, d_y * sign_y, dir_xz.z * d_x);
                    corner_vec.normalize()
                } else if d_y > d_x {
                    // Plus proche des "bouchons" haut/bas (que ce soit à l'intérieur ou à l'extérieur)
                    Vector3::new(0.0, sign_y, 0.0)
                } else {
                    // Plus proche du mur cylindrique (que ce soit à l'intérieur ou à l'extérieur)
                    dir_xz
                }
            }

            SdfPrimitive::Torus {
                major_radius,
                minor_radius: _,
            } => {
                let p_xz = Vector3::new(position.x, 0.0, position.z);
                let length_xz = p_xz.norm();

                // Vecteur directeur depuis l'origine sur le plan XZ
                let dir_xz = if length_xz > 1e-6 {
                    p_xz / length_xz
                } else {
                    Vector3::x()
                };

                // Le point le plus proche sur "l'âme" (le grand cercle) du tore
                let ring_point = dir_xz * *major_radius;

                let diff = position - ring_point;
                if diff.coords.norm_squared() < 1e-6 {
                    Vector3::y()
                } else {
                    diff.coords.normalize()
                }
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

        if -empty_dist > solid_dist {
            -empty_normal
        } else {
            solid_normal
        }
    }

    pub fn normal_fd(nodes: &[&SdfNode], position: &Point3<f32>) -> Vector3<f32> {
        // Analytical normals are mathematically invalid inside a smoothed zone.
        // Finite differences evaluate the true gradient of the blended distance field.
        const EPS: f32 = VOXEL_DIST * 0.25;

        let v_x1 = Self::value(
            nodes,
            &Point3::new(position.x + EPS, position.y, position.z),
        );
        let v_x2 = Self::value(
            nodes,
            &Point3::new(position.x - EPS, position.y, position.z),
        );

        let v_y1 = Self::value(
            nodes,
            &Point3::new(position.x, position.y + EPS, position.z),
        );
        let v_y2 = Self::value(
            nodes,
            &Point3::new(position.x, position.y - EPS, position.z),
        );

        let v_z1 = Self::value(
            nodes,
            &Point3::new(position.x, position.y, position.z + EPS),
        );
        let v_z2 = Self::value(
            nodes,
            &Point3::new(position.x, position.y, position.z - EPS),
        );

        let n = Vector3::new(v_x1 - v_x2, v_y1 - v_y2, v_z1 - v_z2);
        n.try_normalize(1e-6).unwrap_or_else(|| Vector3::y())
    }
}

impl Default for SdfScene {
    fn default() -> Self {
        Self::new()
    }
}
