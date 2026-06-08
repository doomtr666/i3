use i3_math::{AABB, Transform};
use i3_noise::{Generator, NoisePoint};
use nalgebra::{Point3, Vector2, Vector3};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct SdfSample {
    pub value: f32,
    pub gradient: Vector3<f32>,
}

impl Default for SdfSample {
    fn default() -> Self {
        Self {
            value: f32::MAX,
            gradient: Vector3::zeros(),
        }
    }
}

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
        generator: Arc<dyn Generator + Send + Sync>,
    },
    /// 3D volumetric terrain: `d(p) = p.y − amplitude·noise(p)` (the noise graph,
    /// frequency included, lives in `generator`). The CPU SDF estimate is
    /// `d / max(|∇d|, 1)` — tight where the field is well-conditioned, floored so it
    /// can't blow up at verticals and false-cull; the GPU bake uses `d/|∇d|` (same zero
    /// set). The noise depends on all of `p`, so the isosurface folds → overhangs/caves.
    VolumeTerrain {
        half_extents: Vector3<f32>,
        amplitude: f32,
        generator: Arc<dyn Generator + Send + Sync>,
    },
}

impl Clone for SdfPrimitive {
    fn clone(&self) -> Self {
        match self {
            Self::Sphere { radius } => Self::Sphere { radius: *radius },
            Self::Box { half_extents } => Self::Box {
                half_extents: *half_extents,
            },
            Self::Capsule {
                half_height,
                radius,
            } => Self::Capsule {
                half_height: *half_height,
                radius: *radius,
            },
            Self::Cylinder {
                half_height,
                radius,
            } => Self::Cylinder {
                half_height: *half_height,
                radius: *radius,
            },
            Self::Torus {
                major_radius,
                minor_radius,
            } => Self::Torus {
                major_radius: *major_radius,
                minor_radius: *minor_radius,
            },
            Self::TerrainBox {
                half_extents,
                amplitude,
                generator,
            } => Self::TerrainBox {
                half_extents: *half_extents,
                amplitude: *amplitude,
                generator: generator.clone(),
            },
            Self::VolumeTerrain {
                half_extents,
                amplitude,
                generator,
            } => Self::VolumeTerrain {
                half_extents: *half_extents,
                amplitude: *amplitude,
                generator: generator.clone(),
            },
        }
    }
}

impl SdfPrimitive {
    fn local_aabb(&self) -> AABB {
        match self {
            SdfPrimitive::Sphere { radius } => AABB::new(
                Point3::new(-radius, -radius, -radius),
                Point3::new(*radius, *radius, *radius),
            ),
            SdfPrimitive::Box { half_extents }
            | SdfPrimitive::TerrainBox { half_extents, .. }
            | SdfPrimitive::VolumeTerrain { half_extents, .. } => {
                AABB::new(
                    Point3::new(-half_extents.x, -half_extents.y, -half_extents.z),
                    Point3::new(half_extents.x, half_extents.y, half_extents.z),
                )
            }
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

    fn local_sample(&self, position: &Point3<f32>) -> SdfSample {
        match self {
            SdfPrimitive::Sphere { radius } => {
                let d = position.coords.norm();
                let gradient = if d > 1e-6 {
                    position.coords / d
                } else {
                    Vector3::y()
                };
                SdfSample {
                    value: d - radius,
                    gradient,
                }
            }
            SdfPrimitive::Box { half_extents } => {
                let p = position.coords;
                let d = Vector3::new(
                    p.x.abs() - half_extents.x,
                    p.y.abs() - half_extents.y,
                    p.z.abs() - half_extents.z,
                );
                let out_d = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0));
                let in_d = d.x.max(d.y).max(d.z).min(0.0);
                let value = out_d.norm() + in_d;

                let gradient = if out_d.norm_squared() > 1e-6 {
                    Vector3::new(
                        if d.x > 0.0 {
                            p.x.signum() * out_d.x
                        } else {
                            0.0
                        },
                        if d.y > 0.0 {
                            p.y.signum() * out_d.y
                        } else {
                            0.0
                        },
                        if d.z > 0.0 {
                            p.z.signum() * out_d.z
                        } else {
                            0.0
                        },
                    )
                    .normalize()
                } else {
                    let mut g = Vector3::zeros();
                    if d.x > d.y && d.x > d.z {
                        g.x = p.x.signum();
                    } else if d.y > d.z {
                        g.y = p.y.signum();
                    } else {
                        g.z = p.z.signum();
                    }
                    g
                };
                SdfSample { value, gradient }
            }
            SdfPrimitive::Capsule {
                half_height,
                radius,
            } => {
                let p = position.coords;
                let clamped_y = p.y.clamp(-*half_height, *half_height);
                let closest = Vector3::new(0.0, clamped_y, 0.0);
                let diff = p - closest;
                let d = diff.norm();
                let gradient = if d > 1e-6 { diff / d } else { Vector3::x() };
                SdfSample {
                    value: d - radius,
                    gradient,
                }
            }
            SdfPrimitive::Cylinder {
                half_height,
                radius,
            } => {
                let p = position.coords;
                let dxz = Vector2::new(p.x, p.z);
                let l = dxz.norm();
                let d = Vector2::new(l - radius, p.y.abs() - half_height);
                let out_d = Vector2::new(d.x.max(0.0), d.y.max(0.0));
                let in_d = d.x.max(d.y).min(0.0);
                let value = out_d.norm() + in_d;

                let gradient = if out_d.norm_squared() > 1e-6 {
                    let g2 = Vector2::new(
                        if d.x > 0.0 { out_d.x } else { 0.0 },
                        if d.y > 0.0 {
                            p.y.signum() * out_d.y
                        } else {
                            0.0
                        },
                    )
                    .normalize();
                    let dxz_norm = if l > 1e-6 { dxz / l } else { Vector2::x() };
                    Vector3::new(g2.x * dxz_norm.x, g2.y, g2.x * dxz_norm.y)
                } else {
                    if d.x > d.y {
                        let dxz_norm = if l > 1e-6 { dxz / l } else { Vector2::x() };
                        Vector3::new(dxz_norm.x, 0.0, dxz_norm.y)
                    } else {
                        Vector3::new(0.0, p.y.signum(), 0.0)
                    }
                };
                SdfSample { value, gradient }
            }
            SdfPrimitive::Torus {
                major_radius,
                minor_radius,
            } => {
                let p = position.coords;
                let dxz = Vector2::new(p.x, p.z);
                let l = dxz.norm();
                let q = Vector2::new(l - major_radius, p.y);
                let d = q.norm();
                let value = d - minor_radius;

                let gradient = if d > 1e-6 && l > 1e-6 {
                    let dq = q / d;
                    let dxz_norm = dxz / l;
                    Vector3::new(dq.x * dxz_norm.x, dq.y, dq.x * dxz_norm.y)
                } else {
                    Vector3::y()
                };
                SdfSample { value, gradient }
            }
            SdfPrimitive::TerrainBox {
                half_extents,
                amplitude,
                generator,
            } => {
                let p = position.coords;
                let mut vpos = NoisePoint::default();
                let nx = (p.x / half_extents.x + 1.0) * 0.5;
                let nz = (p.z / half_extents.z + 1.0) * 0.5;
                vpos.x[0] = nx;
                vpos.y[0] = 0.0;
                vpos.z[0] = nz;

                let res = generator.sample(&vpos);
                let h = res.value[0] * amplitude;

                // Box intersection logic
                let d = Vector3::new(
                    p.x.abs() - half_extents.x,
                    p.y.abs() - half_extents.y,
                    p.z.abs() - half_extents.z,
                );
                let out_d = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0));
                let in_d = d.x.max(d.y).max(d.z).min(0.0);
                let box_v = out_d.norm() + in_d;

                let box_g = if out_d.norm_squared() > 1e-6 {
                    Vector3::new(
                        if d.x > 0.0 {
                            p.x.signum() * out_d.x
                        } else {
                            0.0
                        },
                        if d.y > 0.0 {
                            p.y.signum() * out_d.y
                        } else {
                            0.0
                        },
                        if d.z > 0.0 {
                            p.z.signum() * out_d.z
                        } else {
                            0.0
                        },
                    )
                    .normalize()
                } else {
                    let mut g = Vector3::zeros();
                    if d.x > d.y && d.x > d.z {
                        g.x = p.x.signum();
                    } else if d.y > d.z {
                        g.y = p.y.signum();
                    } else {
                        g.z = p.z.signum();
                    }
                    g
                };

                let terrain_v = p.y - h;
                let dh_dx = amplitude * res.dx[0] * (0.5 / half_extents.x);
                let dh_dz = amplitude * res.dz[0] * (0.5 / half_extents.z);
                let mut terrain_g = Vector3::new(-dh_dx, 1.0, -dh_dz);
                terrain_g.try_normalize_mut(1e-6);

                if terrain_v > box_v {
                    SdfSample {
                        value: terrain_v,
                        gradient: terrain_g,
                    }
                } else {
                    SdfSample {
                        value: box_v,
                        gradient: box_g,
                    }
                }
            }
            SdfPrimitive::VolumeTerrain {
                amplitude,
                generator,
                ..
            } => {
                // d(p) = p.y − A·noise(p), with the frequency etc. inside `generator`
                // (the noise graph) so this matches `terrainDensity` in noise.slangh.
                // Footprint 0 (full detail) for now.
                let p = position.coords;
                let a = *amplitude;
                let mut np = NoisePoint::default();
                np.x[0] = p.x;
                np.y[0] = p.y;
                np.z[0] = p.z;
                let s = generator.sample(&np);

                let d = p.y - a * s.value[0];
                // ∂d/∂p = (0,1,0) − A·∇noise   (the graph's gradient is already w.r.t p)
                let g = Vector3::new(-a * s.dx[0], 1.0 - a * s.dy[0], -a * s.dz[0]);
                let gl = g.norm();
                // SDF estimate = d / max(|∇d|, 1). Tight (≈ true distance) where |∇d|≥1, so
                // near_surface bakes a THIN shell (no over-baking). The floor at 1 stops
                // d/|∇d| from blowing up at verticals (|∇d|→0) where it would otherwise
                // overestimate distance and false-cull crossing leaves → holes. The GPU
                // bake keeps d/|∇d| — same zero set, so the surface itself is identical.
                SdfSample {
                    value: d / gl.max(1.0),
                    gradient: g / gl.max(1e-6),
                }
            }
        }
    }

    fn local_sample_vec(&self, pos: &NoisePoint, out: &mut [SdfSample; 8]) {
        match self {
            SdfPrimitive::TerrainBox {
                half_extents,
                amplitude,
                generator,
            } => {
                let mut vpos = NoisePoint::default();
                for i in 0..8 {
                    let nx = (pos.x[i] / half_extents.x + 1.0) * 0.5;
                    let nz = (pos.z[i] / half_extents.z + 1.0) * 0.5;
                    vpos.x[i] = nx;
                    vpos.y[i] = 0.0;
                    vpos.z[i] = nz;
                }

                let res = generator.sample(&vpos);

                for i in 0..8 {
                    let px = pos.x[i];
                    let py = pos.y[i];
                    let pz = pos.z[i];

                    let h = res.value[i] * amplitude;

                    let d = Vector3::new(
                        px.abs() - half_extents.x,
                        py.abs() - half_extents.y,
                        pz.abs() - half_extents.z,
                    );
                    let out_d = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0));
                    let in_d = d.x.max(d.y).max(d.z).min(0.0);
                    let box_v = out_d.norm() + in_d;

                    let box_g = if out_d.norm_squared() > 1e-6 {
                        Vector3::new(
                            if d.x > 0.0 {
                                px.signum() * out_d.x
                            } else {
                                0.0
                            },
                            if d.y > 0.0 {
                                py.signum() * out_d.y
                            } else {
                                0.0
                            },
                            if d.z > 0.0 {
                                pz.signum() * out_d.z
                            } else {
                                0.0
                            },
                        )
                        .normalize()
                    } else {
                        let mut g = Vector3::zeros();
                        if d.x > d.y && d.x > d.z {
                            g.x = px.signum();
                        } else if d.y > d.z {
                            g.y = py.signum();
                        } else {
                            g.z = pz.signum();
                        }
                        g
                    };

                    let terrain_v = py - h;
                    let dh_dx = amplitude * res.dx[i] * (0.5 / half_extents.x);
                    let dh_dz = amplitude * res.dz[i] * (0.5 / half_extents.z);
                    let mut terrain_g = Vector3::new(-dh_dx, 1.0, -dh_dz);
                    terrain_g.try_normalize_mut(1e-6);

                    if terrain_v > box_v {
                        out[i] = SdfSample {
                            value: terrain_v,
                            gradient: terrain_g,
                        };
                    } else {
                        out[i] = SdfSample {
                            value: box_v,
                            gradient: box_g,
                        };
                    }
                }
            }
            _ => {
                for i in 0..8 {
                    out[i] = self.local_sample(&Point3::new(pos.x[i], pos.y[i], pos.z[i]));
                }
            }
        }
    }
}

impl SdfPrimitive {
    pub fn terrain_box(
        half_extents: Vector3<f32>,
        amplitude: f32,
        generator: Arc<dyn Generator + Send + Sync>,
    ) -> Self {
        SdfPrimitive::TerrainBox {
            half_extents,
            amplitude,
            generator,
        }
    }

    pub fn volume_terrain(
        half_extents: Vector3<f32>,
        amplitude: f32,
        generator: Arc<dyn Generator + Send + Sync>,
    ) -> Self {
        SdfPrimitive::VolumeTerrain {
            half_extents,
            amplitude,
            generator,
        }
    }
}

pub struct SdfNode {
    pub(crate) primitive: SdfPrimitive,
    pub(crate) world_aabb: AABB,
    pub(crate) transform: Transform,
    pub(crate) subtract: bool,
    pub(crate) material_id: u32,
}

impl SdfNode {
    pub fn new(
        transform: &Transform,
        primitive: &SdfPrimitive,
        subtract: bool,
        material_id: u32,
    ) -> SdfNode {
        SdfNode {
            primitive: primitive.clone(),
            world_aabb: AABB::transform(&primitive.local_aabb(), transform),
            transform: *transform,
            subtract,
            material_id,
        }
    }

    pub fn primitive(&self) -> &SdfPrimitive {
        &self.primitive
    }
    pub fn transform(&self) -> &Transform {
        &self.transform
    }
    pub fn is_subtract(&self) -> bool {
        self.subtract
    }
    pub fn material_id(&self) -> u32 {
        self.material_id
    }
}

pub struct BvhNode {
    pub aabb: AABB,
    pub left: u32, // index in bvh_nodes[]; u32::MAX = leaf
    pub right: u32,
    pub prim_idx: u32, // index in nodes[]; valid only when left == u32::MAX
}

pub struct SdfScene {
    nodes: Vec<SdfNode>,
    bvh_nodes: Vec<BvhNode>,
}

impl SdfScene {
    pub fn new() -> SdfScene {
        SdfScene {
            nodes: Vec::new(),
            bvh_nodes: Vec::new(),
        }
    }

    pub fn add(&mut self, transform: &Transform, primitive: &SdfPrimitive) {
        self.nodes
            .push(SdfNode::new(transform, primitive, false, 0));
        self.bvh_nodes.clear();
    }

    pub fn sub(&mut self, transform: &Transform, primitive: &SdfPrimitive) {
        self.nodes.push(SdfNode::new(transform, primitive, true, 0));
        self.bvh_nodes.clear();
    }

    pub fn add_mat(&mut self, transform: &Transform, primitive: &SdfPrimitive, material_id: u32) {
        self.nodes
            .push(SdfNode::new(transform, primitive, false, material_id));
        self.bvh_nodes.clear();
    }

    pub fn sub_mat(&mut self, transform: &Transform, primitive: &SdfPrimitive, material_id: u32) {
        self.nodes
            .push(SdfNode::new(transform, primitive, true, material_id));
        self.bvh_nodes.clear();
    }

    pub fn build_bvh(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        let mut indices: Vec<u32> = (0..self.nodes.len() as u32).collect();
        let mut out_bvh = Vec::with_capacity(self.nodes.len() * 2);
        Self::bvh_build(&self.nodes, &mut indices, &mut out_bvh);
        self.bvh_nodes = out_bvh;
    }

    fn bvh_build(nodes: &[SdfNode], indices: &mut [u32], out: &mut Vec<BvhNode>) -> u32 {
        let idx = out.len() as u32;
        if indices.len() == 1 {
            let prim_idx = indices[0];
            out.push(BvhNode {
                aabb: nodes[prim_idx as usize].world_aabb.clone(),
                left: u32::MAX,
                right: u32::MAX,
                prim_idx,
            });
            return idx;
        }

        let mut union_aabb = nodes[indices[0] as usize].world_aabb.clone();
        for &i in indices.iter().skip(1) {
            union_aabb = union_aabb.union(&nodes[i as usize].world_aabb);
        }

        let ext = union_aabb.extent();
        let axis = if ext.x >= ext.y && ext.x >= ext.z {
            0
        } else if ext.y >= ext.z {
            1
        } else {
            2
        };

        indices.sort_unstable_by(|&a, &b| {
            let ca = nodes[a as usize].world_aabb.center().coords[axis];
            let cb = nodes[b as usize].world_aabb.center().coords[axis];
            ca.partial_cmp(&cb).unwrap()
        });

        let mid = indices.len() / 2;
        let (left_idx, right_idx) = indices.split_at_mut(mid);

        out.push(BvhNode {
            aabb: union_aabb,
            left: 0,
            right: 0,
            prim_idx: u32::MAX,
        });
        let left = Self::bvh_build(nodes, left_idx, out);
        let right = Self::bvh_build(nodes, right_idx, out);
        out[idx as usize].left = left;
        out[idx as usize].right = right;
        idx
    }

    pub fn get_nodes<'a>(&'a self, query: &AABB) -> Vec<&'a SdfNode> {
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

    pub fn nodes(&self) -> &[SdfNode] {
        &self.nodes
    }
    pub fn bvh_nodes(&self) -> &[BvhNode] {
        &self.bvh_nodes
    }

    pub fn nodes_mut(&mut self) -> &mut Vec<SdfNode> {
        self.bvh_nodes.clear();
        &mut self.nodes
    }

    pub fn sample(nodes: &[&SdfNode], position: &Point3<f32>) -> SdfSample {
        let mut solid = SdfSample {
            value: f32::MAX,
            gradient: Vector3::zeros(),
        };
        let mut empty = SdfSample {
            value: f32::MAX,
            gradient: Vector3::zeros(),
        };

        for node in nodes {
            let local_p = node.transform.inv_transform_point(position);
            let mut s = node.primitive.local_sample(&local_p);

            s.value *= node.transform.scale();
            s.gradient = node.transform.transform_normal(&s.gradient);

            if node.subtract {
                if s.value < empty.value {
                    empty = s;
                }
            } else {
                if s.value < solid.value {
                    solid = s;
                }
            }
        }

        if solid.value > -empty.value {
            solid
        } else {
            SdfSample {
                value: -empty.value,
                gradient: -empty.gradient,
            }
        }
    }

    pub fn fill_cache_vec(nodes: &[&SdfNode], pos: &NoisePoint, out: &mut [SdfSample; 8]) {
        let mut solid = [SdfSample {
            value: f32::MAX,
            gradient: Vector3::zeros(),
        }; 8];
        let mut empty = [SdfSample {
            value: f32::MAX,
            gradient: Vector3::zeros(),
        }; 8];
        let mut temp = [SdfSample::default(); 8];

        for node in nodes {
            let mut local_pos = NoisePoint::default();
            for i in 0..8 {
                let p = Point3::new(pos.x[i], pos.y[i], pos.z[i]);
                let local_p = node.transform.inv_transform_point(&p);
                local_pos.x[i] = local_p.x;
                local_pos.y[i] = local_p.y;
                local_pos.z[i] = local_p.z;
            }

            node.primitive.local_sample_vec(&local_pos, &mut temp);

            for i in 0..8 {
                temp[i].value *= node.transform.scale();
                temp[i].gradient = node.transform.transform_normal(&temp[i].gradient);

                if node.subtract {
                    if temp[i].value < empty[i].value {
                        empty[i] = temp[i];
                    }
                } else {
                    if temp[i].value < solid[i].value {
                        solid[i] = temp[i];
                    }
                }
            }
        }

        for i in 0..8 {
            if solid[i].value > -empty[i].value {
                out[i] = solid[i];
            } else {
                out[i] = SdfSample {
                    value: -empty[i].value,
                    gradient: -empty[i].gradient,
                };
            }
        }
    }
}

impl Default for SdfScene {
    fn default() -> Self {
        Self::new()
    }
}
