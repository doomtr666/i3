use i3_math::Transform;

// ─── GPU node layout ──────────────────────────────────────────────────────────

pub const TAG_SPHERE:       u32 = 0;  // geom.x = radius
pub const TAG_BOX:          u32 = 1;  // geom.xyz = half_extents
pub const TAG_CAPSULE:      u32 = 2;  // geom.x = half_height, geom.y = radius
pub const TAG_CYLINDER:     u32 = 3;  // geom.x = half_height, geom.y = radius
pub const TAG_TORUS:        u32 = 4;  // geom.x = major_r, geom.y = minor_r
pub const TAG_TERRAIN:      u32 = 5;  // geom.x = hscale, geom.y = amplitude, geom.z = octaves
pub const TAG_UNION:        u32 = 10; // pops 2, push min(a,b)
pub const TAG_SUBTRACTION:  u32 = 11; // pops 2, push max(a,-b)
pub const TAG_SMOOTH_UNION: u32 = 12; // pops 2, smooth-min; geom[0] = k

/// GPU node — every field is a float4/uint4 to match std430 alignment exactly.
/// Layout (6 × 16 = 96 bytes):
///   header:       x=tag, yzw=pad
///   pos_scale:    xyz=position, w=scale
///   rotation:     quaternion xyzw
///   geom:         geometry params (sphere: x=r; box: xyz=he; capsule/cylinder: x=h,y=r; torus: x=R,y=r)
///                 for ops: x=smooth_k
///   albedo_rough: xyz=albedo, w=roughness
///   metal_pad:    x=metallic, yzw=0
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuSdfNode {
    pub header:       [u32; 4], // x=tag
    pub pos_scale:    [f32; 4], // xyz=position, w=scale
    pub rotation:     [f32; 4], // quaternion xyzw
    pub geom:         [f32; 4],
    pub albedo_rough: [f32; 4], // xyz=albedo, w=roughness
    pub metal_pad:    [f32; 4], // x=metallic
}

// ─── CPU material ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SdfMaterial {
    pub albedo:    [f32; 3],
    pub roughness: f32,
    pub metallic:  f32,
}

impl Default for SdfMaterial {
    fn default() -> Self {
        Self {
            albedo:    [0.75, 0.73, 0.70],
            roughness: 0.6,
            metallic:  0.0,
        }
    }
}

// ─── CPU shape ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub enum SdfShape {
    Sphere   { radius: f32 },
    Box      { half_extents: [f32; 3] },
    Capsule  { half_height: f32, radius: f32 },
    Cylinder { half_height: f32, radius: f32 },
    Torus    { major_radius: f32, minor_radius: f32 },
    Terrain  { scale: f32, amplitude: f32, octaves: u32 },
}

// ─── CPU CSG tree ─────────────────────────────────────────────────────────────

pub enum SdfNode {
    Leaf {
        shape:     SdfShape,
        transform: Transform,
        material:  SdfMaterial,
    },
    Union {
        left:  Box<SdfNode>,
        right: Box<SdfNode>,
    },
    Subtraction {
        solid: Box<SdfNode>,
        hole:  Box<SdfNode>,
    },
    SmoothUnion {
        left:  Box<SdfNode>,
        right: Box<SdfNode>,
        k:     f32,
    },
}

impl SdfNode {
    /// Serialize the CSG tree to a post-order (RPN) flat GPU list.
    pub fn to_gpu_flat(&self, out: &mut Vec<GpuSdfNode>) {
        match self {
            SdfNode::Leaf { shape, transform, material } => {
                let t = transform.translation();
                let r = transform.rotation();
                let rv = r.as_vector(); // (i, j, k, w) in nalgebra
                let geom = match shape {
                    SdfShape::Sphere   { radius }          => [*radius, 0.0, 0.0, 0.0],
                    SdfShape::Box      { half_extents: e } => [e[0], e[1], e[2], 0.0],
                    SdfShape::Capsule  { half_height: h, radius: r } => [*h, *r, 0.0, 0.0],
                    SdfShape::Cylinder { half_height: h, radius: r } => [*h, *r, 0.0, 0.0],
                    SdfShape::Torus    { major_radius: a, minor_radius: b } => [*a, *b, 0.0, 0.0],
                    SdfShape::Terrain  { scale: sc, amplitude: a, octaves: o } => [*sc, *a, *o as f32, 0.0],
                };
                let tag = match shape {
                    SdfShape::Sphere   { .. } => TAG_SPHERE,
                    SdfShape::Box      { .. } => TAG_BOX,
                    SdfShape::Capsule  { .. } => TAG_CAPSULE,
                    SdfShape::Cylinder { .. } => TAG_CYLINDER,
                    SdfShape::Torus    { .. } => TAG_TORUS,
                    SdfShape::Terrain  { .. } => TAG_TERRAIN,
                };
                out.push(GpuSdfNode {
                    header:       [tag, 0, 0, 0],
                    pos_scale:    [t.x, t.y, t.z, transform.scale()],
                    rotation:     [rv.x, rv.y, rv.z, rv.w],
                    geom,
                    albedo_rough: [material.albedo[0], material.albedo[1], material.albedo[2], material.roughness],
                    metal_pad:    [material.metallic, 0.0, 0.0, 0.0],
                });
            }

            SdfNode::Union { left, right } => {
                left.to_gpu_flat(out);
                right.to_gpu_flat(out);
                out.push(GpuSdfNode {
                    header: [TAG_UNION, 0, 0, 0],
                    ..bytemuck::Zeroable::zeroed()
                });
            }

            SdfNode::Subtraction { solid, hole } => {
                solid.to_gpu_flat(out);
                hole.to_gpu_flat(out);
                out.push(GpuSdfNode {
                    header: [TAG_SUBTRACTION, 0, 0, 0],
                    ..bytemuck::Zeroable::zeroed()
                });
            }

            SdfNode::SmoothUnion { left, right, k } => {
                left.to_gpu_flat(out);
                right.to_gpu_flat(out);
                out.push(GpuSdfNode {
                    header: [TAG_SMOOTH_UNION, 0, 0, 0],
                    geom:   [*k, 0.0, 0.0, 0.0],
                    ..bytemuck::Zeroable::zeroed()
                });
            }
        }
    }
}
