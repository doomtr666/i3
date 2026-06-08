use crate::{Generator, NoiseSample, NoisePoint};
use wide::f32x8;

pub struct DomainScale<G> {
    pub inner: G,
    pub factor: f32,
}

impl<G> DomainScale<G> {
    pub fn new(inner: G, factor: f32) -> Self {
        Self { inner, factor }
    }
}

impl<G: Generator> Generator for DomainScale<G> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let f = f32x8::splat(self.factor);
        let scaled = NoisePoint {
            x: (f32x8::from(point.x) * f).into(),
            y: (f32x8::from(point.y) * f).into(),
            z: (f32x8::from(point.z) * f).into(),
            dx: (f32x8::from(point.dx) * f).into(),
            dy: (f32x8::from(point.dy) * f).into(),
            dz: (f32x8::from(point.dz) * f).into(),
        };
        let s = self.inner.sample(&scaled);
        // Chain rule: d(noise(factor*p))/dp = factor * d(noise)/d(scaled_p)
        NoiseSample {
            value: s.value,
            dx: (f32x8::from(s.dx) * f).into(),
            dy: (f32x8::from(s.dy) * f).into(),
            dz: (f32x8::from(s.dz) * f).into(),
        }
    }
}

pub struct DomainWarp<G, Gx, Gy, Gz> {
    pub inner: G,
    pub gen_x: Gx,
    pub gen_y: Gy,
    pub gen_z: Gz,
    pub intensity: f32,
}

impl<G, Gx, Gy, Gz> DomainWarp<G, Gx, Gy, Gz>
where
    G: Generator,
    Gx: Generator,
    Gy: Generator,
    Gz: Generator,
{
    pub fn new(inner: G, gen_x: Gx, gen_y: Gy, gen_z: Gz, intensity: f32) -> Self {
        Self { inner, gen_x, gen_y, gen_z, intensity }
    }
}

impl<G, Gx, Gy, Gz> Generator for DomainWarp<G, Gx, Gy, Gz>
where
    G: Generator,
    Gx: Generator,
    Gy: Generator,
    Gz: Generator,
{
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        // 1. Evaluate the 3 warp fields at the original point
        let sx = self.gen_x.sample(point);
        let sy = self.gen_y.sample(point);
        let sz = self.gen_z.sample(point);

        let iv = f32x8::splat(self.intensity);
        let px = f32x8::from(point.x);
        let py = f32x8::from(point.y);
        let pz = f32x8::from(point.z);
        let d0 = f32x8::from(point.dx);
        let d1 = f32x8::from(point.dy);
        let d2 = f32x8::from(point.dz);

        // 2. Warped position: p' = p + intensity * W(p)
        let new_x = iv.mul_add(f32x8::from(sx.value), px);
        let new_y = iv.mul_add(f32x8::from(sy.value), py);
        let new_z = iv.mul_add(f32x8::from(sz.value), pz);

        // 3. Forward Jacobian on footprint: d' = d + intensity * dot(∇W_i, d)
        let dot_dx = f32x8::from(sx.dx).mul_add(d0, f32x8::from(sx.dy).mul_add(d1, f32x8::from(sx.dz) * d2));
        let dot_dy = f32x8::from(sy.dx).mul_add(d0, f32x8::from(sy.dy).mul_add(d1, f32x8::from(sy.dz) * d2));
        let dot_dz = f32x8::from(sz.dx).mul_add(d0, f32x8::from(sz.dy).mul_add(d1, f32x8::from(sz.dz) * d2));

        let eval_pos = NoisePoint {
            x: new_x.into(),
            y: new_y.into(),
            z: new_z.into(),
            dx: iv.mul_add(dot_dx, d0).into(),
            dy: iv.mul_add(dot_dy, d1).into(),
            dz: iv.mul_add(dot_dz, d2).into(),
        };

        // 4. Sample inner generator at warped position
        let s = self.inner.sample(&eval_pos);

        let sdx = f32x8::from(s.dx);
        let sdy = f32x8::from(s.dy);
        let sdz = f32x8::from(s.dz);

        // 5. Transpose Jacobian on gradient: grad_world = J_warp^T * grad_warped
        //    grad_world_j = s.d_j + intensity * sum_i(s.d_i * d(Wi)/d(p_j))
        let dot_gx = sdx.mul_add(f32x8::from(sx.dx), sdy.mul_add(f32x8::from(sy.dx), sdz * f32x8::from(sz.dx)));
        let dot_gy = sdx.mul_add(f32x8::from(sx.dy), sdy.mul_add(f32x8::from(sy.dy), sdz * f32x8::from(sz.dy)));
        let dot_gz = sdx.mul_add(f32x8::from(sx.dz), sdy.mul_add(f32x8::from(sy.dz), sdz * f32x8::from(sz.dz)));

        NoiseSample {
            value: s.value,
            dx: iv.mul_add(dot_gx, sdx).into(),
            dy: iv.mul_add(dot_gy, sdy).into(),
            dz: iv.mul_add(dot_gz, sdz).into(),
        }
    }
}

pub struct DomainTranslate<G> {
    pub inner: G,
    pub tx: f32,
    pub ty: f32,
    pub tz: f32,
}

impl<G> DomainTranslate<G> {
    pub fn new(inner: G, tx: f32, ty: f32, tz: f32) -> Self {
        Self { inner, tx, ty, tz }
    }
}

impl<G: Generator> Generator for DomainTranslate<G> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        // Translation is a constant offset — derivatives are unchanged
        let translated = NoisePoint {
            x: (f32x8::from(point.x) + f32x8::splat(self.tx)).into(),
            y: (f32x8::from(point.y) + f32x8::splat(self.ty)).into(),
            z: (f32x8::from(point.z) + f32x8::splat(self.tz)).into(),
            dx: point.dx,
            dy: point.dy,
            dz: point.dz,
        };
        self.inner.sample(&translated)
    }
}

pub struct DomainRotate<G> {
    pub inner: G,
    // Row-major rotation matrix, built from XYZ intrinsic Euler angles.
    // p' = matrix * p ;  grad_world = matrix^T * grad_local  (R orthogonal → R^T = R^{-1})
    pub matrix: [[f32; 3]; 3],
}

impl<G> DomainRotate<G> {
    /// `pitch_x`, `yaw_y`, `roll_z` in radians. Rotation order: first Z, then Y, then X.
    pub fn new(inner: G, pitch_x: f32, yaw_y: f32, roll_z: f32) -> Self {
        let (sx, cx) = pitch_x.sin_cos();
        let (sy, cy) = yaw_y.sin_cos();
        let (sz, cz) = roll_z.sin_cos();
        let matrix = [
            [ cy * cz,                   -cy * sz,                  sy       ],
            [ cx * sz + sx * sy * cz,     cx * cz - sx * sy * sz,  -sx * cy  ],
            [ sx * sz - cx * sy * cz,     sx * cz + cx * sy * sz,   cx * cy  ],
        ];
        Self { inner, matrix }
    }
}

impl<G: Generator> Generator for DomainRotate<G> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let r = &self.matrix;
        let px = f32x8::from(point.x);
        let py = f32x8::from(point.y);
        let pz = f32x8::from(point.z);
        let fdx = f32x8::from(point.dx);
        let fdy = f32x8::from(point.dy);
        let fdz = f32x8::from(point.dz);

        let rot = |v0: f32x8, v1: f32x8, v2: f32x8, row: usize| {
            f32x8::splat(r[row][0]) * v0
                + f32x8::splat(r[row][1]) * v1
                + f32x8::splat(r[row][2]) * v2
        };

        let rotated = NoisePoint {
            x: rot(px, py, pz, 0).into(),
            y: rot(px, py, pz, 1).into(),
            z: rot(px, py, pz, 2).into(),
            dx: rot(fdx, fdy, fdz, 0).into(),
            dy: rot(fdx, fdy, fdz, 1).into(),
            dz: rot(fdx, fdy, fdz, 2).into(),
        };

        let s = self.inner.sample(&rotated);
        let gdx = f32x8::from(s.dx);
        let gdy = f32x8::from(s.dy);
        let gdz = f32x8::from(s.dz);

        // grad_world = R^T * grad_local  (transpose = column access)
        NoiseSample {
            value: s.value,
            dx: (f32x8::splat(r[0][0]) * gdx + f32x8::splat(r[1][0]) * gdy + f32x8::splat(r[2][0]) * gdz).into(),
            dy: (f32x8::splat(r[0][1]) * gdx + f32x8::splat(r[1][1]) * gdy + f32x8::splat(r[2][1]) * gdz).into(),
            dz: (f32x8::splat(r[0][2]) * gdx + f32x8::splat(r[1][2]) * gdy + f32x8::splat(r[2][2]) * gdz).into(),
        }
    }
}
