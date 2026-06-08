use crate::{Generator, NoisePoint, NoiseSample};
use wide::{f32x8, i32x8};

const SKEW_3D: f32 = 1.0 / 3.0;
const UNSKEW_3D: f32 = 1.0 / 6.0;

pub struct Simplex {
    seed: u32,
}

impl Simplex {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    #[inline(always)]
    fn hash_simd(seed: i32x8, x: i32x8, y: i32x8, z: i32x8) -> i32x8 {
        let c1 = i32x8::splat(0x9E3779B1u32 as i32);
        let c2 = i32x8::splat(0x85EBCA6Bu32 as i32);
        let c3 = i32x8::splat(0xC2B2AE35u32 as i32);

        let mut h = (seed + x) * c1;
        h = ((h ^ (h >> 16)) + y) * c2;
        h = ((h ^ (h >> 13)) + z) * c3;

        h ^ (h >> 16)
    }

    #[inline(always)]
    fn get_gradient_simd(hash: i32x8) -> (f32x8, f32x8, f32x8) {
        use wide::bytemuck::cast;

        // Map to 0..11: gives all 12 standard gradients with equal probability.
        let h = (hash & i32x8::splat(15)).min(i32x8::splat(11));

        let one = f32x8::splat(1.0);
        let neg_one = f32x8::splat(-1.0);

        let s0 =
            cast::<i32x8, f32x8>((h & i32x8::splat(1)).simd_eq(i32x8::ZERO)).blend(one, neg_one);
        let s1 =
            cast::<i32x8, f32x8>((h & i32x8::splat(2)).simd_eq(i32x8::ZERO)).blend(one, neg_one);

        // h  0-3 : gz=0, gx=s0, gy=s1  → (±1, ±1,  0)
        // h  4-7 : gy=0, gx=s0, gz=s1  → (±1,  0, ±1)
        // h 8-11 : gx=0, gy=s0, gz=s1  → ( 0, ±1, ±1)
        let is_z_zero = cast::<i32x8, f32x8>(h.simd_lt(i32x8::splat(4)));
        let is_x_zero = cast::<i32x8, f32x8>(h.simd_gt(i32x8::splat(7)));

        let gx = is_x_zero.blend(f32x8::ZERO, s0);
        let gy_nz = is_z_zero.blend(s1, f32x8::ZERO);
        let gy = is_x_zero.blend(s0, gy_nz);
        let gz = is_z_zero.blend(f32x8::ZERO, s1);

        (gx, gy, gz)
    }

    #[inline(always)]
    fn extrapolate_simd(
        &self,
        xsb: i32x8,
        ysb: i32x8,
        zsb: i32x8,
        dx: f32x8,
        dy: f32x8,
        dz: f32x8,
    ) -> (f32x8, f32x8, f32x8, f32x8) {
        // r² = 0.5 is the largest value that keeps excluded simplex vertices
        // outside the sphere of influence (max |d_exclu|² at boundaries = 13/24 ≈ 0.5417).
        // Using r² > 0.5417 (e.g. 0.6) causes C0 discontinuities at simplex boundaries.
        let attn = dx.mul_add(-dx, dy.mul_add(-dy, dz.mul_add(-dz, f32x8::splat(0.5))));
        let active = attn.simd_gt(f32x8::ZERO);

        let seed = i32x8::splat(self.seed as i32);
        let hash = Self::hash_simd(seed, xsb, ysb, zsb);
        let (gx, gy, gz) = Self::get_gradient_simd(hash);

        let dot = dx.mul_add(gx, dy.mul_add(gy, dz * gz));
        let attn2 = attn * attn;
        let attn4 = attn2 * attn2;
        let deriv = f32x8::splat(-8.0) * attn2 * attn * dot;

        let value = attn4 * dot;
        let d_x = deriv.mul_add(dx, attn4 * gx);
        let d_y = deriv.mul_add(dy, attn4 * gy);
        let d_z = deriv.mul_add(dz, attn4 * gz);

        let z = f32x8::ZERO;
        (
            active.blend(value, z),
            active.blend(d_x, z),
            active.blend(d_y, z),
            active.blend(d_z, z),
        )
    }
}

impl Generator for Simplex {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let skew = f32x8::splat(SKEW_3D);
        let unskew = f32x8::splat(UNSKEW_3D);

        let x = f32x8::from(point.x);
        let y = f32x8::from(point.y);
        let z = f32x8::from(point.z);

        let stretch = (x + y + z) * skew;
        let xs = x + stretch;
        let ys = y + stretch;
        let zs = z + stretch;

        let xs_f = xs.floor();
        let ys_f = ys.floor();
        let zs_f = zs.floor();
        let xsb = xs_f.round_int();
        let ysb = ys_f.round_int();
        let zsb = zs_f.round_int();

        let squish = (xs_f + ys_f + zs_f) * unskew;
        let dx0 = x - (xs_f - squish);
        let dy0 = y - (ys_f - squish);
        let dz0 = z - (zs_f - squish);

        let dx_ge_dy = dx0.simd_ge(dy0);
        let dy_ge_dz = dy0.simd_ge(dz0);
        let dx_ge_dz = dx0.simd_ge(dz0);
        let dx_lt_dy = dx0.simd_lt(dy0);
        let dy_lt_dz = dy0.simd_lt(dz0);
        let dx_lt_dz = dx0.simd_lt(dz0);

        let zero = f32x8::ZERO;
        let one = f32x8::splat(1.0);

        let i1 = (dx_ge_dy & (dy_ge_dz | dx_ge_dz)).blend(one, zero);
        let j1 = (dx_lt_dy & dy_ge_dz).blend(one, zero);
        let k1 = (dy_lt_dz & dx_lt_dz).blend(one, zero);

        let i2 = (dx_ge_dy | (dy_ge_dz & dx_ge_dz)).blend(one, zero);
        let j2 = (dx_lt_dy | dy_ge_dz).blend(one, zero);
        let k2 = (dy_lt_dz | (dx_lt_dy & dx_lt_dz)).blend(one, zero);

        let u1 = unskew;
        let u2 = unskew * f32x8::splat(2.0);
        let u3 = unskew * f32x8::splat(3.0);

        let (v0, d0x, d0y, d0z) = self.extrapolate_simd(xsb, ysb, zsb, dx0, dy0, dz0);

        let (v1, d1x, d1y, d1z) = self.extrapolate_simd(
            xsb + i1.round_int(),
            ysb + j1.round_int(),
            zsb + k1.round_int(),
            dx0 - i1 + u1,
            dy0 - j1 + u1,
            dz0 - k1 + u1,
        );

        let (v2, d2x, d2y, d2z) = self.extrapolate_simd(
            xsb + i2.round_int(),
            ysb + j2.round_int(),
            zsb + k2.round_int(),
            dx0 - i2 + u2,
            dy0 - j2 + u2,
            dz0 - k2 + u2,
        );

        let one_i = i32x8::splat(1);
        let one_f = f32x8::splat(1.0);
        let (v3, d3x, d3y, d3z) = self.extrapolate_simd(
            xsb + one_i,
            ysb + one_i,
            zsb + one_i,
            dx0 - one_f + u3,
            dy0 - one_f + u3,
            dz0 - one_f + u3,
        );

        // Normalization: max single-vertex contribution with r²=0.5 is
        // (4/9)⁴ × 1/3 ≈ 0.0130  →  norm ≈ 1/0.0130 ≈ 76.
        let norm = f32x8::splat(76.0);
        NoiseSample {
            value: <[f32; 8]>::from((v0 + v1 + v2 + v3) * norm),
            dx: <[f32; 8]>::from((d0x + d1x + d2x + d3x) * norm),
            dy: <[f32; 8]>::from((d0y + d1y + d2y + d3y) * norm),
            dz: <[f32; 8]>::from((d0z + d1z + d2z + d3z) * norm),
        }
    }
}
