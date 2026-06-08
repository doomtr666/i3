use crate::{Generator, NoisePoint, NoiseSample};
use wide::{f32x8, i32x8};

pub struct Value {
    seed: u32,
}

impl Value {
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
    fn get_value_simd(hash: i32x8) -> f32x8 {
        use wide::bytemuck::cast;
        // hash is a u32 (i32 reinterpreted). We want a float in [-1, 1].
        // 0x007FFFFF is the mantissa mask for f32.
        // 0x3F800000 is 1.0f32.
        let mantissa_mask = i32x8::splat(0x007FFFFF);
        let float_one_bits = i32x8::splat(0x3F800000);
        let bits = (hash & mantissa_mask) | float_one_bits;
        let floats = cast::<i32x8, f32x8>(bits); // Range [1.0, 2.0)
        (floats - f32x8::splat(1.5)) * f32x8::splat(2.0) // Range [-1.0, 1.0)
    }

    #[inline(always)]
    fn fade_simd(t: f32x8) -> f32x8 {
        let t2 = t * t;
        let t3 = t2 * t;
        t3 * t.mul_add(
            t.mul_add(f32x8::splat(6.0), f32x8::splat(-15.0)),
            f32x8::splat(10.0),
        )
    }

    #[inline(always)]
    fn fade_deriv_simd(t: f32x8) -> f32x8 {
        let t2 = t * t;
        let tm1 = t - f32x8::splat(1.0);
        f32x8::splat(30.0) * t2 * tm1 * tm1
    }
}

impl Generator for Value {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let x = f32x8::from(point.x);
        let y = f32x8::from(point.y);
        let z = f32x8::from(point.z);

        let x0_f = x.floor();
        let y0_f = y.floor();
        let z0_f = z.floor();

        let x0 = x0_f.round_int();
        let y0 = y0_f.round_int();
        let z0 = z0_f.round_int();

        let one_i = i32x8::splat(1);
        let x1 = x0 + one_i;
        let y1 = y0 + one_i;
        let z1 = z0 + one_i;

        let fx0 = x - x0_f;
        let fy0 = y - y0_f;
        let fz0 = z - z0_f;

        let seed = i32x8::splat(self.seed as i32);

        // Hashing
        let h000 = Self::hash_simd(seed, x0, y0, z0);
        let h100 = Self::hash_simd(seed, x1, y0, z0);
        let h010 = Self::hash_simd(seed, x0, y1, z0);
        let h110 = Self::hash_simd(seed, x1, y1, z0);
        let h001 = Self::hash_simd(seed, x0, y0, z1);
        let h101 = Self::hash_simd(seed, x1, y0, z1);
        let h011 = Self::hash_simd(seed, x0, y1, z1);
        let h111 = Self::hash_simd(seed, x1, y1, z1);

        // Random values at corners
        let v000 = Self::get_value_simd(h000);
        let v100 = Self::get_value_simd(h100);
        let v010 = Self::get_value_simd(h010);
        let v110 = Self::get_value_simd(h110);
        let v001 = Self::get_value_simd(h001);
        let v101 = Self::get_value_simd(h101);
        let v011 = Self::get_value_simd(h011);
        let v111 = Self::get_value_simd(h111);

        // Fade curves
        let u = Self::fade_simd(fx0);
        let v = Self::fade_simd(fy0);
        let w = Self::fade_simd(fz0);

        // Fade derivatives
        let du = Self::fade_deriv_simd(fx0);
        let dv = Self::fade_deriv_simd(fy0);
        let dw = Self::fade_deriv_simd(fz0);

        // Differences
        let k0 = v100 - v000;
        let k1 = v010 - v000;
        let k2 = v001 - v000;
        let k3 = v110 - v010 - k0;
        let k4 = v101 - v001 - k0;
        let k5 = v011 - v001 - k1;
        let k6 = v111 - v011 - v101 + v001 - k3;

        // Interpolation expanded:
        // value = v000 + u*k0 + v*k1 + w*k2 + u*v*k3 + u*w*k4 + v*w*k5 + u*v*w*k6
        let value = v000
            + u * k0
            + v * k1
            + w * k2
            + u * v * k3
            + u * w * k4
            + v * w * k5
            + u * v * w * k6;

        // Analytical derivatives
        // Note: k constants don't have gradients because they are fixed values at lattice nodes
        let local_dx = du * (k0 + v * k3 + w * k4 + v * w * k6);
        let local_dy = dv * (k1 + u * k3 + w * k5 + u * w * k6);
        let local_dz = dw * (k2 + u * k4 + v * k5 + u * v * k6);

        NoiseSample {
            value: <[f32; 8]>::from(value),
            dx: <[f32; 8]>::from(local_dx),
            dy: <[f32; 8]>::from(local_dy),
            dz: <[f32; 8]>::from(local_dz),
        }
    }
}
