use crate::{Generator, NoisePoint, NoiseSample};
use wide::{f32x8, i32x8};

pub struct Perlin {
    seed: u32,
}

impl Perlin {
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
        // (h & 15 then clamp to 11 folds values 12-15 back into 0-3.)
        let h = (hash & i32x8::splat(15)).min(i32x8::splat(11));

        let one = f32x8::splat(1.0);
        let neg_one = f32x8::splat(-1.0);

        // s0 = sign from bit 0, s1 = sign from bit 1
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
    fn fade_simd(t: f32x8) -> f32x8 {
        // 6t^5 - 15t^4 + 10t^3 = t^3 * (t * (t * 6 - 15) + 10)
        let t2 = t * t;
        let t3 = t2 * t;
        t3 * t.mul_add(
            t.mul_add(f32x8::splat(6.0), f32x8::splat(-15.0)),
            f32x8::splat(10.0),
        )
    }

    #[inline(always)]
    fn fade_deriv_simd(t: f32x8) -> f32x8 {
        // 30t^4 - 60t^3 + 30t^2 = 30t^2 * (t^2 - 2t + 1) = 30 * t^2 * (t - 1)^2
        let t2 = t * t;
        let tm1 = t - f32x8::splat(1.0);
        f32x8::splat(30.0) * t2 * tm1 * tm1
    }

    #[inline(always)]
    fn lerp_simd(a: f32x8, b: f32x8, t: f32x8) -> f32x8 {
        a + t * (b - a)
    }
}

impl Generator for Perlin {
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

        let one_f = f32x8::splat(1.0);
        let fx1 = fx0 - one_f;
        let fy1 = fy0 - one_f;
        let fz1 = fz0 - one_f;

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

        // Gradients
        let (g000_x, g000_y, g000_z) = Self::get_gradient_simd(h000);
        let (g100_x, g100_y, g100_z) = Self::get_gradient_simd(h100);
        let (g010_x, g010_y, g010_z) = Self::get_gradient_simd(h010);
        let (g110_x, g110_y, g110_z) = Self::get_gradient_simd(h110);
        let (g001_x, g001_y, g001_z) = Self::get_gradient_simd(h001);
        let (g101_x, g101_y, g101_z) = Self::get_gradient_simd(h101);
        let (g011_x, g011_y, g011_z) = Self::get_gradient_simd(h011);
        let (g111_x, g111_y, g111_z) = Self::get_gradient_simd(h111);

        // Dot products
        let d000 = g000_x * fx0 + g000_y * fy0 + g000_z * fz0;
        let d100 = g100_x * fx1 + g100_y * fy0 + g100_z * fz0;
        let d010 = g010_x * fx0 + g010_y * fy1 + g010_z * fz0;
        let d110 = g110_x * fx1 + g110_y * fy1 + g110_z * fz0;
        let d001 = g001_x * fx0 + g001_y * fy0 + g001_z * fz1;
        let d101 = g101_x * fx1 + g101_y * fy0 + g101_z * fz1;
        let d011 = g011_x * fx0 + g011_y * fy1 + g011_z * fz1;
        let d111 = g111_x * fx1 + g111_y * fy1 + g111_z * fz1;

        // Fade curves
        let u = Self::fade_simd(fx0);
        let v = Self::fade_simd(fy0);
        let w = Self::fade_simd(fz0);

        // Interpolation
        let c00 = Self::lerp_simd(d000, d100, u);
        let c10 = Self::lerp_simd(d010, d110, u);
        let c01 = Self::lerp_simd(d001, d101, u);
        let c11 = Self::lerp_simd(d011, d111, u);

        let c0 = Self::lerp_simd(c00, c10, v);
        let c1 = Self::lerp_simd(c01, c11, v);

        let value = Self::lerp_simd(c0, c1, w);

        // Fade derivatives
        let du = Self::fade_deriv_simd(fx0);
        let dv = Self::fade_deriv_simd(fy0);
        let dw = Self::fade_deriv_simd(fz0);

        // Analytical derivatives
        // derivative of lerp(a, b, t) w.r.t x is da/dx + t*(db/dx - da/dx) + dt/dx * (b - a)

        let k0 = d100 - d000;
        let k1 = d010 - d000;
        let k2 = d001 - d000;
        let k3 = d110 - d010 - k0;
        let k4 = d101 - d001 - k0;
        let k5 = d011 - d001 - k1;
        let k6 = d111 - d011 - d101 + d001 - k3; // Which simplifies to (d111 - d011 - d101 + d001) - (d110 - d010 - d100 + d000)

        // Interpolation expanded:
        // value = d000 + u*k0 + v*k1 + w*k2 + u*v*k3 + u*w*k4 + v*w*k5 + u*v*w*k6

        // Compute base gradients of the dot products
        // We know d(d000)/dx = g000_x.
        let grad000_x = g000_x;
        let grad000_y = g000_y;
        let grad000_z = g000_z;

        let grad100_x = g100_x;
        let grad100_y = g100_y;
        let grad100_z = g100_z;
        let grad010_x = g010_x;
        let grad010_y = g010_y;
        let grad010_z = g010_z;
        let grad110_x = g110_x;
        let grad110_y = g110_y;
        let grad110_z = g110_z;
        let grad001_x = g001_x;
        let grad001_y = g001_y;
        let grad001_z = g001_z;
        let grad101_x = g101_x;
        let grad101_y = g101_y;
        let grad101_z = g101_z;
        let grad011_x = g011_x;
        let grad011_y = g011_y;
        let grad011_z = g011_z;
        let grad111_x = g111_x;
        let grad111_y = g111_y;
        let grad111_z = g111_z;

        // k0_grad_x = d(k0)/dx = grad100_x - grad000_x
        let k0_gx = grad100_x - grad000_x;
        let k0_gy = grad100_y - grad000_y;
        let k0_gz = grad100_z - grad000_z;

        let k1_gx = grad010_x - grad000_x;
        let k1_gy = grad010_y - grad000_y;
        let k1_gz = grad010_z - grad000_z;

        let k2_gx = grad001_x - grad000_x;
        let k2_gy = grad001_y - grad000_y;
        let k2_gz = grad001_z - grad000_z;

        let k3_gx = grad110_x - grad010_x - k0_gx;
        let k3_gy = grad110_y - grad010_y - k0_gy;
        let k3_gz = grad110_z - grad010_z - k0_gz;

        let k4_gx = grad101_x - grad001_x - k0_gx;
        let k4_gy = grad101_y - grad001_y - k0_gy;
        let k4_gz = grad101_z - grad001_z - k0_gz;

        let k5_gx = grad011_x - grad001_x - k1_gx;
        let k5_gy = grad011_y - grad001_y - k1_gy;
        let k5_gz = grad011_z - grad001_z - k1_gz;

        let k6_gx = grad111_x - grad011_x - grad101_x + grad001_x - k3_gx;
        let k6_gy = grad111_y - grad011_y - grad101_y + grad001_y - k3_gy;
        let k6_gz = grad111_z - grad011_z - grad101_z + grad001_z - k3_gz;

        // Total derivative w.r.t Local x
        // d(value)/dx = d000_gx
        //             + du*k0 + u*k0_gx
        //             + v*k1_gx
        //             + w*k2_gx
        //             + du*v*k3 + u*v*k3_gx
        //             + du*w*k4 + u*w*k4_gx
        //             + v*w*k5_gx
        //             + du*v*w*k6 + u*v*w*k6_gx

        let local_dx = grad000_x
            + du * k0
            + u * k0_gx
            + v * k1_gx
            + w * k2_gx
            + du * v * k3
            + u * v * k3_gx
            + du * w * k4
            + u * w * k4_gx
            + v * w * k5_gx
            + du * v * w * k6
            + u * v * w * k6_gx;

        let local_dy = grad000_y
            + u * k0_gy
            + dv * k1
            + v * k1_gy
            + w * k2_gy
            + u * dv * k3
            + u * v * k3_gy
            + u * w * k4_gy
            + dv * w * k5
            + v * w * k5_gy
            + u * dv * w * k6
            + u * v * w * k6_gy;

        let local_dz = grad000_z
            + u * k0_gz
            + v * k1_gz
            + dw * k2
            + w * k2_gz
            + u * v * k3_gz
            + u * dw * k4
            + u * w * k4_gz
            + v * dw * k5
            + v * w * k5_gz
            + u * v * dw * k6
            + u * v * w * k6_gz;

        NoiseSample {
            value: <[f32; 8]>::from(value),
            dx: <[f32; 8]>::from(local_dx),
            dy: <[f32; 8]>::from(local_dy),
            dz: <[f32; 8]>::from(local_dz),
        }
    }
}
