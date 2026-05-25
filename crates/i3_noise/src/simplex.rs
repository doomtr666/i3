use crate::scalar_generator::{ScalarGenerator, ScalarPosition, ScalarSample};

#[derive(Clone)]
pub struct Simplex {
    seed: u32,
}

impl Simplex {
    const GRADIENTS: [(f32, f32, f32); 12] = [
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (-1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0),
        (-1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
        (-1.0, 0.0, -1.0),
        (0.0, 1.0, 1.0),
        (0.0, -1.0, 1.0),
        (0.0, 1.0, -1.0),
        (0.0, -1.0, -1.0),
    ];

    pub fn new(seed: u32) -> Self {
        Simplex { seed }
    }

    #[inline(always)]
    fn hash(&self, i: i32, j: i32, k: i32) -> (f32, f32, f32) {
        let mut h = self.seed;
        const PRIME_X: u32 = 0x9E3779B9;
        const PRIME_Y: u32 = 0x85EBCA6B;
        const PRIME_Z: u32 = 0xC2B2AE35;

        h = h.wrapping_add((i as u32).wrapping_mul(PRIME_X));
        h = h.wrapping_add((j as u32).wrapping_mul(PRIME_Y));
        h = h.wrapping_add((k as u32).wrapping_mul(PRIME_Z));

        h ^= h >> 16;
        h = h.wrapping_mul(0x85EBCA6B);
        h ^= h >> 13;
        h = h.wrapping_mul(0xC2B2AE35);
        h ^= h >> 16;

        let idx = ((h as u64 * 12) >> 32) as usize;
        Self::GRADIENTS[idx]
    }
}

impl ScalarGenerator for Simplex {
    #[inline(always)]
    fn eval_scalar(&self, pos: &ScalarPosition, _dd: f32) -> ScalarSample {
        const F3: f32 = 1.0 / 3.0;
        const G3: f32 = 1.0 / 6.0;

        let s = (pos.x + pos.y + pos.z) * F3;
        let i = (pos.x + s).floor() as i32;
        let j = (pos.y + s).floor() as i32;
        let k = (pos.z + s).floor() as i32;

        let t_unskew = (i + j + k) as f32 * G3;
        let x0 = pos.x - (i as f32 - t_unskew);
        let y0 = pos.y - (j as f32 - t_unskew);
        let z0 = pos.z - (k as f32 - t_unskew);

        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0)
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1)
            } else {
                (0, 0, 1, 1, 0, 1)
            }
        } else {
            if y0 < z0 {
                (0, 0, 1, 0, 1, 1)
            } else if x0 < z0 {
                (0, 1, 0, 0, 1, 1)
            } else {
                (0, 1, 0, 1, 1, 0)
            }
        };

        let d0 = (x0, y0, z0);
        let d1 = (
            x0 - i1 as f32 + G3,
            y0 - j1 as f32 + G3,
            z0 - k1 as f32 + G3,
        );
        let d2 = (
            x0 - i2 as f32 + 2.0 * G3,
            y0 - j2 as f32 + 2.0 * G3,
            z0 - k2 as f32 + 2.0 * G3,
        );
        let d3 = (
            x0 - 1.0 + 3.0 * G3,
            y0 - 1.0 + 3.0 * G3,
            z0 - 1.0 + 3.0 * G3,
        );

        let g0 = self.hash(i, j, k);
        let g1 = self.hash(i + i1, j + j1, k + k1);
        let g2 = self.hash(i + i2, j + j2, k + k2);
        let g3 = self.hash(i + 1, j + 1, k + 1);

        let mut v = 0.0;
        let mut dx = 0.0;
        let mut dy = 0.0;
        let mut dz = 0.0;

        for (d, g) in [(d0, g0), (d1, g1), (d2, g2), (d3, g3)].iter() {
            let t_val = 0.6 - d.0 * d.0 - d.1 * d.1 - d.2 * d.2;
            if t_val > 0.0 {
                let t2 = t_val * t_val;
                let t3 = t2 * t_val;
                let t4 = t2 * t2;

                let dot_val = d.0 * g.0 + d.1 * g.1 + d.2 * g.2;
                v += t4 * dot_val;

                let temp = t3 * dot_val * -8.0;
                dx += temp * d.0 + t4 * g.0;
                dy += temp * d.1 + t4 * g.1;
                dz += temp * d.2 + t4 * g.2;
            }
        }

        ScalarSample {
            value: v * 32.0,
            dx: dx * 32.0,
            dy: dy * 32.0,
            dz: dz * 32.0,
        }
    }
}
