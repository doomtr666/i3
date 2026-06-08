use crate::{Generator, NoisePoint, NoiseSample};
use wide::{f32x8, i32x8};

const PX: i32 = 0x5205_402Bu32 as i32;
const PY: i32 = 0x598C_D327u32 as i32;
const PZ: i32 = 0x5BCC_226Eu32 as i32;
const HM: i32 = 0x53A3_F72Du32 as i32;
const HM2: i32 = 0x6D2B_79F5u32 as i32; // independent scrambler for y offset
const HM3: i32 = 0x27D4_EB2Fu32 as i32; // independent scrambler for z offset

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorleyOutput {
    F1,        // distance to nearest seed
    F2,        // distance to second nearest
    F2MinusF1, // cell border pattern
    CellValue, // random constant per Voronoi cell (gradient = 0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorleyDistance {
    Euclidean,   // sqrt(dx²+dy²+dz²)  gradient = unit vector
    EuclideanSq, // dx²+dy²+dz²         gradient = 2·d
    Manhattan,   // |dx|+|dy|+|dz|      gradient = sign(d)
}

pub struct Worley {
    pub seed: u32,
    pub jitter: f32, // 0 = grid  1 = fully random
    pub output: WorleyOutput,
    pub distance: WorleyDistance,
}

impl Worley {
    pub fn new(seed: u32) -> Self {
        Self {
            seed,
            jitter: 1.0,
            output: WorleyOutput::F1,
            distance: WorleyDistance::Euclidean,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    #[inline(always)]
    fn cell_hash(seed: i32x8, ix: i32x8, iy: i32x8, iz: i32x8) -> i32x8 {
        let mut h =
            seed ^ (ix * i32x8::splat(PX)) ^ (iy * i32x8::splat(PY)) ^ (iz * i32x8::splat(PZ));
        h = h * i32x8::splat(HM);
        h ^ (h >> 16)
    }

    /// Hash bits → uniform float in [0, 1)
    #[inline(always)]
    fn to_01(h: i32x8) -> f32x8 {
        use wide::bytemuck::cast;
        let b = (h & i32x8::splat(0x007FFFFF)) | i32x8::splat(0x3F800000u32 as i32);
        cast::<i32x8, f32x8>(b) - f32x8::splat(1.0)
    }

    /// Blend i32x8 values using an f32x8 boolean mask (0.0 = false, 0xFFFFFFFF = true)
    #[inline(always)]
    fn blend_i(mask: f32x8, a: i32x8, b: i32x8) -> i32x8 {
        use wide::bytemuck::cast;
        let m: i32x8 = cast(mask);
        (m & a) | ((!m) & b)
    }

    /// Compute (value, gradient) for a distance+displacement, according to self.distance.
    #[inline(always)]
    fn value_and_grad(
        &self,
        dist: f32x8,
        dx: f32x8,
        dy: f32x8,
        dz: f32x8,
    ) -> (f32x8, f32x8, f32x8, f32x8) {
        // dx = feature - query  →  d(dist)/d(query) = (query - feature)/dist = -dx/dist
        match self.distance {
            WorleyDistance::Euclidean => {
                let inv = f32x8::splat(1.0) / dist.max(f32x8::splat(1e-10));
                (dist, -dx * inv, -dy * inv, -dz * inv)
            }
            WorleyDistance::EuclideanSq => {
                let two = f32x8::splat(2.0);
                (dist, -two * dx, -two * dy, -two * dz)
            }
            WorleyDistance::Manhattan => (dist, -dx.signum(), -dy.signum(), -dz.signum()),
        }
    }
}

// ── Generator impl ────────────────────────────────────────────────────────────

impl Generator for Worley {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let px = f32x8::from(point.x);
        let py = f32x8::from(point.y);
        let pz = f32x8::from(point.z);

        // Integer cell coords and fractional offset
        let px_a: [f32; 8] = px.into();
        let py_a: [f32; 8] = py.into();
        let pz_a: [f32; 8] = pz.into();
        let ix = i32x8::from(px_a.map(|v| v.floor() as i32));
        let iy = i32x8::from(py_a.map(|v| v.floor() as i32));
        let iz = i32x8::from(pz_a.map(|v| v.floor() as i32));
        let fx = px
            - f32x8::from({
                let a: [i32; 8] = ix.into();
                a.map(|v| v as f32)
            });
        let fy = py
            - f32x8::from({
                let a: [i32; 8] = iy.into();
                a.map(|v| v as f32)
            });
        let fz = pz
            - f32x8::from({
                let a: [i32; 8] = iz.into();
                a.map(|v| v as f32)
            });

        let seed = i32x8::splat(self.seed as i32);
        let jitter = f32x8::splat(self.jitter);
        let inf = f32x8::splat(f32::MAX);

        let mut f1_dist = inf;
        let mut f1_dx = f32x8::ZERO;
        let mut f1_dy = f32x8::ZERO;
        let mut f1_dz = f32x8::ZERO;
        let mut f2_dist = inf;
        let mut f2_dx = f32x8::ZERO;
        let mut f2_dy = f32x8::ZERO;
        let mut f2_dz = f32x8::ZERO;
        let mut f1_hash = i32x8::splat(0);

        // Search 3×3×3 neighboring cells
        for oz in -1i32..=1 {
            for oy in -1i32..=1 {
                for ox in -1i32..=1 {
                    let cx = ix + i32x8::splat(ox);
                    let cy = iy + i32x8::splat(oy);
                    let cz = iz + i32x8::splat(oz);
                    let h = Self::cell_hash(seed, cx, cy, cz);

                    // Feature point: cell center (ox+0.5, oy+0.5, oz+0.5) + random jitter in [-0.5, 0.5)*jitter
                    let half = f32x8::splat(0.5);
                    let hfx = (Self::to_01(h) - half) * jitter;
                    let hfy = (Self::to_01(h * i32x8::splat(HM2)) - half) * jitter;
                    let hfz = (Self::to_01(h * i32x8::splat(HM3)) - half) * jitter;

                    // Displacement from query fractional position to feature point
                    let dx = f32x8::splat(ox as f32) + half + hfx - fx;
                    let dy = f32x8::splat(oy as f32) + half + hfy - fy;
                    let dz = f32x8::splat(oz as f32) + half + hfz - fz;

                    let dist = match self.distance {
                        WorleyDistance::Euclidean => (dx * dx + dy * dy + dz * dz).sqrt(),
                        WorleyDistance::EuclideanSq => dx * dx + dy * dy + dz * dz,
                        WorleyDistance::Manhattan => dx.abs() + dy.abs() + dz.abs(),
                    };

                    let is_f1 = dist.simd_lt(f1_dist); // new F1
                    let is_f2 = dist.simd_lt(f2_dist) & f1_dist.simd_le(dist); // new F2 only

                    // Old F1 → F2 where we have a better F1
                    f2_dist = is_f1.blend(f1_dist, f2_dist);
                    f2_dx = is_f1.blend(f1_dx, f2_dx);
                    f2_dy = is_f1.blend(f1_dy, f2_dy);
                    f2_dz = is_f1.blend(f1_dz, f2_dz);

                    // New dist → F2 (between F1 and old F2)
                    f2_dist = is_f2.blend(dist, f2_dist);
                    f2_dx = is_f2.blend(dx, f2_dx);
                    f2_dy = is_f2.blend(dy, f2_dy);
                    f2_dz = is_f2.blend(dz, f2_dz);

                    // Update F1
                    f1_dist = is_f1.blend(dist, f1_dist);
                    f1_dx = is_f1.blend(dx, f1_dx);
                    f1_dy = is_f1.blend(dy, f1_dy);
                    f1_dz = is_f1.blend(dz, f1_dz);
                    f1_hash = Self::blend_i(is_f1, h, f1_hash);
                }
            }
        }

        // Compute output
        let (value, gx, gy, gz) = match self.output {
            WorleyOutput::F1 => self.value_and_grad(f1_dist, f1_dx, f1_dy, f1_dz),
            WorleyOutput::F2 => self.value_and_grad(f2_dist, f2_dx, f2_dy, f2_dz),
            WorleyOutput::F2MinusF1 => {
                let (v1, gx1, gy1, gz1) = self.value_and_grad(f1_dist, f1_dx, f1_dy, f1_dz);
                let (v2, gx2, gy2, gz2) = self.value_and_grad(f2_dist, f2_dx, f2_dy, f2_dz);
                (v2 - v1, gx2 - gx1, gy2 - gy1, gz2 - gz1)
            }
            WorleyOutput::CellValue => {
                // Random constant per Voronoi cell; gradient is zero (piecewise constant).
                let v = Self::to_01(f1_hash) * f32x8::splat(2.0) - f32x8::splat(1.0);
                (v, f32x8::ZERO, f32x8::ZERO, f32x8::ZERO)
            }
        };

        NoiseSample {
            value: value.into(),
            dx: gx.into(),
            dy: gy.into(),
            dz: gz.into(),
        }
    }
}
