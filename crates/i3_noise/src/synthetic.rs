use crate::{Generator, NoisePoint, NoiseSample};
use std::f32::consts::TAU;
use wide::{f32x8, i32x8};

#[inline(always)]
fn i2f(v: i32x8) -> f32x8 {
    let a: [i32; 8] = v.into();
    f32x8::from(a.map(|x| x as f32))
}

// ── Checkerboard ──────────────────────────────────────────────────────────────
// Alternating +1 / -1 per unit cell.  Gradient = 0 everywhere (piecewise constant).

pub struct Checkerboard;

impl Generator for Checkerboard {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let ix = f32x8::from(point.x).floor().round_int();
        let iy = f32x8::from(point.y).floor().round_int();
        let iz = f32x8::from(point.z).floor().round_int();
        // parity ∈ {0, 1}, value = 1 - 2*parity = +1 or -1
        let parity = (ix + iy + iz) & i32x8::splat(1);
        NoiseSample {
            value: (f32x8::splat(1.0) - i2f(parity * i32x8::splat(2))).into(),
            dx: [0.0; 8],
            dy: [0.0; 8],
            dz: [0.0; 8],
        }
    }
}

// ── Spheres ───────────────────────────────────────────────────────────────────
// Concentric spheres centered at origin: cos(2π · frequency · r)
// C∞ everywhere; gradient undefined only at r=0.

pub struct Spheres {
    pub frequency: f32,
}

impl Spheres {
    pub fn new(frequency: f32) -> Self { Self { frequency } }
}

impl Generator for Spheres {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let x = f32x8::from(point.x);
        let y = f32x8::from(point.y);
        let z = f32x8::from(point.z);
        let r = (x * x + y * y + z * z).sqrt();
        let (sin_a, cos_a) = (r * f32x8::splat(self.frequency * TAU)).sin_cos();
        // d(cos(2πfr))/dp = -sin(2πfr) · 2πf · p/r
        let scale = -sin_a * f32x8::splat(self.frequency * TAU)
            / r.max(f32x8::splat(1e-10));
        NoiseSample {
            value: cos_a.into(),
            dx: (scale * x).into(),
            dy: (scale * y).into(),
            dz: (scale * z).into(),
        }
    }
}

// ── Cylinders ─────────────────────────────────────────────────────────────────
// Concentric cylinders around the Y axis: cos(2π · frequency · sqrt(x²+z²))
// C∞ everywhere; gradient undefined only on the Y axis.

pub struct Cylinders {
    pub frequency: f32,
}

impl Cylinders {
    pub fn new(frequency: f32) -> Self { Self { frequency } }
}

impl Generator for Cylinders {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let x = f32x8::from(point.x);
        let z = f32x8::from(point.z);
        let r = (x * x + z * z).sqrt();
        let (sin_a, cos_a) = (r * f32x8::splat(self.frequency * TAU)).sin_cos();
        // d/dp_x = -sin · 2πf · x/r,  d/dp_y = 0,  d/dp_z = -sin · 2πf · z/r
        let scale = -sin_a * f32x8::splat(self.frequency * TAU)
            / r.max(f32x8::splat(1e-10));
        NoiseSample {
            value: cos_a.into(),
            dx: (scale * x).into(),
            dy: [0.0; 8],
            dz: (scale * z).into(),
        }
    }
}
