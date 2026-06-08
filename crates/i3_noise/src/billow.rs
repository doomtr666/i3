use crate::{Generator, NoisePoint, NoiseSample};
use wide::f32x8;

pub struct Billow<N> {
    pub source: N,
    pub octaves: usize,
    pub lacunarity: f32,
    pub gain: f32,
}

impl<N: Generator> Billow<N> {
    pub fn new(source: N, octaves: usize, lacunarity: f32, gain: f32) -> Self {
        Self { source, octaves, lacunarity, gain }
    }
}

impl<N: Generator> Generator for Billow<N> {
    fn sample(&self, pos: &NoisePoint) -> NoiseSample {
        let x = f32x8::from(pos.x);
        let y = f32x8::from(pos.y);
        let z = f32x8::from(pos.z);
        let pdx = f32x8::from(pos.dx);
        let pdy = f32x8::from(pos.dy);
        let pdz = f32x8::from(pos.dz);

        let mut freq = 1.0f32;
        let mut amp = 1.0f32;

        let mut acc_v  = f32x8::ZERO;
        let mut acc_dx = f32x8::ZERO;
        let mut acc_dy = f32x8::ZERO;
        let mut acc_dz = f32x8::ZERO;

        for _ in 0..self.octaves {
            let fv = f32x8::splat(freq);

            let max_d = (pdx * fv).abs()
                .max((pdy * fv).abs())
                .max((pdz * fv).abs());
            let lod = (f32x8::splat(1.0) - max_d).max(f32x8::ZERO);

            let scaled_pos = NoisePoint {
                x: (x * fv).into(),
                y: (y * fv).into(),
                z: (z * fv).into(),
                dx: (pdx * fv).into(),
                dy: (pdy * fv).into(),
                dz: (pdz * fv).into(),
            };

            let s = self.source.sample(&scaled_pos);
            let sv = f32x8::from(s.value);

            // Billow transform: |sv| * 2 - 1
            // d(|sv|*2-1)/d(sv) = 2 * sign(sv)
            let transformed = sv.abs().mul_add(f32x8::splat(2.0), f32x8::splat(-1.0));
            let amp_v       = f32x8::splat(amp) * lod;
            let deriv_scale = f32x8::splat(freq * amp) * lod * sv.signum() * f32x8::splat(2.0);

            acc_v  = transformed.mul_add(amp_v, acc_v);
            acc_dx = f32x8::from(s.dx).mul_add(deriv_scale, acc_dx);
            acc_dy = f32x8::from(s.dy).mul_add(deriv_scale, acc_dy);
            acc_dz = f32x8::from(s.dz).mul_add(deriv_scale, acc_dz);

            freq *= self.lacunarity;
            amp  *= self.gain;
        }

        NoiseSample {
            value: acc_v.into(),
            dx:    acc_dx.into(),
            dy:    acc_dy.into(),
            dz:    acc_dz.into(),
        }
    }
}
