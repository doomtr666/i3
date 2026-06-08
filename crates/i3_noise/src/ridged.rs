use crate::{Generator, NoisePoint, NoiseSample};
use wide::f32x8;

pub struct Ridged<N> {
    pub source: N,
    pub octaves: usize,
    pub lacunarity: f32,
    pub gain: f32,
    pub offset: f32,
}

impl<N: Generator> Ridged<N> {
    pub fn new(source: N, octaves: usize, lacunarity: f32, gain: f32) -> Self {
        Self {
            source,
            octaves,
            lacunarity,
            gain,
            offset: 1.0,
        }
    }

    pub fn with_offset(mut self, offset: f32) -> Self {
        self.offset = offset;
        self
    }
}

impl<N: Generator> Generator for Ridged<N> {
    fn sample(&self, pos: &NoisePoint) -> NoiseSample {
        let x = f32x8::from(pos.x);
        let y = f32x8::from(pos.y);
        let z = f32x8::from(pos.z);
        let pdx = f32x8::from(pos.dx);
        let pdy = f32x8::from(pos.dy);
        let pdz = f32x8::from(pos.dz);

        let offset = f32x8::splat(self.offset);
        let one = f32x8::splat(1.0);

        let mut freq = 1.0f32;
        let mut amp = 1.0f32;
        let mut weight = one;
        // World-space gradient of `weight` from the previous octave.
        // Needed for the inter-octave coupling term: signal² * d(weight)/dx * amp * lod
        let mut weight_dx = f32x8::ZERO;
        let mut weight_dy = f32x8::ZERO;
        let mut weight_dz = f32x8::ZERO;

        let mut acc_v = f32x8::ZERO;
        let mut acc_dx = f32x8::ZERO;
        let mut acc_dy = f32x8::ZERO;
        let mut acc_dz = f32x8::ZERO;

        for _ in 0..self.octaves {
            let fv = f32x8::splat(freq);

            // Per-lane LOD: linear fade when scaled footprint approaches Nyquist (>= 1)
            let max_d = (pdx * fv).abs().max((pdy * fv).abs()).max((pdz * fv).abs());
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
            let signal = offset - sv.abs();
            let signal2 = signal * signal;
            let amp_v = f32x8::splat(amp) * lod;

            acc_v = (signal2 * weight).mul_add(amp_v, acc_v);

            // Direct term:   d(signal²)/dx_world * weight * amp * lod
            //   d(signal)/dx_world = -sign(sv) * freq * s.dx
            let direct_scale =
                f32x8::splat(-2.0 * freq * amp) * signal * sv.signum() * weight * lod;
            // Coupling term: signal² * d(weight)/dx_world * amp * lod
            let coupling_scale = signal2 * amp_v;
            acc_dx =
                f32x8::from(s.dx).mul_add(direct_scale, weight_dx.mul_add(coupling_scale, acc_dx));
            acc_dy =
                f32x8::from(s.dy).mul_add(direct_scale, weight_dy.mul_add(coupling_scale, acc_dy));
            acc_dz =
                f32x8::from(s.dz).mul_add(direct_scale, weight_dz.mul_add(coupling_scale, acc_dz));

            // Propagate weight and its world-space gradient to the next octave.
            // W_next = min(signal² * gain, 1)
            // d(W_next)/dx_world = 2*gain*signal*(-sign(sv))*freq * s.dx   (when unsaturated)
            let w_next = (signal2 * f32x8::splat(self.gain)).min(one);
            let w_unsaturated = (signal2 * f32x8::splat(self.gain)).simd_lt(one);
            let new_w_scale = f32x8::splat(-2.0 * self.gain * freq) * signal * sv.signum();
            let z = f32x8::ZERO;
            weight_dx = w_unsaturated.blend(new_w_scale * f32x8::from(s.dx), z);
            weight_dy = w_unsaturated.blend(new_w_scale * f32x8::from(s.dy), z);
            weight_dz = w_unsaturated.blend(new_w_scale * f32x8::from(s.dz), z);
            weight = w_next;

            freq *= self.lacunarity;
            amp *= self.gain;
        }

        NoiseSample {
            value: acc_v.into(),
            dx: acc_dx.into(),
            dy: acc_dy.into(),
            dz: acc_dz.into(),
        }
    }
}
