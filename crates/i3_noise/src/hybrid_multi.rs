use crate::{Generator, NoisePoint, NoiseSample};
use wide::f32x8;

// ── HybridMulti ───────────────────────────────────────────────────────────────
// Musgrave's hybrid multifractal: octaves are weighted by the running product of
// all previous signals, so high-amplitude regions amplify subsequent octaves,
// creating terrain with smooth valleys and sharp peaks.
//
// Algorithm (Musgrave 1994):
//   result = signal₀ · amp₀ · lod₀
//   weight = signal₀ · amp₀ · lod₀
//   for i in 1..N:
//     w_eff  = min(weight, 1)           (clamp at TOP only — negatives allowed)
//     signal = (source(pᵢ) + offset) · ampᵢ · lodᵢ
//     result += signal · w_eff
//     weight *= signal                  (product of raw, unclamped weights)
//
// Gradient bookkeeping:
//   d(w_eff)/dp   = weight_d  when weight < 1, else 0   (upper saturation only)
//   d(weight·signal)/dp = weight_d·signal + weight·sig_d  (product rule, unclamped)

pub struct HybridMulti<N> {
    pub source: N,
    pub octaves: usize,
    pub lacunarity: f32,
    pub gain: f32,
    pub offset: f32,
}

impl<N: Generator> HybridMulti<N> {
    pub fn new(source: N, octaves: usize, lacunarity: f32, gain: f32) -> Self {
        Self {
            source,
            octaves,
            lacunarity,
            gain,
            offset: 0.7,
        }
    }

    pub fn with_offset(mut self, offset: f32) -> Self {
        self.offset = offset;
        self
    }
}

impl<N: Generator> Generator for HybridMulti<N> {
    fn sample(&self, pos: &NoisePoint) -> NoiseSample {
        let x = f32x8::from(pos.x);
        let y = f32x8::from(pos.y);
        let z = f32x8::from(pos.z);
        let pdx = f32x8::from(pos.dx);
        let pdy = f32x8::from(pos.dy);
        let pdz = f32x8::from(pos.dz);

        let one = f32x8::splat(1.0);
        let off = f32x8::splat(self.offset);

        let mut freq = 1.0f32;
        let mut amp = 1.0f32;

        let mut acc_v = f32x8::ZERO;
        let mut acc_dx = f32x8::ZERO;
        let mut acc_dy = f32x8::ZERO;
        let mut acc_dz = f32x8::ZERO;

        let mut weight = f32x8::ZERO;
        let mut weight_dx = f32x8::ZERO;
        let mut weight_dy = f32x8::ZERO;
        let mut weight_dz = f32x8::ZERO;

        for i in 0..self.octaves {
            let fv = f32x8::splat(freq);

            let max_d = (pdx * fv).abs().max((pdy * fv).abs()).max((pdz * fv).abs());
            let lod = (one - max_d).max(f32x8::ZERO);

            let scaled = NoisePoint {
                x: (x * fv).into(),
                y: (y * fv).into(),
                z: (z * fv).into(),
                dx: (pdx * fv).into(),
                dy: (pdy * fv).into(),
                dz: (pdz * fv).into(),
            };
            let s = self.source.sample(&scaled);
            let sv = f32x8::from(s.value);
            let amp_lod = f32x8::splat(amp) * lod;
            let signal = (sv + off) * amp_lod;
            let sig_dx = f32x8::from(s.dx) * f32x8::splat(freq * amp) * lod;
            let sig_dy = f32x8::from(s.dy) * f32x8::splat(freq * amp) * lod;
            let sig_dz = f32x8::from(s.dz) * f32x8::splat(freq * amp) * lod;

            if i == 0 {
                acc_v = signal;
                acc_dx = sig_dx;
                acc_dy = sig_dy;
                acc_dz = sig_dz;
                weight = signal;
                weight_dx = sig_dx;
                weight_dy = sig_dy;
                weight_dz = sig_dz;
            } else {
                // Clamp at top only — negative weights contribute (per Musgrave)
                let w_eff = weight.min(one);
                let unsaturated = weight.simd_lt(one); // d(w_eff)/dp = 0 when weight >= 1
                let w_dx = unsaturated.blend(weight_dx, f32x8::ZERO);
                let w_dy = unsaturated.blend(weight_dy, f32x8::ZERO);
                let w_dz = unsaturated.blend(weight_dz, f32x8::ZERO);

                // Contribution: signal * w_eff
                acc_v = signal.mul_add(w_eff, acc_v);
                acc_dx = sig_dx.mul_add(w_eff, w_dx.mul_add(signal, acc_dx));
                acc_dy = sig_dy.mul_add(w_eff, w_dy.mul_add(signal, acc_dy));
                acc_dz = sig_dz.mul_add(w_eff, w_dz.mul_add(signal, acc_dz));

                // Update raw (unclamped) weight: weight *= signal
                let new_w_dx = weight_dx.mul_add(signal, weight * sig_dx);
                let new_w_dy = weight_dy.mul_add(signal, weight * sig_dy);
                let new_w_dz = weight_dz.mul_add(signal, weight * sig_dz);
                weight = weight * signal;
                weight_dx = new_w_dx;
                weight_dy = new_w_dy;
                weight_dz = new_w_dz;
            }

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
