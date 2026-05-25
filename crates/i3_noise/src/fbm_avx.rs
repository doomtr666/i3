#![allow(unsafe_op_in_unsafe_fn)]
use crate::{SimplexAvx, VecGenerator, VecPosition, VecSample};
use core::arch::x86_64::*;

#[derive(Clone)]
pub struct FbmAvx {
    pub source: SimplexAvx,
    pub octaves: usize,
    pub lacunarity: f32,
    pub gain: f32,
}

impl FbmAvx {
    pub fn new(source: SimplexAvx, octaves: usize, lacunarity: f32, gain: f32) -> FbmAvx {
        FbmAvx {
            source,
            octaves,
            lacunarity,
            gain,
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn eval_vec_impl(&self, pos: &VecPosition, dd: f32) -> VecSample {
        let mut freq = 1.0_f32;
        let mut amp = 1.0_f32;

        let px = _mm256_loadu_ps(pos.x.as_ptr());
        let py = _mm256_loadu_ps(pos.y.as_ptr());
        let pz = _mm256_loadu_ps(pos.z.as_ptr());

        let mut out_v = _mm256_setzero_ps();
        let mut out_dx = _mm256_setzero_ps();
        let mut out_dy = _mm256_setzero_ps();
        let mut out_dz = _mm256_setzero_ps();

        for _ in 0..self.octaves {
            let freq_v = _mm256_set1_ps(freq);
            
            let scaled_x = _mm256_mul_ps(px, freq_v);
            let scaled_y = _mm256_mul_ps(py, freq_v);
            let scaled_z = _mm256_mul_ps(pz, freq_v);

            let detail_size = 1.0_f32 / freq;
            let fade_scalar = smoothstep(0.0_f32, 2.0_f32 * dd, detail_size);
            
            if fade_scalar <= 0.0_f32 {
                break;
            }

            let effective_amp = amp * fade_scalar;
            let eff_amp_v = _mm256_set1_ps(effective_amp);
            let freq_eff_amp_v = _mm256_set1_ps(freq * effective_amp);

            let (sample_v, sample_dx, sample_dy, sample_dz) = self.source.eval_m256(scaled_x, scaled_y, scaled_z);

            out_v = _mm256_fmadd_ps(sample_v, eff_amp_v, out_v);
            out_dx = _mm256_fmadd_ps(sample_dx, freq_eff_amp_v, out_dx);
            out_dy = _mm256_fmadd_ps(sample_dy, freq_eff_amp_v, out_dy);
            out_dz = _mm256_fmadd_ps(sample_dz, freq_eff_amp_v, out_dz);

            freq *= self.lacunarity;
            amp *= self.gain;
        }

        let mut out = VecSample::default();
        _mm256_storeu_ps(out.value.as_mut_ptr(), out_v);
        _mm256_storeu_ps(out.dx.as_mut_ptr(), out_dx);
        _mm256_storeu_ps(out.dy.as_mut_ptr(), out_dy);
        _mm256_storeu_ps(out.dz.as_mut_ptr(), out_dz);

        out
    }
}

impl VecGenerator for FbmAvx {
    #[inline(always)]
    fn eval_vec(&self, pos: &VecPosition, dd: f32) -> VecSample {
        unsafe { self.eval_vec_impl(pos, dd) }
    }
}

#[inline(always)]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
