#![allow(unsafe_op_in_unsafe_fn)]
use crate::{SimplexAvx, VecGenerator, VecPosition, VecSample};
use core::arch::x86_64::*;

#[derive(Clone)]
pub struct ErosionFbmAvx {
    pub source: SimplexAvx,
    pub octaves: usize,
    pub lacunarity: f32,
    pub gain: f32,
    pub erosion_strength: f32,
}

impl ErosionFbmAvx {
    pub fn new(source: SimplexAvx, octaves: usize, lacunarity: f32, gain: f32, erosion_strength: f32) -> ErosionFbmAvx {
        ErosionFbmAvx {
            source,
            octaves,
            lacunarity,
            gain,
            erosion_strength,
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
        let mut weight_v = _mm256_set1_ps(1.0);
        let strength_v = _mm256_set1_ps(self.erosion_strength);
        let sign_mask = _mm256_set1_ps(-0.0_f32); // 0x80000000
        let one_v = _mm256_set1_ps(1.0_f32);
        let minus_two_v = _mm256_set1_ps(-2.0_f32);

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

            // Ridged math: abs_v = abs(sample_v)
            let abs_v = _mm256_andnot_ps(sign_mask, sample_v);
            // ridge = 1.0 - abs(v)
            let ridge = _mm256_sub_ps(one_v, abs_v);
            // n_v = ridge * ridge
            let n_v = _mm256_mul_ps(ridge, ridge);

            // Derivative scale: -2.0 * ridge * sign(v)
            let sign_v_bit = _mm256_and_ps(sample_v, sign_mask);
            let sign_v_float = _mm256_or_ps(one_v, sign_v_bit); // +1.0 or -1.0
            
            let d_scale_base = _mm256_mul_ps(minus_two_v, ridge);
            let d_scale = _mm256_mul_ps(d_scale_base, sign_v_float);

            let n_dx = _mm256_mul_ps(sample_dx, d_scale);
            let n_dy = _mm256_mul_ps(sample_dy, d_scale);
            let n_dz = _mm256_mul_ps(sample_dz, d_scale);

            // Accumulate
            let weighted_amp = _mm256_mul_ps(eff_amp_v, weight_v);
            let weighted_freq_amp = _mm256_mul_ps(freq_eff_amp_v, weight_v);

            // Shift n_v to [-1, 1] to avoid displacing the whole terrain upwards
            let two_v = _mm256_set1_ps(2.0_f32);
            let n_v_shifted = _mm256_sub_ps(_mm256_mul_ps(n_v, two_v), one_v);
            let n_dx_shifted = _mm256_mul_ps(n_dx, two_v);
            let n_dy_shifted = _mm256_mul_ps(n_dy, two_v);
            let n_dz_shifted = _mm256_mul_ps(n_dz, two_v);

            out_v = _mm256_fmadd_ps(n_v_shifted, weighted_amp, out_v);
            out_dx = _mm256_fmadd_ps(n_dx_shifted, weighted_freq_amp, out_dx);
            out_dy = _mm256_fmadd_ps(n_dy_shifted, weighted_freq_amp, out_dy);
            out_dz = _mm256_fmadd_ps(n_dz_shifted, weighted_freq_amp, out_dz);

            // Update weight for next octave based on the UN-SHIFTED ridge [0, 1]
            let next_w = _mm256_mul_ps(n_v, strength_v);
            weight_v = _mm256_max_ps(_mm256_setzero_ps(), _mm256_min_ps(one_v, next_w));

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

impl VecGenerator for ErosionFbmAvx {
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
