#![allow(unsafe_op_in_unsafe_fn)]
use crate::{VecGenerator, VecPosition, VecSample};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[derive(Clone)]
pub struct SimplexAvx {
    seed: u32,
}

impl SimplexAvx {
    pub fn new(seed: u32) -> Self {
        SimplexAvx { seed }
    }
}

#[cfg(target_arch = "x86_64")]
impl SimplexAvx {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn hash_avx(&self, i: __m256i, j: __m256i, k: __m256i) -> (__m256, __m256, __m256) {
        let prime_x = _mm256_set1_epi32(0x9E3779B9_u32 as i32);
        let prime_y = _mm256_set1_epi32(0x85EBCA6B_u32 as i32);
        let prime_z = _mm256_set1_epi32(0xC2B2AE35_u32 as i32);

        let mut h = _mm256_set1_epi32(self.seed as i32);

        let ix = _mm256_mullo_epi32(i, prime_x);
        let jy = _mm256_mullo_epi32(j, prime_y);
        let kz = _mm256_mullo_epi32(k, prime_z);

        h = _mm256_add_epi32(h, ix);
        h = _mm256_add_epi32(h, jy);
        h = _mm256_add_epi32(h, kz);

        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));
        h = _mm256_mullo_epi32(h, prime_y);
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 13));
        h = _mm256_mullo_epi32(h, prime_z);
        h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 16));

        let twelve = _mm256_set1_epi64x(12);
        let h_even = _mm256_and_si256(h, _mm256_set1_epi64x(0xFFFFFFFF_u32 as i64));
        let h_odd = _mm256_srli_epi64(h, 32);

        let mul_even = _mm256_mul_epu32(h_even, twelve);
        let mul_odd = _mm256_mul_epu32(h_odd, twelve);

        let res_even = _mm256_srli_epi64(mul_even, 32);
        let res_odd = _mm256_srli_epi64(mul_odd, 32);
        let res_odd_shifted = _mm256_slli_epi64(res_odd, 32);

        let idx = _mm256_or_si256(res_even, res_odd_shifted);

        let b0 = _mm256_slli_epi32(_mm256_and_si256(idx, _mm256_set1_epi32(1)), 31);
        let b1 = _mm256_slli_epi32(_mm256_and_si256(idx, _mm256_set1_epi32(2)), 30);

        let cat = _mm256_srli_epi32(idx, 2);

        let cat_eq_0 = _mm256_cmpeq_epi32(cat, _mm256_setzero_si256());
        let cat_eq_1 = _mm256_cmpeq_epi32(cat, _mm256_set1_epi32(1));
        let cat_eq_2 = _mm256_cmpeq_epi32(cat, _mm256_set1_epi32(2));

        let float_one = _mm256_set1_epi32(0x3F800000);

        let x_nonzero = _mm256_andnot_si256(cat_eq_2, float_one);
        let gx = _mm256_castsi256_ps(_mm256_or_si256(x_nonzero, b0));

        let y_nonzero = _mm256_andnot_si256(cat_eq_1, float_one);
        let y_sign = _mm256_blendv_epi8(b1, b0, cat_eq_2);
        let gy = _mm256_castsi256_ps(_mm256_or_si256(y_nonzero, y_sign));

        let z_nonzero = _mm256_andnot_si256(cat_eq_0, float_one);
        let gz = _mm256_castsi256_ps(_mm256_or_si256(z_nonzero, b1));

        (gx, gy, gz)
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) unsafe fn eval_m256(&self, px: __m256, py: __m256, pz: __m256) -> (__m256, __m256, __m256, __m256) {
        let f3 = _mm256_set1_ps(1.0_f32 / 3.0_f32);
        let g3 = _mm256_set1_ps(1.0_f32 / 6.0_f32);

        let p_sum = _mm256_add_ps(px, _mm256_add_ps(py, pz));
        let s = _mm256_mul_ps(p_sum, f3);

        let i = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_add_ps(px, s)));
        let j = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_add_ps(py, s)));
        let k = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_add_ps(pz, s)));

        let ijk_sum = _mm256_cvtepi32_ps(_mm256_add_epi32(i, _mm256_add_epi32(j, k)));
        let t_unskew = _mm256_mul_ps(ijk_sum, g3);

        let x0 = _mm256_sub_ps(px, _mm256_sub_ps(_mm256_cvtepi32_ps(i), t_unskew));
        let y0 = _mm256_sub_ps(py, _mm256_sub_ps(_mm256_cvtepi32_ps(j), t_unskew));
        let z0 = _mm256_sub_ps(pz, _mm256_sub_ps(_mm256_cvtepi32_ps(k), t_unskew));

        let mask_x0_ge_y0 = _mm256_castps_si256(_mm256_cmp_ps(x0, y0, _CMP_GE_OQ));
        let mask_y0_ge_z0 = _mm256_castps_si256(_mm256_cmp_ps(y0, z0, _CMP_GE_OQ));
        let mask_x0_ge_z0 = _mm256_castps_si256(_mm256_cmp_ps(x0, z0, _CMP_GE_OQ));

        let mask_y0_lt_z0 = _mm256_xor_si256(mask_y0_ge_z0, _mm256_set1_epi32(-1));
        let mask_x0_lt_y0 = _mm256_xor_si256(mask_x0_ge_y0, _mm256_set1_epi32(-1));
        let mask_x0_lt_z0 = _mm256_xor_si256(mask_x0_ge_z0, _mm256_set1_epi32(-1));

        let b1 = _mm256_and_si256(mask_x0_ge_y0, mask_y0_ge_z0);
        let b2 = _mm256_and_si256(
            _mm256_and_si256(mask_x0_ge_y0, mask_y0_lt_z0),
            mask_x0_ge_z0,
        );
        let b3 = _mm256_and_si256(
            _mm256_and_si256(mask_x0_ge_y0, mask_y0_lt_z0),
            mask_x0_lt_z0,
        );
        let b4 = _mm256_and_si256(mask_x0_lt_y0, mask_y0_lt_z0);
        let b5 = _mm256_and_si256(
            _mm256_and_si256(mask_x0_lt_y0, mask_y0_ge_z0),
            mask_x0_lt_z0,
        );
        let b6 = _mm256_and_si256(
            _mm256_and_si256(mask_x0_lt_y0, mask_y0_ge_z0),
            mask_x0_ge_z0,
        );

        let i1 = _mm256_or_si256(b1, b2);
        let j1 = _mm256_or_si256(b5, b6);
        let k1 = _mm256_or_si256(b3, b4);

        let i2 = _mm256_or_si256(_mm256_or_si256(b1, b2), _mm256_or_si256(b3, b6));
        let j2 = _mm256_or_si256(_mm256_or_si256(b1, b4), _mm256_or_si256(b5, b6));
        let k2 = _mm256_or_si256(_mm256_or_si256(b2, b3), _mm256_or_si256(b4, b5));

        let one_i = _mm256_set1_epi32(1);
        let i1 = _mm256_and_si256(i1, one_i);
        let j1 = _mm256_and_si256(j1, one_i);
        let k1 = _mm256_and_si256(k1, one_i);
        let i2 = _mm256_and_si256(i2, one_i);
        let j2 = _mm256_and_si256(j2, one_i);
        let k2 = _mm256_and_si256(k2, one_i);

        let i1f = _mm256_cvtepi32_ps(i1);
        let j1f = _mm256_cvtepi32_ps(j1);
        let k1f = _mm256_cvtepi32_ps(k1);

        let i2f = _mm256_cvtepi32_ps(i2);
        let j2f = _mm256_cvtepi32_ps(j2);
        let k2f = _mm256_cvtepi32_ps(k2);

        let g3_vec = _mm256_set1_ps(1.0_f32 / 6.0_f32);
        let two_g3 = _mm256_set1_ps(2.0_f32 / 6.0_f32);
        let three_g3 = _mm256_set1_ps(3.0_f32 / 6.0_f32);
        let one_f = _mm256_set1_ps(1.0_f32);

        let d1x = _mm256_add_ps(_mm256_sub_ps(x0, i1f), g3_vec);
        let d1y = _mm256_add_ps(_mm256_sub_ps(y0, j1f), g3_vec);
        let d1z = _mm256_add_ps(_mm256_sub_ps(z0, k1f), g3_vec);

        let d2x = _mm256_add_ps(_mm256_sub_ps(x0, i2f), two_g3);
        let d2y = _mm256_add_ps(_mm256_sub_ps(y0, j2f), two_g3);
        let d2z = _mm256_add_ps(_mm256_sub_ps(z0, k2f), two_g3);

        let d3x = _mm256_add_ps(_mm256_sub_ps(x0, one_f), three_g3);
        let d3y = _mm256_add_ps(_mm256_sub_ps(y0, one_f), three_g3);
        let d3z = _mm256_add_ps(_mm256_sub_ps(z0, one_f), three_g3);

        let (g0x, g0y, g0z) = self.hash_avx(i, j, k);
        let (g1x, g1y, g1z) = self.hash_avx(
            _mm256_add_epi32(i, i1),
            _mm256_add_epi32(j, j1),
            _mm256_add_epi32(k, k1),
        );
        let (g2x, g2y, g2z) = self.hash_avx(
            _mm256_add_epi32(i, i2),
            _mm256_add_epi32(j, j2),
            _mm256_add_epi32(k, k2),
        );
        let (g3x, g3y, g3z) = self.hash_avx(
            _mm256_add_epi32(i, one_i),
            _mm256_add_epi32(j, one_i),
            _mm256_add_epi32(k, one_i),
        );

        let mut v = _mm256_setzero_ps();
        let mut dx = _mm256_setzero_ps();
        let mut dy = _mm256_setzero_ps();
        let mut dz = _mm256_setzero_ps();

        let mut add_contribution = |dx_v: __m256,
                                    dy_v: __m256,
                                    dz_v: __m256,
                                    gx_v: __m256,
                                    gy_v: __m256,
                                    gz_v: __m256| {
            unsafe {
                let mut t_val = _mm256_set1_ps(0.6_f32);
                t_val = _mm256_fnmadd_ps(dx_v, dx_v, t_val);
                t_val = _mm256_fnmadd_ps(dy_v, dy_v, t_val);
                t_val = _mm256_fnmadd_ps(dz_v, dz_v, t_val);

                let mask = _mm256_cmp_ps(t_val, _mm256_setzero_ps(), _CMP_GT_OQ);
                let mask_cast = _mm256_castps_si256(mask);

                let t2 = _mm256_mul_ps(t_val, t_val);
                let t3 = _mm256_mul_ps(t2, t_val);
                let t4 = _mm256_mul_ps(t2, t2);

                let mut dot_val = _mm256_mul_ps(dx_v, gx_v);
                dot_val = _mm256_fmadd_ps(dy_v, gy_v, dot_val);
                dot_val = _mm256_fmadd_ps(dz_v, gz_v, dot_val);

                let v_contrib = _mm256_mul_ps(t4, dot_val);
                v = _mm256_add_ps(v, _mm256_and_ps(v_contrib, mask));

                let temp = _mm256_mul_ps(_mm256_mul_ps(t3, dot_val), _mm256_set1_ps(-8.0_f32));

                let dx_contrib = _mm256_fmadd_ps(temp, dx_v, _mm256_mul_ps(t4, gx_v));
                dx = _mm256_add_ps(dx, _mm256_and_ps(dx_contrib, mask));

                let dy_contrib = _mm256_fmadd_ps(temp, dy_v, _mm256_mul_ps(t4, gy_v));
                dy = _mm256_add_ps(dy, _mm256_and_ps(dy_contrib, mask));

                let dz_contrib = _mm256_fmadd_ps(temp, dz_v, _mm256_mul_ps(t4, gz_v));
                dz = _mm256_add_ps(dz, _mm256_and_ps(dz_contrib, mask));
            }
        };

        add_contribution(x0, y0, z0, g0x, g0y, g0z);
        add_contribution(d1x, d1y, d1z, g1x, g1y, g1z);
        add_contribution(d2x, d2y, d2z, g2x, g2y, g2z);
        add_contribution(d3x, d3y, d3z, g3x, g3y, g3z);

        let thirty_two = _mm256_set1_ps(32.0_f32);
        v = _mm256_mul_ps(v, thirty_two);
        dx = _mm256_mul_ps(dx, thirty_two);
        dy = _mm256_mul_ps(dy, thirty_two);
        dz = _mm256_mul_ps(dz, thirty_two);

        (v, dx, dy, dz)
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn eval_vec_impl(&self, pos: &VecPosition) -> VecSample {
        let px = _mm256_loadu_ps(pos.x.as_ptr());
        let py = _mm256_loadu_ps(pos.y.as_ptr());
        let pz = _mm256_loadu_ps(pos.z.as_ptr());

        let (v, dx, dy, dz) = self.eval_m256(px, py, pz);

        let mut out = VecSample::default();
        _mm256_storeu_ps(out.value.as_mut_ptr(), v);
        _mm256_storeu_ps(out.dx.as_mut_ptr(), dx);
        _mm256_storeu_ps(out.dy.as_mut_ptr(), dy);
        _mm256_storeu_ps(out.dz.as_mut_ptr(), dz);

        out
    }
}

#[cfg(target_arch = "x86_64")]
impl VecGenerator for SimplexAvx {
    #[inline(always)]
    fn eval_vec(&self, pos: &VecPosition, _dd: f32) -> VecSample {
        unsafe { self.eval_vec_impl(pos) }
    }
}
