use crate::{Generator, NoiseSample, NoisePoint};
use wide::f32x8;

// ── DomainWarpFractal ─────────────────────────────────────────────────────────
// Multi-octave cumulative domain warp: each octave displaces the domain using
// the ALREADY-WARPED position from all previous octaves.
// Contrast with DomainWarp(Fbm(...)): there, every warp octave sees the same
// original position.  Here each octave compounds on the previous, producing
// much more complex, organic patterns.
//
// At each octave:
//   p_current += intensity · amp · (Gx(p_current·freq), Gy(p_current·freq), Gz(p_current·freq))
//
// The Jacobian of the full warp chain is propagated for the footprint (LOD) and
// for the output gradient.  For octave i the per-step Jacobian is:
//   J_step_i ≈ I + intensity·amp·[[∂Wx/∂x …], [∂Wy/∂x …], [∂Wz/∂x …]]
// and the accumulated Jacobian J = J_k · … · J_1 satisfies:
//   J_new = J_step · J_old
//
// Storing a full 3×3 SIMD Jacobian (9 f32x8 values) gives the exact gradient.

pub struct DomainWarpFractal<G, Gx, Gy, Gz> {
    pub inner:     G,
    pub gen_x:     Gx,
    pub gen_y:     Gy,
    pub gen_z:     Gz,
    pub octaves:   usize,
    pub lacunarity: f32,
    pub gain:      f32,
    pub intensity: f32,
}

impl<G, Gx, Gy, Gz> DomainWarpFractal<G, Gx, Gy, Gz>
where G: Generator, Gx: Generator, Gy: Generator, Gz: Generator
{
    pub fn new(inner: G, gen_x: Gx, gen_y: Gy, gen_z: Gz,
               octaves: usize, lacunarity: f32, gain: f32, intensity: f32) -> Self {
        Self { inner, gen_x, gen_y, gen_z, octaves, lacunarity, gain, intensity }
    }
}

impl<G, Gx, Gy, Gz> Generator for DomainWarpFractal<G, Gx, Gy, Gz>
where G: Generator, Gx: Generator, Gy: Generator, Gz: Generator
{
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let mut cx  = f32x8::from(point.x);
        let mut cy  = f32x8::from(point.y);
        let mut cz  = f32x8::from(point.z);
        let mut cdx = f32x8::from(point.dx);
        let mut cdy = f32x8::from(point.dy);
        let mut cdz = f32x8::from(point.dz);

        // Accumulated Jacobian J (row-major 3×3, starts as identity)
        let mut j00 = f32x8::splat(1.0); let mut j01 = f32x8::ZERO; let mut j02 = f32x8::ZERO;
        let mut j10 = f32x8::ZERO; let mut j11 = f32x8::splat(1.0); let mut j12 = f32x8::ZERO;
        let mut j20 = f32x8::ZERO; let mut j21 = f32x8::ZERO; let mut j22 = f32x8::splat(1.0);

        let mut amp = 1.0f32;
        let mut freq = 1.0f32;

        for _ in 0..self.octaves {
            let fv = f32x8::splat(freq);
            let iv = f32x8::splat(self.intensity * amp);

            let warp_pos = NoisePoint {
                x:  (cx  * fv).into(), y:  (cy  * fv).into(), z:  (cz  * fv).into(),
                dx: (cdx * fv).into(), dy: (cdy * fv).into(), dz: (cdz * fv).into(),
            };

            let sx = self.gen_x.sample(&warp_pos);
            let sy = self.gen_y.sample(&warp_pos);
            let sz = self.gen_z.sample(&warp_pos);

            // Displace position
            cx = iv.mul_add(f32x8::from(sx.value), cx);
            cy = iv.mul_add(f32x8::from(sy.value), cy);
            cz = iv.mul_add(f32x8::from(sz.value), cz);

            // Displace footprint (forward Jacobian of this warp step on the footprint)
            // dω' = d + intensity·amp · dot(∇W_i, d)  — same as single DomainWarp
            let freq_iv = iv * fv;
            let dot_dx = f32x8::from(sx.dx).mul_add(cdx, f32x8::from(sx.dy).mul_add(cdy, f32x8::from(sx.dz) * cdz));
            let dot_dy = f32x8::from(sy.dx).mul_add(cdx, f32x8::from(sy.dy).mul_add(cdy, f32x8::from(sy.dz) * cdz));
            let dot_dz = f32x8::from(sz.dx).mul_add(cdx, f32x8::from(sz.dy).mul_add(cdy, f32x8::from(sz.dz) * cdz));
            cdx = freq_iv.mul_add(dot_dx, cdx);
            cdy = freq_iv.mul_add(dot_dy, cdy);
            cdz = freq_iv.mul_add(dot_dz, cdz);

            // Update accumulated Jacobian: J_new = J_step · J_old
            // J_step = I + freq_iv · [[sx.dx, sx.dy, sx.dz], [sy.dx, …], [sz.dx, …]]
            let wdx = f32x8::from(sx.dx) * freq_iv; let wdy = f32x8::from(sx.dy) * freq_iv; let wdz = f32x8::from(sx.dz) * freq_iv;
            let xdx = f32x8::from(sy.dx) * freq_iv; let xdy = f32x8::from(sy.dy) * freq_iv; let xdz = f32x8::from(sy.dz) * freq_iv;
            let ydx = f32x8::from(sz.dx) * freq_iv; let ydy = f32x8::from(sz.dy) * freq_iv; let ydz = f32x8::from(sz.dz) * freq_iv;

            // New row 0: (I + W_grad) · J_old where W_grad[0] = (wdx, wdy, wdz)
            let n00 = j00 + wdx.mul_add(j00, wdy.mul_add(j10, wdz * j20));
            let n01 = j01 + wdx.mul_add(j01, wdy.mul_add(j11, wdz * j21));
            let n02 = j02 + wdx.mul_add(j02, wdy.mul_add(j12, wdz * j22));
            let n10 = j10 + xdx.mul_add(j00, xdy.mul_add(j10, xdz * j20));
            let n11 = j11 + xdx.mul_add(j01, xdy.mul_add(j11, xdz * j21));
            let n12 = j12 + xdx.mul_add(j02, xdy.mul_add(j12, xdz * j22));
            let n20 = j20 + ydx.mul_add(j00, ydy.mul_add(j10, ydz * j20));
            let n21 = j21 + ydx.mul_add(j01, ydy.mul_add(j11, ydz * j21));
            let n22 = j22 + ydx.mul_add(j02, ydy.mul_add(j12, ydz * j22));
            j00=n00; j01=n01; j02=n02;
            j10=n10; j11=n11; j12=n12;
            j20=n20; j21=n21; j22=n22;

            amp  *= self.gain;
            freq *= self.lacunarity;
        }

        // Evaluate inner generator at the final warped position
        let final_pos = NoisePoint {
            x: cx.into(), y: cy.into(), z: cz.into(),
            dx: cdx.into(), dy: cdy.into(), dz: cdz.into(),
        };
        let s = self.inner.sample(&final_pos);
        let sdx = f32x8::from(s.dx);
        let sdy = f32x8::from(s.dy);
        let sdz = f32x8::from(s.dz);

        // grad_world = J^T · grad_warped   (J = ∂p_final/∂p_input)
        let gx = j00 * sdx + j10 * sdy + j20 * sdz;
        let gy = j01 * sdx + j11 * sdy + j21 * sdz;
        let gz = j02 * sdx + j12 * sdy + j22 * sdz;

        NoiseSample {
            value: s.value,
            dx: gx.into(),
            dy: gy.into(),
            dz: gz.into(),
        }
    }
}
