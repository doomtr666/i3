use crate::{Generator, NoisePoint, NoiseSample};
use wide::{f32x8, i32x8};

// ── Constants ─────────────────────────────────────────────────────────────────

const R3: f32 = 0.577_350_27; // sqrt(3)/3
const ORT: f32 = -0.211_324_87; // (1/sqrt(3) - 1) / 2  — rotation orthogonalizer
const RSQUARED: f32 = 0.75; // kernel radius²
const NORM: f32 = 3.2; // empirical output scale to [-1, 1]

const PX: i32 = 0x5205_402Bu32 as i32;
const PY: i32 = 0x598C_D327u32 as i32;
const PZ: i32 = 0x5BCC_226Eu32 as i32;
const HM: i32 = 0x53A3_F72Du32 as i32;
const SEED_FLIP: i32 = 0xAD2A_B84Du32 as i32;

// 24 gradient vectors from OpenSimplex2S, pre-divided by normalizer 0.27819
const ND: f32 = 0.277_192_6;
const GA: f32 = 2.224_744_87 / ND;
const GB: f32 = 1.172_151_34 / ND;
const GC: f32 = 3.086_266_47 / ND;
const GI: f32 = 1.0 / ND;
// Padded to 32 entries (next power of 2 after 24) so we can use `& 31` instead of `% 24`.
// Entries 24-31 repeat entries 0-7 — minor bias but avoids a slow modulo.
#[rustfmt::skip]
const GRAD: [[f32; 3]; 32] = [
    [ GA,  GA, -GI], [ GA,  GA,  GI], [ GC,  GB,  0.0], [ GB,  GC,  0.0],
    [-GA,  GA, -GI], [-GA,  GA,  GI], [-GB,  GC,  0.0], [-GC,  GB,  0.0],
    [-GI, -GA, -GA], [ GI, -GA, -GA], [ 0.0, -GC, -GB], [ 0.0, -GB, -GC],
    [-GI, -GA,  GA], [ GI, -GA,  GA], [ 0.0, -GB,  GC], [ 0.0, -GC,  GB],
    [-GA, -GA, -GI], [-GA, -GA,  GI], [-GC, -GB,  0.0], [-GB, -GC,  0.0],
    [-GA, -GI, -GA], [-GA,  GI, -GA], [-GB,  0.0, -GC], [-GC,  0.0, -GB],
    // repeat 0-7
    [ GA,  GA, -GI], [ GA,  GA,  GI], [ GC,  GB,  0.0], [ GB,  GC,  0.0],
    [-GA,  GA, -GI], [-GA,  GA,  GI], [-GB,  GC,  0.0], [-GC,  GB,  0.0],
];

// ── i32x8 → f32x8 helper ─────────────────────────────────────────────────────

#[inline(always)]
fn i2f(v: i32x8) -> f32x8 {
    let a: [i32; 8] = v.into();
    f32x8::from(a.map(|x| x as f32))
}

// ── Core ──────────────────────────────────────────────────────────────────────

pub struct OpenSimplex2 {
    pub seed: u32,
}

impl OpenSimplex2 {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    #[inline(always)]
    fn gradient(seed: i32x8, xp: i32x8, yp: i32x8, zp: i32x8) -> (f32x8, f32x8, f32x8) {
        let mut h = (seed ^ xp) ^ (yp ^ zp);
        h = h * i32x8::splat(HM);
        h = h ^ (h >> 18);
        let hi: [i32; 8] = h.into();
        let (mut gx, mut gy, mut gz) = ([0f32; 8], [0f32; 8], [0f32; 8]);
        for i in 0..8 {
            let g = &GRAD[(hi[i] as usize) & 31];
            gx[i] = g[0];
            gy[i] = g[1];
            gz[i] = g[2];
        }
        (f32x8::from(gx), f32x8::from(gy), f32x8::from(gz))
    }

    /// Evaluate one vertex. Returns (val, dvx, dvy, dvz), already zeroed where attn ≤ 0.
    #[inline(always)]
    fn vertex(
        seed: i32x8,
        xp: i32x8,
        yp: i32x8,
        zp: i32x8,
        dx: f32x8,
        dy: f32x8,
        dz: f32x8,
    ) -> (f32x8, f32x8, f32x8, f32x8) {
        let attn = dx.mul_add(
            -dx,
            dy.mul_add(-dy, dz.mul_add(-dz, f32x8::splat(RSQUARED))),
        );
        let active = attn.simd_gt(f32x8::ZERO);
        let (gx, gy, gz) = Self::gradient(seed, xp, yp, zp);
        let dot = dx.mul_add(gx, dy.mul_add(gy, dz * gz));
        let a2 = attn * attn;
        let a4 = a2 * a2;
        let coef = f32x8::splat(-8.0) * a2 * attn * dot;
        let z = f32x8::ZERO;
        (
            active.blend(a4 * dot, z),
            active.blend(coef.mul_add(dx, a4 * gx), z),
            active.blend(coef.mul_add(dy, a4 * gy), z),
            active.blend(coef.mul_add(dz, a4 * gz), z),
        )
    }
}

impl Generator for OpenSimplex2 {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let px = f32x8::from(point.x);
        let py = f32x8::from(point.y);
        let pz = f32x8::from(point.z);

        // Orthonormal rotation — ImproveXY orientation
        let xy = px + py;
        let s2 = xy * f32x8::splat(ORT);
        let zz = pz * f32x8::splat(R3);
        let xr = px + s2 + zz;
        let yr = py + s2 + zz;
        let zr = xy * f32x8::splat(-R3) + zz;

        // Lattice cell and fractional part
        let xr_a: [f32; 8] = xr.into();
        let yr_a: [f32; 8] = yr.into();
        let zr_a: [f32; 8] = zr.into();
        let xfl = i32x8::from(xr_a.map(|v| v.floor() as i32));
        let yfl = i32x8::from(yr_a.map(|v| v.floor() as i32));
        let zfl = i32x8::from(zr_a.map(|v| v.floor() as i32));
        let xi = xr - i2f(xfl);
        let yi = yr - i2f(yfl);
        let zi = zr - i2f(zfl);

        let s1 = i32x8::splat(self.seed as i32);
        let s2_seed = s1 ^ i32x8::splat(SEED_FLIP);
        let bx = xfl * i32x8::splat(PX);
        let by = yfl * i32x8::splat(PY);
        let bz = zfl * i32x8::splat(PZ);
        let ppx = i32x8::splat(PX);
        let ppy = i32x8::splat(PY);
        let ppz = i32x8::splat(PZ);

        // xNMask: -1 if xi < 0.5, 0 otherwise (floor(-0.5 - xi) trick)
        let half = f32x8::splat(0.5);
        let xnm = i32x8::from({
            let a: [f32; 8] = (-half - xi).into();
            a.map(|v| v as i32)
        });
        let ynm = i32x8::from({
            let a: [f32; 8] = (-half - yi).into();
            a.map(|v| v as i32)
        });
        let znm = i32x8::from({
            let a: [f32; 8] = (-half - zi).into();
            a.map(|v| v as i32)
        });

        // xNMask as float (-1.0 or 0.0) and (xNMask | 1) as float (-1.0 or 1.0)
        let xnm_f = i2f(xnm);
        let ynm_f = i2f(ynm);
        let znm_f = i2f(znm);
        let xnm1_f = i2f(xnm | i32x8::splat(1));
        let ynm1_f = i2f(ynm | i32x8::splat(1));
        let znm1_f = i2f(znm | i32x8::splat(1));

        // Two base vertex displacements
        let x0 = xi + xnm_f; // xi or xi-1
        let y0 = yi + ynm_f;
        let z0 = zi + znm_f;
        let x1 = xi - half; // xi-0.5 (second BCC copy)
        let y1 = yi - half;
        let z1 = zi - half;

        // Hashed cell primes for the two BCC lattice copies
        let hp0x = bx + (xnm & ppx);
        let hp0y = by + (ynm & ppy);
        let hp0z = bz + (znm & ppz);
        let hp1x = bx + ppx;
        let hp1y = by + ppy;
        let hp1z = bz + ppz;

        // Attenuation flip masks (see KdotJPG's derivation)
        let two = i32x8::splat(2);
        let four = i32x8::splat(4);
        let xaf0 = i2f((xnm | i32x8::splat(1)) * two) * x1;
        let yaf0 = i2f((ynm | i32x8::splat(1)) * two) * y1;
        let zaf0 = i2f((znm | i32x8::splat(1)) * two) * z1;
        let xaf1 = i2f(-two - (xnm * four)) * x1 - f32x8::splat(1.0);
        let yaf1 = i2f(-two - (ynm * four)) * y1 - f32x8::splat(1.0);
        let zaf1 = i2f(-two - (znm * four)) * z1 - f32x8::splat(1.0);

        // Base attenuations
        let a0 = x0.mul_add(
            -x0,
            y0.mul_add(-y0, z0.mul_add(-z0, f32x8::splat(RSQUARED))),
        );
        let a1 = x1.mul_add(
            -x1,
            y1.mul_add(-y1, z1.mul_add(-z1, f32x8::splat(RSQUARED))),
        );

        // Conditional attenuations for the 3 branches × 4 vertices each
        let a2 = xaf0 + a0;
        let a4 = xaf1 + a1;
        let a6 = yaf0 + a0;
        let a8 = yaf1 + a1;
        let aa = zaf0 + a0;
        let ac = zaf1 + a1;

        // Branch masks: each branch takes the "if" path (use_aX) or the "else" path (not_aX)
        let use_a2 = a2.simd_gt(f32x8::ZERO);
        let not_a2 = a2.simd_le(f32x8::ZERO); // ≡ !(a2 > 0)
        let use_a6 = a6.simd_gt(f32x8::ZERO);
        let not_a6 = a6.simd_le(f32x8::ZERO);
        let use_aa = aa.simd_gt(f32x8::ZERO);
        let not_aa = aa.simd_le(f32x8::ZERO);

        // skip5/9/D: vertex 5/9/D is suppressed when a4/a8/aC is active in its else branch.
        // no_skipX uses simd_eq(ZERO): mask is 0.0 (not skipped) or NaN (skipped, 0xFFFFFFFF).
        // Comparing NaN == 0.0 → false; comparing 0.0 == 0.0 → true. Correct mask inversion.
        let skip5 = not_a2 & a4.simd_gt(f32x8::ZERO);
        let skip9 = not_a6 & a8.simd_gt(f32x8::ZERO);
        let skipd = not_aa & ac.simd_gt(f32x8::ZERO);
        let no_skip5 = skip5.simd_eq(f32x8::ZERO);
        let no_skip9 = skip9.simd_eq(f32x8::ZERO);
        let no_skipd = skipd.simd_eq(f32x8::ZERO);

        // ── Evaluate all 14 candidate vertices ────────────────────────────────────

        // Lattice hashes for flipped coords
        let bxf = bx + (!xnm & ppx);
        let bxn = bx + (xnm & (ppx << 1));
        let byf = by + (!ynm & ppy);
        let byn = by + (ynm & (ppy << 1));
        let bzf = bz + (!znm & ppz);
        let bzn = bz + (znm & (ppz << 1));

        let (v0, d0x, d0y, d0z) = Self::vertex(s1, hp0x, hp0y, hp0z, x0, y0, z0);
        let (v1, d1x, d1y, d1z) = Self::vertex(s2_seed, hp1x, hp1y, hp1z, x1, y1, z1);
        let (va2, d2x, d2y, d2z) = Self::vertex(s1, bxf, hp0y, hp0z, x0 - xnm1_f, y0, z0);
        let (va3, d3x, d3y, d3z) = Self::vertex(s1, hp0x, byf, bzf, x0, y0 - ynm1_f, z0 - znm1_f);
        let (va4, d4x, d4y, d4z) = Self::vertex(s2_seed, bxn, hp1y, hp1z, x1 + xnm1_f, y1, z1);
        let (va6, d6x, d6y, d6z) = Self::vertex(s1, hp0x, byf, hp0z, x0, y0 - ynm1_f, z0);
        let (va7, d7x, d7y, d7z) = Self::vertex(s1, bxf, hp0y, bzf, x0 - xnm1_f, y0, z0 - znm1_f);
        let (va8, d8x, d8y, d8z) = Self::vertex(s2_seed, hp1x, byn, hp1z, x1, y1 + ynm1_f, z1);
        let (vaa, dax, day, daz) = Self::vertex(s1, hp0x, hp0y, bzf, x0, y0, z0 - znm1_f);
        let (vab, dbx, dby, dbz) = Self::vertex(s1, bxf, byf, hp0z, x0 - xnm1_f, y0 - ynm1_f, z0);
        let (vac, dcx, dcy, dcz) = Self::vertex(s2_seed, hp1x, hp1y, bzn, x1, y1, z1 + znm1_f);
        let (va5, d5x, d5y, d5z) =
            Self::vertex(s2_seed, hp1x, byn, bzn, x1, y1 + ynm1_f, z1 + znm1_f);
        let (va9, d9x, d9y, d9z) =
            Self::vertex(s2_seed, bxn, hp1y, bzn, x1 + xnm1_f, y1, z1 + znm1_f);
        let (vad, ddx, ddy, ddz) =
            Self::vertex(s2_seed, bxn, byn, hp1z, x1 + xnm1_f, y1 + ynm1_f, z1);

        // ── Sum with branch masks ─────────────────────────────────────────────────
        macro_rules! m {
            ($mask:expr, $v:expr) => {
                $mask.blend($v, f32x8::ZERO)
            };
        }

        let vsum = v0
            + v1
            + m!(use_a2, va2)
            + m!(not_a2, va3)
            + m!(not_a2, va4)
            + m!(use_a6, va6)
            + m!(not_a6, va7)
            + m!(not_a6, va8)
            + m!(use_aa, vaa)
            + m!(not_aa, vab)
            + m!(not_aa, vac)
            + m!(no_skip5, va5)
            + m!(no_skip9, va9)
            + m!(no_skipd, vad);

        let mut sx = d0x + d1x;
        let mut sy = d0y + d1y;
        let mut sz = d0z + d1z;
        macro_rules! acc {
            ($mask:expr, $dx:expr, $dy:expr, $dz:expr) => {
                sx = sx + m!($mask, $dx);
                sy = sy + m!($mask, $dy);
                sz = sz + m!($mask, $dz);
            };
        }
        acc!(use_a2, d2x, d2y, d2z);
        acc!(not_a2, d3x, d3y, d3z);
        acc!(not_a2, d4x, d4y, d4z);
        acc!(use_a6, d6x, d6y, d6z);
        acc!(not_a6, d7x, d7y, d7z);
        acc!(not_a6, d8x, d8y, d8z);
        acc!(use_aa, dax, day, daz);
        acc!(not_aa, dbx, dby, dbz);
        acc!(not_aa, dcx, dcy, dcz);
        acc!(no_skip5, d5x, d5y, d5z);
        acc!(no_skip9, d9x, d9y, d9z);
        acc!(no_skipd, ddx, ddy, ddz);

        // ── Jacobian J^T of the input rotation ────────────────────────────────────
        // xr = x + (x+y)*ORT + z*R3  ∂xr/∂x=1+ORT  ∂xr/∂y=ORT   ∂xr/∂z=R3
        // yr = y + (x+y)*ORT + z*R3  ∂yr/∂x=ORT    ∂yr/∂y=1+ORT  ∂yr/∂z=R3
        // zr = (x+y)*(-R3) + z*R3    ∂zr/∂x=-R3    ∂zr/∂y=-R3   ∂zr/∂z=R3
        let ort1 = f32x8::splat(1.0 + ORT);
        let ort0 = f32x8::splat(ORT);
        let rp = f32x8::splat(R3);
        let rn = f32x8::splat(-R3);
        let gx = ort1 * sx + ort0 * sy + rn * sz;
        let gy = ort0 * sx + ort1 * sy + rn * sz;
        let gz = rp * sx + rp * sy + rp * sz;

        let n = f32x8::splat(NORM);
        NoiseSample {
            value: (vsum * n).into(),
            dx: (gx * n).into(),
            dy: (gy * n).into(),
            dz: (gz * n).into(),
        }
    }
}
