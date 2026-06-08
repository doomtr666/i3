use crate::{Generator, NoisePoint, NoiseSample};
use std::f32::consts::TAU;
use wide::f32x8;

// ── Constant ─────────────────────────────────────────────────────────────────

pub struct Const {
    pub value: f32,
}

impl Const {
    pub fn new(value: f32) -> Self {
        Self { value }
    }
}

impl Generator for Const {
    fn sample(&self, _: &NoisePoint) -> NoiseSample {
        NoiseSample {
            value: [self.value; 8],
            dx: [0.0; 8],
            dy: [0.0; 8],
            dz: [0.0; 8],
        }
    }
}

// ── Add ──────────────────────────────────────────────────────────────────────

pub struct Add<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> Add<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Generator, B: Generator> Generator for Add<A, B> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let a = self.a.sample(point);
        let b = self.b.sample(point);
        NoiseSample {
            value: (f32x8::from(a.value) + f32x8::from(b.value)).into(),
            dx: (f32x8::from(a.dx) + f32x8::from(b.dx)).into(),
            dy: (f32x8::from(a.dy) + f32x8::from(b.dy)).into(),
            dz: (f32x8::from(a.dz) + f32x8::from(b.dz)).into(),
        }
    }
}

// ── Sub ──────────────────────────────────────────────────────────────────────

pub struct Sub<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> Sub<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Generator, B: Generator> Generator for Sub<A, B> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let a = self.a.sample(point);
        let b = self.b.sample(point);
        NoiseSample {
            value: (f32x8::from(a.value) - f32x8::from(b.value)).into(),
            dx: (f32x8::from(a.dx) - f32x8::from(b.dx)).into(),
            dy: (f32x8::from(a.dy) - f32x8::from(b.dy)).into(),
            dz: (f32x8::from(a.dz) - f32x8::from(b.dz)).into(),
        }
    }
}

// ── Mul ──────────────────────────────────────────────────────────────────────
// ∂(a·b)/∂x = a·∂b/∂x + b·∂a/∂x

pub struct Mul<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> Mul<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Generator, B: Generator> Generator for Mul<A, B> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let a = self.a.sample(point);
        let b = self.b.sample(point);
        let av = f32x8::from(a.value);
        let bv = f32x8::from(b.value);
        NoiseSample {
            value: (av * bv).into(),
            dx: av.mul_add(f32x8::from(b.dx), bv * f32x8::from(a.dx)).into(),
            dy: av.mul_add(f32x8::from(b.dy), bv * f32x8::from(a.dy)).into(),
            dz: av.mul_add(f32x8::from(b.dz), bv * f32x8::from(a.dz)).into(),
        }
    }
}

// ── Neg ──────────────────────────────────────────────────────────────────────

pub struct Neg<G> {
    pub source: G,
}

impl<G> Neg<G> {
    pub fn new(source: G) -> Self {
        Self { source }
    }
}

impl<G: Generator> Generator for Neg<G> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let s = self.source.sample(point);
        NoiseSample {
            value: (-f32x8::from(s.value)).into(),
            dx: (-f32x8::from(s.dx)).into(),
            dy: (-f32x8::from(s.dy)).into(),
            dz: (-f32x8::from(s.dz)).into(),
        }
    }
}

// ── ScaleBias ─────────────────────────────────────────────────────────────────
// value * scale + bias  —  gradient * scale (bias is constant, no gradient)

pub struct ScaleBias<G> {
    pub source: G,
    pub scale: f32,
    pub bias: f32,
}

impl<G> ScaleBias<G> {
    pub fn new(source: G, scale: f32, bias: f32) -> Self {
        Self {
            source,
            scale,
            bias,
        }
    }
}

impl<G: Generator> Generator for ScaleBias<G> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let s = self.source.sample(point);
        let sc = f32x8::splat(self.scale);
        let bi = f32x8::splat(self.bias);
        NoiseSample {
            value: f32x8::from(s.value).mul_add(sc, bi).into(),
            dx: (f32x8::from(s.dx) * sc).into(),
            dy: (f32x8::from(s.dy) * sc).into(),
            dz: (f32x8::from(s.dz) * sc).into(),
        }
    }
}

// ── Abs ───────────────────────────────────────────────────────────────────────
// ∂|f|/∂x = sign(f) · ∂f/∂x

pub struct Abs<G> {
    pub source: G,
}

impl<G> Abs<G> {
    pub fn new(source: G) -> Self {
        Self { source }
    }
}

impl<G: Generator> Generator for Abs<G> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let s = self.source.sample(point);
        let v = f32x8::from(s.value);
        let sign = v.signum();
        NoiseSample {
            value: v.abs().into(),
            dx: (sign * f32x8::from(s.dx)).into(),
            dy: (sign * f32x8::from(s.dy)).into(),
            dz: (sign * f32x8::from(s.dz)).into(),
        }
    }
}

// ── Min ───────────────────────────────────────────────────────────────────────
// Gradient of the winning (smaller) generator

pub struct Min<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> Min<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Generator, B: Generator> Generator for Min<A, B> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let a = self.a.sample(point);
        let b = self.b.sample(point);
        let av = f32x8::from(a.value);
        let bv = f32x8::from(b.value);
        let a_wins = av.simd_le(bv);
        NoiseSample {
            value: av.min(bv).into(),
            dx: a_wins.blend(f32x8::from(a.dx), f32x8::from(b.dx)).into(),
            dy: a_wins.blend(f32x8::from(a.dy), f32x8::from(b.dy)).into(),
            dz: a_wins.blend(f32x8::from(a.dz), f32x8::from(b.dz)).into(),
        }
    }
}

// ── Max ───────────────────────────────────────────────────────────────────────

pub struct Max<A, B> {
    pub a: A,
    pub b: B,
}

impl<A, B> Max<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Generator, B: Generator> Generator for Max<A, B> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let a = self.a.sample(point);
        let b = self.b.sample(point);
        let av = f32x8::from(a.value);
        let bv = f32x8::from(b.value);
        let a_wins = av.simd_ge(bv);
        NoiseSample {
            value: av.max(bv).into(),
            dx: a_wins.blend(f32x8::from(a.dx), f32x8::from(b.dx)).into(),
            dy: a_wins.blend(f32x8::from(a.dy), f32x8::from(b.dy)).into(),
            dz: a_wins.blend(f32x8::from(a.dz), f32x8::from(b.dz)).into(),
        }
    }
}

// ── Blend ─────────────────────────────────────────────────────────────────────
// value = a + m*(b - a)  with m = clamp(mask, 0, 1)
// ∂/∂x  = (1-m)·∂a/∂x + m·∂b/∂x + (b-a)·∂m/∂x  (Leibniz)

pub struct Blend<A, B, M> {
    pub a: A,
    pub b: B,
    pub mask: M,
}

impl<A, B, M> Blend<A, B, M> {
    pub fn new(a: A, b: B, mask: M) -> Self {
        Self { a, b, mask }
    }
}

impl<A: Generator, B: Generator, M: Generator> Generator for Blend<A, B, M> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let a = self.a.sample(point);
        let b = self.b.sample(point);
        let m = self.mask.sample(point);
        let av = f32x8::from(a.value);
        let bv = f32x8::from(b.value);
        let mv_raw = f32x8::from(m.value);
        let mv = mv_raw.max(f32x8::ZERO).min(f32x8::splat(1.0));
        let one_m = f32x8::splat(1.0) - mv;
        let diff = bv - av;
        // Mask gradient is zero when clamped (mask is constant outside [0, 1])
        let mask_active = mv_raw.simd_gt(f32x8::ZERO) & mv_raw.simd_lt(f32x8::splat(1.0));
        let z = f32x8::ZERO;
        let mdx = mask_active.blend(f32x8::from(m.dx), z);
        let mdy = mask_active.blend(f32x8::from(m.dy), z);
        let mdz = mask_active.blend(f32x8::from(m.dz), z);
        NoiseSample {
            value: mv.mul_add(diff, av).into(),
            // (1-m)*da + m*db + (b-a)*dm
            dx: one_m
                .mul_add(f32x8::from(a.dx), mv.mul_add(f32x8::from(b.dx), diff * mdx))
                .into(),
            dy: one_m
                .mul_add(f32x8::from(a.dy), mv.mul_add(f32x8::from(b.dy), diff * mdy))
                .into(),
            dz: one_m
                .mul_add(f32x8::from(a.dz), mv.mul_add(f32x8::from(b.dz), diff * mdz))
                .into(),
        }
    }
}

// ── Terrace ───────────────────────────────────────────────────────────────────
// f(v)  = v - sin(2π·freq·v) / (2π·freq)
// f'(v) = 1 - cos(2π·freq·v)   ∈ [0, 2]
// Flat at terrace centers (cos=1), steep at transitions (cos=-1)

pub struct Terrace<G> {
    pub source: G,
    pub frequency: f32,
}

impl<G> Terrace<G> {
    pub fn new(source: G, frequency: f32) -> Self {
        Self { source, frequency }
    }
}

impl<G: Generator> Generator for Terrace<G> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let s = self.source.sample(point);
        let v = f32x8::from(s.value);
        let tau_k = f32x8::splat(TAU * self.frequency);
        let angle = tau_k * v;
        let (sin_a, cos_a) = angle.sin_cos();
        let value = v - sin_a / tau_k;
        let deriv = f32x8::splat(1.0) - cos_a;
        NoiseSample {
            value: value.into(),
            dx: (deriv * f32x8::from(s.dx)).into(),
            dy: (deriv * f32x8::from(s.dy)).into(),
            dz: (deriv * f32x8::from(s.dz)).into(),
        }
    }
}

// ── Clamp ─────────────────────────────────────────────────────────────────────
// Gradient is zero outside [lo, hi)

pub struct Clamp<G> {
    pub source: G,
    pub lo: f32,
    pub hi: f32,
}

impl<G> Clamp<G> {
    pub fn new(source: G, lo: f32, hi: f32) -> Self {
        Self { source, lo, hi }
    }
}

impl<G: Generator> Generator for Clamp<G> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let s = self.source.sample(point);
        let v = f32x8::from(s.value);
        let lo = f32x8::splat(self.lo);
        let hi = f32x8::splat(self.hi);
        let active = v.simd_ge(lo) & v.simd_lt(hi);
        let z = f32x8::ZERO;
        NoiseSample {
            value: v.max(lo).min(hi).into(),
            dx: active.blend(f32x8::from(s.dx), z).into(),
            dy: active.blend(f32x8::from(s.dy), z).into(),
            dz: active.blend(f32x8::from(s.dz), z).into(),
        }
    }
}

// ── Exponent ──────────────────────────────────────────────────────────────────
// f(v)  = sign(v) · |v|^n   (preserves sign, contracts/expands the range)
// f'(v) = n · |v|^(n-1)

pub struct Exponent<G> {
    pub source: G,
    pub exponent: f32,
}

impl<G> Exponent<G> {
    pub fn new(source: G, exponent: f32) -> Self {
        Self { source, exponent }
    }
}

impl<G: Generator> Generator for Exponent<G> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let s = self.source.sample(point);
        let v = f32x8::from(s.value);
        let e = self.exponent;
        let abs_v = v.abs();
        let powered = abs_v.powf(e);
        let deriv = f32x8::splat(e) * abs_v.powf(e - 1.0);
        NoiseSample {
            value: (v.signum() * powered).into(),
            dx: (deriv * f32x8::from(s.dx)).into(),
            dy: (deriv * f32x8::from(s.dy)).into(),
            dz: (deriv * f32x8::from(s.dz)).into(),
        }
    }
}

// ── Power ─────────────────────────────────────────────────────────────────────
// f(a, b)  = sign(a) · |a|^b
// ∂f/∂a   = b · sign(a) · |a|^(b-1)
// ∂f/∂b   = sign(a) · |a|^b · ln|a|

pub struct Power<A, B> {
    pub base: A,
    pub exp: B,
}

impl<A, B> Power<A, B> {
    pub fn new(base: A, exp: B) -> Self {
        Self { base, exp }
    }
}

impl<A: Generator, B: Generator> Generator for Power<A, B> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let a = self.base.sample(point);
        let b = self.exp.sample(point);
        let av = f32x8::from(a.value);
        let bv = f32x8::from(b.value);
        let sign_a = av.signum();
        let abs_a = av.abs();
        // Clamp away from zero to avoid ln(0); powered → 0 naturally for positive b.
        let safe_abs = abs_a.max(f32x8::splat(f32::MIN_POSITIVE));
        let ln_abs = safe_abs.ln();
        let powered = (bv * ln_abs).exp();
        // ∂/∂a = b · sign(a) · |a|^(b-1) = b · sign(a) · powered / safe_abs
        let da = bv * sign_a * powered / safe_abs;
        // ∂/∂b = sign(a) · |a|^b · ln|a|
        let db = sign_a * powered * ln_abs;
        NoiseSample {
            value: (sign_a * powered).into(),
            dx: da.mul_add(f32x8::from(a.dx), db * f32x8::from(b.dx)).into(),
            dy: da.mul_add(f32x8::from(a.dy), db * f32x8::from(b.dy)).into(),
            dz: da.mul_add(f32x8::from(a.dz), db * f32x8::from(b.dz)).into(),
        }
    }
}

// ── Select ─────────────────────────────────────────────────────────────────────
// A when control < threshold, B when control > threshold.
// Smooth cubic transition over [threshold - falloff, threshold + falloff].

pub struct Select<A, B, C> {
    pub a: A,
    pub b: B,
    pub control: C,
    pub threshold: f32,
    pub falloff: f32,
}

impl<A, B, C> Select<A, B, C> {
    pub fn new(a: A, b: B, control: C, threshold: f32, falloff: f32) -> Self {
        Self {
            a,
            b,
            control,
            threshold,
            falloff,
        }
    }
}

impl<A: Generator, B: Generator, C: Generator> Generator for Select<A, B, C> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let a = self.a.sample(point);
        let b = self.b.sample(point);
        let c = self.control.sample(point);
        let av = f32x8::from(a.value);
        let bv = f32x8::from(b.value);
        let cv = f32x8::from(c.value);
        let diff = bv - av;

        let (mv, d_mask_dc) = if self.falloff > 0.0 {
            let lo = f32x8::splat(self.threshold - self.falloff);
            let hi = f32x8::splat(self.threshold + self.falloff);
            let t = ((cv - lo) / f32x8::splat(2.0 * self.falloff))
                .max(f32x8::ZERO)
                .min(f32x8::splat(1.0));
            // smoothstep: 3t² - 2t³, derivative: 6t(1-t) / (2*falloff)
            let m = t * t * (f32x8::splat(3.0) - f32x8::splat(2.0) * t);
            let dm_dc =
                f32x8::splat(6.0) * t * (f32x8::splat(1.0) - t) * f32x8::splat(0.5 / self.falloff);
            let in_range = cv.simd_gt(lo) & cv.simd_lt(hi);
            (m, in_range.blend(dm_dc, f32x8::ZERO))
        } else {
            let m = cv
                .simd_gt(f32x8::splat(self.threshold))
                .blend(f32x8::splat(1.0), f32x8::ZERO);
            (m, f32x8::ZERO)
        };

        let one_m = f32x8::splat(1.0) - mv;
        NoiseSample {
            value: mv.mul_add(diff, av).into(),
            dx: one_m
                .mul_add(
                    f32x8::from(a.dx),
                    mv.mul_add(f32x8::from(b.dx), diff * d_mask_dc * f32x8::from(c.dx)),
                )
                .into(),
            dy: one_m
                .mul_add(
                    f32x8::from(a.dy),
                    mv.mul_add(f32x8::from(b.dy), diff * d_mask_dc * f32x8::from(c.dy)),
                )
                .into(),
            dz: one_m
                .mul_add(
                    f32x8::from(a.dz),
                    mv.mul_add(f32x8::from(b.dz), diff * d_mask_dc * f32x8::from(c.dz)),
                )
                .into(),
        }
    }
}
