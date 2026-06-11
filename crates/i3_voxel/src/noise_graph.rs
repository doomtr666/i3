//! Noise graph → two backends, kept in lockstep for CPU/GPU parity:
//!  - [`NoiseGraph::compile`]   → flat `VmOp` bytecode for the GPU interpreter
//!    (`evalGraph` in `noise.slangh`).
//!  - [`NoiseGraph::build_rnoise`] → an `i3_noise` (rnoise) generator, the CPU
//!    reference used by the SVO's cull/refine and as the parity oracle.
//!
//! Opcodes/order MUST match `evalGraph`. Combinators are post-order (children then
//! op); domain ops bracket their inner subtree as `DPUSH … DPOP`.

use std::sync::Arc;

use i3_noise::{
    Abs, Add, Clamp, Const, DomainRotate, DomainScale, DomainTranslate, DomainWarp, Fbm, Generator,
    Max, Min, Mul, Neg, Perlin, ScaleBias, Sub,
};

// ─── GPU op (mirror of `struct VmOp` in noise.slangh).
// 32 BYTES (16-byte multiple) ON PURPOSE: a StructuredBuffer element stride is rounded
// up to 16 by the Slang/SPIRV layout, so a 24-byte struct would read with a 32-byte
// stride on the GPU and desync from a 24-byte CPU upload (every op after the first
// shifts). `e`/`f` are spare params (currently 0).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VmOp {
    pub op:  u32,
    pub aux: u32,
    pub a:   f32,
    pub b:   f32,
    pub c:   f32,
    pub d:   f32,
    pub e:   f32,
    pub f:   f32,
}

impl VmOp {
    fn new(op: u32, aux: u32, a: f32, b: f32, c: f32, d: f32) -> Self {
        Self { op, aux, a, b, c, d, e: 0.0, f: 0.0 }
    }
}

// Opcodes — MUST match noise.slangh.
const OP_CONST: u32 = 0;
const OP_PERLIN: u32 = 1;
const OP_FBM: u32 = 2;
const OP_ADD: u32 = 10;
const OP_SUB: u32 = 11;
const OP_MUL: u32 = 12;
const OP_MIN: u32 = 13;
const OP_MAX: u32 = 14;
const OP_NEG: u32 = 15;
const OP_SCALEBIAS: u32 = 20;
const OP_ABS: u32 = 21;
const OP_CLAMP: u32 = 22;
const OP_DPUSH: u32 = 30;
const OP_DPOP: u32 = 31;
const OP_END: u32 = 99;
const DOM_SCALE: u32 = 0;
const DOM_TRANSLATE: u32 = 1;
const DOM_ROTATE: u32 = 2;
const DOM_WARP: u32 = 3;

// ─── The graph (tree). `Fbm` wraps the Perlin base (M2 constraint: fbm-of-base-noise).
#[derive(Clone)]
pub enum NoiseGraph {
    Const(f32),
    Perlin { seed: u32 },
    Fbm { seed: u32, octaves: i32, lacunarity: f32, gain: f32 },
    Add(Box<NoiseGraph>, Box<NoiseGraph>),
    Sub(Box<NoiseGraph>, Box<NoiseGraph>),
    Mul(Box<NoiseGraph>, Box<NoiseGraph>),
    Min(Box<NoiseGraph>, Box<NoiseGraph>),
    Max(Box<NoiseGraph>, Box<NoiseGraph>),
    Neg(Box<NoiseGraph>),
    ScaleBias { src: Box<NoiseGraph>, scale: f32, bias: f32 },
    Abs(Box<NoiseGraph>),
    Clamp { src: Box<NoiseGraph>, lo: f32, hi: f32 },
    DomainScale { src: Box<NoiseGraph>, factor: f32 },
    DomainTranslate { src: Box<NoiseGraph>, t: [f32; 3] },
    DomainRotate { src: Box<NoiseGraph>, angles: [f32; 3] },
    DomainWarp { inner: Box<NoiseGraph>, x: Box<NoiseGraph>, y: Box<NoiseGraph>, z: Box<NoiseGraph>, intensity: f32 },
}

type Gen = Arc<dyn Generator + Send + Sync>;

impl NoiseGraph {
    /// Flatten to GPU bytecode (post-order; domain ops bracket their inner subtree).
    pub fn compile(&self) -> Vec<VmOp> {
        let mut out = Vec::new();
        self.emit(&mut out);
        out.push(VmOp::new(OP_END, 0, 0.0, 0.0, 0.0, 0.0)); // sentinel (see evalGraph)
        out
    }

    fn emit(&self, out: &mut Vec<VmOp>) {
        match self {
            // Seeds are u32 → carried in `aux` (uint), NOT a float param (the shader
            // reads int(o.aux); a float would be a lossy/garbage value conversion).
            NoiseGraph::Const(v) => out.push(VmOp::new(OP_CONST, 0, *v, 0.0, 0.0, 0.0)),
            NoiseGraph::Perlin { seed } => out.push(VmOp::new(OP_PERLIN, *seed, 0.0, 0.0, 0.0, 0.0)),
            NoiseGraph::Fbm { seed, octaves, lacunarity, gain } => {
                out.push(VmOp::new(OP_FBM, *seed, *octaves as f32, *lacunarity, *gain, 0.0));
            }
            NoiseGraph::Add(a, b) => { a.emit(out); b.emit(out); out.push(VmOp::new(OP_ADD, 0, 0.0, 0.0, 0.0, 0.0)); }
            NoiseGraph::Sub(a, b) => { a.emit(out); b.emit(out); out.push(VmOp::new(OP_SUB, 0, 0.0, 0.0, 0.0, 0.0)); }
            NoiseGraph::Mul(a, b) => { a.emit(out); b.emit(out); out.push(VmOp::new(OP_MUL, 0, 0.0, 0.0, 0.0, 0.0)); }
            NoiseGraph::Min(a, b) => { a.emit(out); b.emit(out); out.push(VmOp::new(OP_MIN, 0, 0.0, 0.0, 0.0, 0.0)); }
            NoiseGraph::Max(a, b) => { a.emit(out); b.emit(out); out.push(VmOp::new(OP_MAX, 0, 0.0, 0.0, 0.0, 0.0)); }
            NoiseGraph::Neg(s) => { s.emit(out); out.push(VmOp::new(OP_NEG, 0, 0.0, 0.0, 0.0, 0.0)); }
            NoiseGraph::ScaleBias { src, scale, bias } => { src.emit(out); out.push(VmOp::new(OP_SCALEBIAS, 0, *scale, *bias, 0.0, 0.0)); }
            NoiseGraph::Abs(s) => { s.emit(out); out.push(VmOp::new(OP_ABS, 0, 0.0, 0.0, 0.0, 0.0)); }
            NoiseGraph::Clamp { src, lo, hi } => { src.emit(out); out.push(VmOp::new(OP_CLAMP, 0, *lo, *hi, 0.0, 0.0)); }
            NoiseGraph::DomainScale { src, factor } => {
                out.push(VmOp::new(OP_DPUSH, DOM_SCALE, *factor, 0.0, 0.0, 0.0));
                src.emit(out);
                out.push(VmOp::new(OP_DPOP, 0, 0.0, 0.0, 0.0, 0.0));
            }
            NoiseGraph::DomainTranslate { src, t } => {
                out.push(VmOp::new(OP_DPUSH, DOM_TRANSLATE, t[0], t[1], t[2], 0.0));
                src.emit(out);
                out.push(VmOp::new(OP_DPOP, 0, 0.0, 0.0, 0.0, 0.0));
            }
            NoiseGraph::DomainRotate { src, angles } => {
                out.push(VmOp::new(OP_DPUSH, DOM_ROTATE, angles[0], angles[1], angles[2], 0.0));
                src.emit(out);
                out.push(VmOp::new(OP_DPOP, 0, 0.0, 0.0, 0.0, 0.0));
            }
            NoiseGraph::DomainWarp { inner, x, y, z, intensity } => {
                // Warp fields first (evaluated at the parent point), then the bracket.
                x.emit(out);
                y.emit(out);
                z.emit(out);
                out.push(VmOp::new(OP_DPUSH, DOM_WARP, *intensity, 0.0, 0.0, 0.0));
                inner.emit(out);
                out.push(VmOp::new(OP_DPOP, 0, 0.0, 0.0, 0.0, 0.0));
            }
        }
    }

    /// Build the equivalent rnoise generator (CPU reference / parity oracle).
    pub fn build_rnoise(&self) -> Gen {
        match self {
            NoiseGraph::Const(v) => Arc::new(Const::new(*v)),
            NoiseGraph::Perlin { seed } => Arc::new(Perlin::new(*seed)),
            NoiseGraph::Fbm { seed, octaves, lacunarity, gain } => {
                Arc::new(Fbm::new(Perlin::new(*seed), *octaves as usize, *lacunarity, *gain))
            }
            NoiseGraph::Add(a, b) => Arc::new(Add::new(a.build_rnoise(), b.build_rnoise())),
            NoiseGraph::Sub(a, b) => Arc::new(Sub::new(a.build_rnoise(), b.build_rnoise())),
            NoiseGraph::Mul(a, b) => Arc::new(Mul::new(a.build_rnoise(), b.build_rnoise())),
            NoiseGraph::Min(a, b) => Arc::new(Min::new(a.build_rnoise(), b.build_rnoise())),
            NoiseGraph::Max(a, b) => Arc::new(Max::new(a.build_rnoise(), b.build_rnoise())),
            NoiseGraph::Neg(s) => Arc::new(Neg::new(s.build_rnoise())),
            NoiseGraph::ScaleBias { src, scale, bias } => Arc::new(ScaleBias::new(src.build_rnoise(), *scale, *bias)),
            NoiseGraph::Abs(s) => Arc::new(Abs::new(s.build_rnoise())),
            NoiseGraph::Clamp { src, lo, hi } => Arc::new(Clamp::new(src.build_rnoise(), *lo, *hi)),
            NoiseGraph::DomainScale { src, factor } => Arc::new(DomainScale::new(src.build_rnoise(), *factor)),
            NoiseGraph::DomainTranslate { src, t } => Arc::new(DomainTranslate::new(src.build_rnoise(), t[0], t[1], t[2])),
            NoiseGraph::DomainRotate { src, angles } => Arc::new(DomainRotate::new(src.build_rnoise(), angles[0], angles[1], angles[2])),
            NoiseGraph::DomainWarp { inner, x, y, z, intensity } => Arc::new(DomainWarp::new(
                inner.build_rnoise(), x.build_rnoise(), y.build_rnoise(), z.build_rnoise(), *intensity,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use i3_noise::NoisePoint;

    // The demo's terrain graph: the CPU generator must VARY across space (not flat) and
    // match a manual freq-scaled fBm — i.e. DomainScale actually scales the domain.
    #[test]
    fn cpu_generator_varies_and_matches_manual_freq() {
        let g = NoiseGraph::DomainScale {
            factor: 1.0 / 32.0,
            src: Box::new(NoiseGraph::Fbm { seed: 1337, octaves: 3, lacunarity: 2.0, gain: 0.5 }),
        };
        let built = g.build_rnoise();
        let manual = i3_noise::Fbm::new(i3_noise::Perlin::new(1337), 3, 2.0, 0.5);

        let sample = |gn: &dyn Generator, x: f32, y: f32, z: f32| {
            let mut np = NoisePoint::default();
            np.x[0] = x; np.y[0] = y; np.z[0] = z;
            gn.sample(&np).value[0]
        };

        let v0 = sample(&*built, 0.0, 0.0, 0.0);
        let v1 = sample(&*built, 40.0, 0.0, 0.0);
        let v2 = sample(&*built, 0.0, 0.0, 40.0);
        eprintln!("graph values: v0={v0} v1={v1} v2={v2}");
        assert!((v0 - v1).abs() > 1e-4 || (v0 - v2).abs() > 1e-4, "CPU generator looks FLAT");

        // DomainScale(1/32) at p  ==  Fbm at p/32.
        for &(x, y, z) in &[(5.0f32, 3.0, 7.0), (50.0, 10.0, -20.0), (-13.0, 4.0, 31.0)] {
            let a = sample(&*built, x, y, z);
            let b = sample(&manual, x / 32.0, y / 32.0, z / 32.0);
            assert!((a - b).abs() < 1e-5, "DomainScale mismatch at ({x},{y},{z}): {a} vs {b}");
        }
    }
}
