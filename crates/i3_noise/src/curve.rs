use crate::{Generator, NoisePoint, NoiseSample};
use wide::f32x8;

// ── Curve ─────────────────────────────────────────────────────────────────────
// Remaps the source output through a user-defined spline.
// Control points are [input, output] pairs sorted by input value.
// Interpolation: Catmull-Rom cubic Hermite between adjacent points.
// Outside the defined range: clamped to the nearest endpoint (gradient = 0).

pub struct Curve<G> {
    pub source: G,
    points: Vec<[f32; 2]>, // sorted by input
}

impl<G> Curve<G> {
    /// Control points are automatically sorted by input value.
    pub fn new(source: G, mut points: Vec<[f32; 2]>) -> Self {
        points.sort_unstable_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
        Self { source, points }
    }

    pub fn points(&self) -> &[[f32; 2]] { &self.points }

    /// Evaluate the spline at `v`.  Returns (mapped_value, d(output)/d(input)).
    fn eval(pts: &[[f32; 2]], v: f32) -> (f32, f32) {
        let n = pts.len();
        match n {
            0 => return (v, 1.0),
            1 => return (pts[0][1], 0.0),
            _ => {}
        }

        // Outside range: clamp, gradient = 0
        if v <= pts[0][0]     { return (pts[0][1],     0.0); }
        if v >= pts[n-1][0]   { return (pts[n-1][1],   0.0); }

        // Find the segment i1..i2 containing v
        let i2 = pts.partition_point(|p| p[0] < v).min(n - 1).max(1);
        let i1 = i2 - 1;

        // Catmull-Rom phantom neighbours at the endpoints
        let i0 = if i1 > 0   { i1 - 1 } else { i1 };
        let i3 = if i2 < n-1 { i2 + 1 } else { i2 };

        let x1 = pts[i1][0]; let y1 = pts[i1][1];
        let x2 = pts[i2][0]; let y2 = pts[i2][1];

        let seg = (x2 - x1).max(1e-10);
        let t   = (v - x1) / seg;

        // Catmull-Rom tangents (scaled to local segment length)
        let dx0 = (pts[i2][0] - pts[i0][0]).max(1e-10);
        let dx3 = (pts[i3][0] - pts[i1][0]).max(1e-10);
        let m1 = (pts[i2][1] - pts[i0][1]) / dx0 * seg;
        let m2 = (pts[i3][1] - pts[i1][1]) / dx3 * seg;

        // Cubic Hermite basis
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 =  2.0*t3 - 3.0*t2 + 1.0;
        let h10 =      t3 - 2.0*t2 + t;
        let h01 = -2.0*t3 + 3.0*t2;
        let h11 =      t3 -     t2;

        let value = h00*y1 + h10*m1 + h01*y2 + h11*m2;

        // Derivative w.r.t. t, then chain rule → d(value)/d(v_input)
        let dh00 =  6.0*t2 - 6.0*t;
        let dh10 =  3.0*t2 - 4.0*t + 1.0;
        let dh01 = -6.0*t2 + 6.0*t;
        let dh11 =  3.0*t2 - 2.0*t;
        let dv_dt = dh00*y1 + dh10*m1 + dh01*y2 + dh11*m2;

        (value, dv_dt / seg)
    }
}

impl<G: Generator> Generator for Curve<G> {
    fn sample(&self, point: &NoisePoint) -> NoiseSample {
        let s = self.source.sample(point);
        let mut values = [0f32; 8];
        let mut slopes = [0f32; 8];
        for i in 0..8 {
            let (v, d) = Self::eval(&self.points, s.value[i]);
            values[i] = v;
            slopes[i] = d;
        }
        let d = f32x8::from(slopes);
        NoiseSample {
            value: values,
            dx: (d * f32x8::from(s.dx)).into(),
            dy: (d * f32x8::from(s.dy)).into(),
            dz: (d * f32x8::from(s.dz)).into(),
        }
    }
}
