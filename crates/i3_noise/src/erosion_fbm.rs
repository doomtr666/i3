use crate::{VecGenerator, VecPosition, VecSample};

#[derive(Clone)]
pub struct ErosionFbmVec<T> {
    pub source: T,
    pub octaves: usize,
    pub lacunarity: f32,
    pub gain: f32,
    pub erosion_strength: f32,
}

impl<T> ErosionFbmVec<T>
where
    T: VecGenerator,
{
    pub fn new(source: T, octaves: usize, lacunarity: f32, gain: f32, erosion_strength: f32) -> ErosionFbmVec<T> {
        ErosionFbmVec::<T> {
            source,
            octaves,
            lacunarity,
            gain,
            erosion_strength,
        }
    }
}

impl<T> VecGenerator for ErosionFbmVec<T>
where
    T: VecGenerator,
{
    #[inline(always)]
    fn eval_vec(&self, pos: &VecPosition, dd: f32) -> VecSample {
        let mut out = VecSample::default();
        let mut freq = 1.0;
        let mut amp = 1.0;

        let mut acc_dx = [1.0; 8]; // weight
        let mut acc_dy = [0.0; 8];
        let mut acc_dz = [0.0; 8];

        for _ in 0..self.octaves {
            let mut scaled_pos = VecPosition {
                x: [0.0; 8],
                y: [0.0; 8],
                z: [0.0; 8],
            };

            let mut scaled_pos = VecPosition {
                x: [0.0; 8],
                y: [0.0; 8],
                z: [0.0; 8],
            };

            for i in 0..8 {
                scaled_pos.x[i] = pos.x[i] * freq;
                scaled_pos.y[i] = pos.y[i] * freq;
                scaled_pos.z[i] = pos.z[i] * freq;
            }

            let detail_size = 1.0 / freq;
            let fade = smoothstep(0.0, 2.0 * dd, detail_size);

            if fade <= 0.0 {
                break;
            }

            let sample = self.source.eval_vec(&scaled_pos, dd * freq);
            let effective_amp = amp * fade;

            for i in 0..8 {
                // Ridged math: n = (1 - abs(v))^2
                let v = sample.value[i];
                let abs_v = v.abs();
                let ridge = 1.0 - abs_v;
                let n_v = ridge * ridge;

                // Derivative of (1 - abs(v))^2 is -2 * (1 - abs(v)) * sign(v) * dv
                let sign_v = if v >= 0.0 { 1.0 } else { -1.0 };
                let d_scale = -2.0 * ridge * sign_v;

                let n_dx = sample.dx[i] * d_scale;
                let n_dy = sample.dy[i] * d_scale;
                let n_dz = sample.dz[i] * d_scale;

                // Weight from previous octaves
                let w = acc_dx[i]; // We repurpose acc_dx to store the weight

                let n_v_shifted = n_v * 2.0 - 1.0;
                let n_dx_shifted = n_dx * 2.0;
                let n_dy_shifted = n_dy * 2.0;
                let n_dz_shifted = n_dz * 2.0;

                out.value[i] += n_v_shifted * effective_amp * w;
                out.dx[i] += n_dx_shifted * freq * effective_amp * w;
                out.dy[i] += n_dy_shifted * freq * effective_amp * w;
                out.dz[i] += n_dz_shifted * freq * effective_amp * w;

                // Update weight for next octave
                let next_w = (n_v * self.erosion_strength).clamp(0.0, 1.0);
                acc_dx[i] = next_w;
            }

            freq *= self.lacunarity;
            amp *= self.gain;
        }

        out
    }
}

#[inline(always)]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
