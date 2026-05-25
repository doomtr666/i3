use crate::{VecGenerator, VecPosition, VecSample};

#[derive(Clone)]
pub struct FbmVec<T> {
    pub source: T,
    pub octaves: usize,
    pub lacunarity: f32,
    pub gain: f32,
}

impl<T> FbmVec<T>
where
    T: VecGenerator,
{
    pub fn new(source: T, octaves: usize, lacunarity: f32, gain: f32) -> FbmVec<T> {
        FbmVec::<T> {
            source,
            octaves,
            lacunarity,
            gain,
        }
    }
}

impl<T> VecGenerator for FbmVec<T>
where
    T: VecGenerator,
{
    #[inline(always)]
    fn eval_vec(&self, pos: &VecPosition, dd: f32) -> VecSample {
        // Assuming VecSample derives Default as discussed
        let mut out = VecSample::default();
        let mut freq = 1.0;
        let mut amp = 1.0;

        for _ in 0..self.octaves {
            // 1. Scale domain for all 8 elements
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

            // 2. Anti-aliasing spatial filter
            let detail_size = 1.0 / freq;
            let fade = smoothstep(0.0, 2.0 * dd, detail_size);

            if fade <= 0.0 {
                break;
            }

            // 3. Evaluate the source with the scaled block
            let sample = self.source.eval_vec(&scaled_pos, dd * freq);
            let effective_amp = amp * fade;

            // 4. Accumulate values and chain-rule derivatives
            for i in 0..8 {
                out.value[i] += sample.value[i] * effective_amp;
                out.dx[i] += sample.dx[i] * freq * effective_amp;
                out.dy[i] += sample.dy[i] * freq * effective_amp;
                out.dz[i] += sample.dz[i] * freq * effective_amp;
            }

            // 5. Next octave
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
