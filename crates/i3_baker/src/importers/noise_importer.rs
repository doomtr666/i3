use std::path::Path;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::pipeline::{BakeContext, BakeOutput, ImportedData, Importer};
use crate::Result;
use i3_io::noise_asset::{NOISE_ASSET_TYPE, NoiseAssetHeader};

// ─── Manifest types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseManifestEntry {
    pub name:      String,
    pub size:      u32,
    #[serde(default)]
    pub seed:      u64,
    pub algorithm: NoiseAlgorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseAlgorithm {
    WhiteNoise,
    BlueNoise { sigma: f32 },
}

// ─── Imported data (virtual — no source file) ─────────────────────────────────

struct NoiseImportedData {
    entry: NoiseManifestEntry,
}

impl ImportedData for NoiseImportedData {
    fn source_path(&self) -> &Path { Path::new("") }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

// ─── Importer ─────────────────────────────────────────────────────────────────

pub struct NoiseImporter {
    pub entry: NoiseManifestEntry,
}

impl Importer for NoiseImporter {
    fn name(&self) -> &str { "NoiseImporter" }
    fn source_extensions(&self) -> &[&str] { &[] }

    fn import(&self, _source_path: &Path) -> Result<Box<dyn ImportedData>> {
        Ok(Box::new(NoiseImportedData { entry: self.entry.clone() }))
    }

    fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let d = data.as_any().downcast_ref::<NoiseImportedData>().unwrap();
        let n = d.entry.size as usize;

        let (pixel_data, algorithm_id, sigma) = match &d.entry.algorithm {
            NoiseAlgorithm::WhiteNoise => (generate_white(n, d.entry.seed), 0u32, 0.0f32),
            NoiseAlgorithm::BlueNoise { sigma } => {
                (generate_blue(n, *sigma as f64, d.entry.seed), 1u32, *sigma)
            }
        };

        let header = NoiseAssetHeader {
            width:     n as u32,
            height:    n as u32,
            channels:  4,
            format:    14, // R16G16B16A16_UNORM
            seed:      d.entry.seed,
            algorithm: algorithm_id,
            sigma,
            data_size: pixel_data.len() as u64,
            _pad:      [0; 24],
        };

        let mut payload = bytemuck::bytes_of(&header).to_vec();
        payload.extend_from_slice(&pixel_data);

        let virtual_path = format!("noise__{}", d.entry.name);
        let asset_id = Uuid::new_v5(&Uuid::NAMESPACE_URL, virtual_path.as_bytes());

        tracing::info!(
            "[NoiseImporter] baked '{}' ({}×{} RGBA16, {} bytes)",
            d.entry.name, n, n, payload.len()
        );

        Ok(vec![BakeOutput {
            asset_id,
            asset_type: NOISE_ASSET_TYPE,
            data: payload,
            name: d.entry.name.clone(),
        }])
    }
}

// ─── PCG hash helpers ─────────────────────────────────────────────────────────

fn pcg32(state: u64) -> u32 {
    let s = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let xorshifted = (((s >> 18) ^ s) >> 27) as u32;
    let rot = (s >> 59) as u32;
    xorshifted.rotate_right(rot)
}

fn hash_pixel(x: u64, y: u64, seed: u64) -> u32 {
    let state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(x.wrapping_mul(2654435761))
        .wrapping_mul(6364136223846793005)
        .wrapping_add(y.wrapping_mul(2246822519));
    pcg32(state)
}

fn hash_freq(u: u64, v: u64, seed: u64) -> f64 {
    hash_pixel(u, v, seed) as f64 / 4294967296.0
}

// ─── White noise generation ───────────────────────────────────────────────────

fn generate_white(size: usize, seed: u64) -> Vec<u8> {
    let n2 = size * size;
    let mut out = vec![0u8; n2 * 4 * 2];

    for pixel in 0..n2 {
        let x = (pixel % size) as u64;
        let y = (pixel / size) as u64;
        for ch in 0..4usize {
            let seed_ch = seed ^ ((ch as u64).wrapping_mul(0x9e3779b97f4a7c15));
            let val = hash_pixel(x, y, seed_ch) as u16;
            let off = (pixel * 4 + ch) * 2;
            out[off]     = (val & 0xFF) as u8;
            out[off + 1] = (val >> 8)   as u8;
        }
    }
    out
}

// ─── Blue noise generation — iDFT + rank normalization ───────────────────────

fn generate_blue(size: usize, sigma: f64, seed: u64) -> Vec<u8> {
    use rustfft::{FftPlanner, num_complex::Complex};

    let n   = size;
    let n2  = n * n;
    let half_n     = (n / 2) as f64;
    let two_sigma2 = 2.0 * sigma * sigma;

    let channel_data: Vec<Vec<u16>> = (0..4u64)
        .map(|ch| {
            let seed_ch = seed ^ ch.wrapping_mul(0x9e3779b97f4a7c15);

            // 1. Build frequency domain (row-major: index = v * n + u)
            let mut freq: Vec<Complex<f64>> = (0..n2).map(|idx| {
                let u = (idx % n) as u64;
                let v = (idx / n) as u64;
                // Centred frequency
                let fu = if u as usize <= n / 2 { u as f64 } else { u as f64 - n as f64 };
                let fv = if v as usize <= n / 2 { v as f64 } else { v as f64 - n as f64 };
                let r = (fu * fu + fv * fv).sqrt() / half_n;
                // High-pass Gaussian weight
                let w = 1.0 - (-r * r / two_sigma2).exp();
                // Random phase
                let phase = hash_freq(u, v, seed_ch) * std::f64::consts::TAU;
                Complex::new(w * phase.cos(), w * phase.sin())
            }).collect();

            // 2. Separable 2D IFFT
            let mut planner = FftPlanner::new();
            let ifft = planner.plan_fft_inverse(n);
            let mut buf = vec![Complex::new(0.0f64, 0.0); n];

            // Column IFFTs (along v for each u)
            for u in 0..n {
                for v in 0..n { buf[v] = freq[v * n + u]; }
                ifft.process(&mut buf);
                for v in 0..n { freq[v * n + u] = buf[v]; }
            }

            // Row IFFTs (along u for each v)
            for v in 0..n {
                buf.copy_from_slice(&freq[v * n..(v + 1) * n]);
                ifft.process(&mut buf);
                freq[v * n..(v + 1) * n].copy_from_slice(&buf);
            }

            // 3. Rank normalization — sort indices by real part, assign uniform ranks
            let mut indices: Vec<usize> = (0..n2).collect();
            indices.sort_unstable_by(|&a, &b| {
                freq[a].re.partial_cmp(&freq[b].re).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut ranked = vec![0u16; n2];
            for (rank, &idx) in indices.iter().enumerate() {
                ranked[idx] = (rank * 65535 / (n2 - 1)) as u16;
            }
            ranked
        })
        .collect();

    // Interleave RGBA u16 → bytes (LE)
    let mut out = vec![0u8; n2 * 4 * 2];
    for pixel in 0..n2 {
        for ch in 0..4 {
            let val = channel_data[ch][pixel];
            let off = (pixel * 4 + ch) * 2;
            out[off]     = (val & 0xFF) as u8;
            out[off + 1] = (val >> 8)   as u8;
        }
    }
    out
}
