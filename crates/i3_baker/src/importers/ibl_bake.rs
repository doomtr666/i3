use crate::importers::ibl_math::*;
use half::f16;
use i3_io::ibl::IblSunData;
use intel_tex_2::{RgbaSurface, bc6h};
use rayon::prelude::*;
use std::f32::consts::PI;

/// Options de bake IBL.
#[derive(Debug, Clone)]
pub struct IblBakeOptions {
    pub extract_sun: bool,

    /// Multiplicateur appliqué à sun_intensity et stocké dans IblHeader.
    /// Convertit les valeurs HDR brutes vers des unités physiques (lux).
    pub intensity_scale: f32,

    /// Ratio direct/ambient cible pour le soleil.
    /// Quand défini, sun_intensity est calculé pour que la contribution directe du soleil soit
    /// exactement `sun_strength_ratio` fois l'irradiance ambiante masquée dans la direction soleil.
    /// Garantie : sur une surface Lambertienne face au soleil (NdotL=1) :
    ///   `L_direct / L_ambient = sun_strength_ratio`
    /// Valeurs typiques extérieur : 5.0 (nuageux) → 20.0 (plein soleil).
    /// None = intensité brute extraite du HDR (pas de calibration).
    pub sun_strength_ratio: Option<f32>,
}

impl Default for IblBakeOptions {
    fn default() -> Self {
        Self {
            extract_sun: true,
            intensity_scale: 1.0,
            sun_strength_ratio: Some(15.0),
        }
    }
}

/// Compresse les pixels f32 RGB equirect en BC6H_UF16.
/// Retourne les données brutes compressées (sans header).
pub fn compress_env_bc6h(pixels_rgb: &[[f32; 3]], width: u32, height: u32) -> Vec<u8> {
    // CompressBlocksBC6H_ispc attend RGBA fp16 (half-float), 8 bytes/pixel
    let mut rgba_f16: Vec<f16> = Vec::with_capacity(width as usize * height as usize * 4);
    for px in pixels_rgb {
        rgba_f16.push(f16::from_f32(px[0]));
        rgba_f16.push(f16::from_f32(px[1]));
        rgba_f16.push(f16::from_f32(px[2]));
        rgba_f16.push(f16::ONE); // alpha ignoré par BC6H
    }
    let surface = RgbaSurface {
        width,
        height,
        stride: width * 8, // 4 × f16 (8 bytes/pixel)
        data: bytemuck::cast_slice(&rgba_f16),
    };
    bc6h::compress_blocks(&bc6h::very_fast_settings(), &surface)
}

/// Encode un triplet RGB f32 en R11G11B10 unsigned float packed dans un u32.
pub fn encode_r11g11b10(r: f32, g: f32, b: f32) -> u32 {
    let r_enc = encode_ufloat(r, 5, 6);
    let g_enc = encode_ufloat(g, 5, 6);
    let b_enc = encode_ufloat(b, 5, 5);
    r_enc | (g_enc << 11) | (b_enc << (11 + 11))
}

fn encode_ufloat(val: f32, exp_bits: u32, mant_bits: u32) -> u32 {
    let val = val.max(0.0);
    if val == 0.0 {
        return 0;
    }
    let bits = val.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let bias = (1 << (exp_bits - 1)) - 1;
    let exp = exponent + bias as i32;

    if exp <= 0 {
        return 0;
    }
    if exp >= (1 << exp_bits) - 1 {
        return ((1 << exp_bits) - 1) << mant_bits | ((1 << mant_bits) - 1);
    }

    let mant = (bits >> (23 - mant_bits)) & ((1 << mant_bits) - 1);
    ((exp as u32) << mant_bits) | mant
}

/// Bake BRDF LUT (CPU, importance sampling GGX).
pub fn bake_brdf_lut(width: u32, height: u32, num_samples: u32) -> Vec<u8> {
    let mut data = vec![0u8; width as usize * height as usize * 4];
    let pixels: Vec<[f16; 2]> = (0..height)
        .into_par_iter()
        .flat_map(|iy| {
            (0..width).into_par_iter().map(move |ix| {
                let nov = (ix as f32 + 0.5) / width as f32;
                let roughness = (iy as f32 + 0.5) / height as f32;
                let alpha = roughness * roughness;

                let mut scale = 0.0f32;
                let mut bias = 0.0f32;

                let v = [(1.0 - nov * nov).sqrt(), nov, 0.0];

                for i in 0..num_samples {
                    let xi = hammersley(i, num_samples);
                    let h = importance_sample_ggx(xi, alpha);
                    let nol_h = (v[0] * h[0] + v[1] * h[1] + v[2] * h[2]).max(0.0);
                    let l = [
                        2.0 * nol_h * h[0] - v[0],
                        2.0 * nol_h * h[1] - v[1],
                        2.0 * nol_h * h[2] - v[2],
                    ];

                    let nol = l[1].max(0.0);
                    let noh = h[1].max(0.0);
                    let nov_h = nol_h;

                    if nol > 0.0 {
                        let g = g_smith_schlick(nol, nov, roughness);
                        let g_vis = (g * nov_h) / (noh * nov).max(1e-4);
                        let fc = (1.0 - nov_h).powi(5);

                        scale += (1.0 - fc) * g_vis;
                        bias += fc * g_vis;
                    }
                }

                [
                    f16::from_f32(scale / num_samples as f32),
                    f16::from_f32(bias / num_samples as f32),
                ]
            })
        })
        .collect();

    for (i, p) in pixels.iter().enumerate() {
        let bytes = bytemuck::bytes_of(p);
        data[i * 4..i * 4 + 4].copy_from_slice(bytes);
    }
    data
}

fn g_smith_schlick(nol: f32, nov: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    let g1l = nol / (nol * (1.0 - k) + k);
    let g1v = nov / (nov * (1.0 - k) + k);
    g1l * g1v
}

/// Bake Irradiance hemi-octa (CPU, intégrale cosine-weighted).
pub fn bake_irradiance(
    pixels_rgb: &[[f32; 3]],
    src_width: u32,
    src_height: u32,
    out_size: u32,
    sun_threshold: f32,
) -> Vec<u8> {
    let mut data = vec![0u8; out_size as usize * out_size as usize * 4];
    let packed_pixels: Vec<u32> = (0..out_size)
        .into_par_iter()
        .flat_map(|iy| {
            (0..out_size).into_par_iter().map(move |ix| {
                let u = (ix as f32 + 0.5) / out_size as f32;
                let v = (iy as f32 + 0.5) / out_size as f32;
                let n = hemi_octa_decode(u, v);

                let mut irr = [0.0f32; 3];

                for sy in 0..src_height {
                    let solid = equirect_solid_angle(sy, src_height);
                    for sx in 0..src_width {
                        let dir = equirect_to_dir(sx, sy, src_width, src_height);
                        let cos_theta = (n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2]).max(0.0);

                        if cos_theta > 0.0 {
                            let px = pixels_rgb[(sx + sy * src_width) as usize];
                            if sun_threshold == f32::MAX
                                || luminance(px[0], px[1], px[2]) <= sun_threshold
                            {
                                irr[0] += px[0] * cos_theta * solid;
                                irr[1] += px[1] * cos_theta * solid;
                                irr[2] += px[2] * cos_theta * solid;
                            }
                        }
                    }
                }

                encode_r11g11b10(irr[0] / PI, irr[1] / PI, irr[2] / PI)
            })
        })
        .collect();

    for (i, p) in packed_pixels.iter().enumerate() {
        data[i * 4..i * 4 + 4].copy_from_slice(&p.to_ne_bytes());
    }
    data
}

/// Bake Pre-filtered hemi-octa (CPU, importance sampling GGX).
pub fn bake_prefiltered(
    pixels_rgb: &[[f32; 3]],
    src_width: u32,
    src_height: u32,
    out_size: u32,
    num_mips: u32,
    num_samples: u32,
    sun_threshold: f32,
) -> Vec<u8> {
    let mut all_data = Vec::new();

    for m in 0..num_mips {
        let roughness = m as f32 / (num_mips - 1) as f32;
        let alpha = roughness * roughness;
        let mip_size = (out_size >> m).max(1);

        let packed_pixels: Vec<u32> = (0..mip_size)
            .into_par_iter()
            .flat_map(|iy| {
                (0..mip_size).into_par_iter().map(move |ix| {
                    let u = (ix as f32 + 0.5) / mip_size as f32;
                    let v = (iy as f32 + 0.5) / mip_size as f32;
                    let r = hemi_octa_decode(u, v);
                    let n = r;
                    let v_dir = r;

                    let mut color = [0.0f32; 3];
                    let mut total_weight = 0.0f32;

                    for i in 0..num_samples {
                        let xi = hammersley(i, num_samples);
                        let h_local = importance_sample_ggx(xi, alpha);
                        let h = tbn_transform(h_local, n);

                        let noh = (n[0] * h[0] + n[1] * h[1] + n[2] * h[2]).max(0.0);
                        let l = [
                            2.0 * noh * h[0] - v_dir[0],
                            2.0 * noh * h[1] - v_dir[1],
                            2.0 * noh * h[2] - v_dir[2],
                        ];

                        let nol = (n[0] * l[0] + n[1] * l[1] + n[2] * l[2]).max(0.0);
                        if nol > 0.0 {
                            let sample = sample_equirect(pixels_rgb, l, src_width, src_height);
                            if sun_threshold == f32::MAX
                                || luminance(sample[0], sample[1], sample[2]) <= sun_threshold
                            {
                                color[0] += sample[0] * nol;
                                color[1] += sample[1] * nol;
                                color[2] += sample[2] * nol;
                                total_weight += nol;
                            }
                        }
                    }

                    if total_weight > 0.0 {
                        encode_r11g11b10(
                            color[0] / total_weight,
                            color[1] / total_weight,
                            color[2] / total_weight,
                        )
                    } else {
                        // Fallback to average or black
                        let sample = sample_equirect(pixels_rgb, r, src_width, src_height);
                        encode_r11g11b10(sample[0], sample[1], sample[2])
                    }
                })
            })
            .collect();

        for p in packed_pixels {
            all_data.extend_from_slice(&p.to_ne_bytes());
        }
    }
    all_data
}

/// Luminance de l'irradiance cosine-weighted masquée dans la direction `n` (normalisée).
/// Utilisé pour calibrer sun_intensity via `sun_strength_ratio`.
pub fn irradiance_lum_at(
    pixels_rgb: &[[f32; 3]],
    src_width: u32,
    src_height: u32,
    n: [f32; 3],
    sun_threshold: f32,
) -> f32 {
    let mut irr = [0.0f32; 3];
    for sy in 0..src_height {
        let solid = equirect_solid_angle(sy, src_height);
        for sx in 0..src_width {
            let dir = equirect_to_dir(sx, sy, src_width, src_height);
            let cos_theta = (n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2]).max(0.0);
            if cos_theta > 0.0 {
                let px = pixels_rgb[(sx + sy * src_width) as usize];
                if sun_threshold == f32::MAX || luminance(px[0], px[1], px[2]) <= sun_threshold {
                    irr[0] += px[0] * cos_theta * solid;
                    irr[1] += px[1] * cos_theta * solid;
                    irr[2] += px[2] * cos_theta * solid;
                }
            }
        }
    }
    luminance(irr[0] / PI, irr[1] / PI, irr[2] / PI)
}

pub fn compute_sun_threshold(pixels_rgb: &[[f32; 3]]) -> f32 {
    let mut lums: Vec<f32> = pixels_rgb
        .iter()
        .map(|p| luminance(p[0], p[1], p[2]))
        .collect();
    lums.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((lums.len() as f32 * 0.99) as usize).min(lums.len() - 1);
    let threshold = lums[idx];
    let n_masked = lums.iter().filter(|&&l| l > threshold).count();
    tracing::warn!(
        "IBL sun threshold: {:.2} ({} pixels masked, {:.4}% of total)",
        threshold,
        n_masked,
        n_masked as f32 / lums.len() as f32 * 100.0
    );
    threshold
}

pub fn extract_sun(
    pixels_rgb: &[[f32; 3]],
    src_width: u32,
    src_height: u32,
    sun_threshold: f32,
) -> IblSunData {
    // Direction centroid weighted by lum × solid_angle (correct angular weighting).
    let mut w_dir = [0.0f32; 3];
    // Color/intensity accumulated as raw radiance (no solid angle) → average sun disk radiance.
    let mut sum_r = 0.0f32;
    let mut sum_g = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut sum_lum = 0.0f32; // raw lum sum, no solid angle
    let mut count = 0usize;

    for sy in 0..src_height {
        let solid = equirect_solid_angle(sy, src_height);
        for sx in 0..src_width {
            let px = pixels_rgb[(sx + sy * src_width) as usize];
            let lum = luminance(px[0], px[1], px[2]);
            if lum > sun_threshold {
                let dir = equirect_to_dir(sx, sy, src_width, src_height);
                let w = lum * solid;
                w_dir[0] += dir[0] * w;
                w_dir[1] += dir[1] * w;
                w_dir[2] += dir[2] * w;

                // Raw accumulation for average radiance
                sum_r += px[0];
                sum_g += px[1];
                sum_b += px[2];
                sum_lum += lum;
                count += 1;
            }
        }
    }

    if count > 0 {
        let n = count as f32;
        // Direction: normalize weighted centroid, then negate to match renderer convention.
        // equirect_to_dir gives "toward sun" but the renderer stores "light ray direction"
        // (from sun toward scene), matching scene LightData::direction and the sky/deferred shaders.
        let len = (w_dir[0] * w_dir[0] + w_dir[1] * w_dir[1] + w_dir[2] * w_dir[2])
            .sqrt()
            .max(1e-6);
        let direction = [-w_dir[0] / len, -w_dir[1] / len, -w_dir[2] / len];

        // Color: average RGB of sun pixels, normalized by max component → chromaticity in [0,1]
        let avg_r = sum_r / n;
        let avg_g = sum_g / n;
        let avg_b = sum_b / n;
        let max_c = avg_r.max(avg_g).max(avg_b).max(1e-6);
        let color = [avg_r / max_c, avg_g / max_c, avg_b / max_c];

        // Intensity: average luminance of the sun disk pixels (raw radiance, no solid angle)
        let intensity = sum_lum / n;

        IblSunData {
            direction,
            intensity,
            color,
            _pad: 0.0,
        }
    } else {
        IblSunData {
            direction: [0.0, -1.0, 0.0],
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            _pad: 0.0,
        }
    }
}

// Helpers

fn hammersley(i: u32, n: u32) -> (f32, f32) {
    let mut bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    let radial_stack = bits as f32 * 2.3283064365386963e-10;
    (i as f32 / n as f32, radial_stack)
}

fn importance_sample_ggx(xi: (f32, f32), alpha: f32) -> [f32; 3] {
    let phi = 2.0 * PI * xi.0;
    let cos_theta = ((1.0 - xi.1) / (1.0 + (alpha * alpha - 1.0) * xi.1)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    [sin_theta * phi.cos(), cos_theta, sin_theta * phi.sin()]
}

fn tbn_transform(h: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    let up = if n[1].abs() < 0.999 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let tangent = normalize([
        up[1] * n[2] - up[2] * n[1],
        up[2] * n[0] - up[0] * n[2],
        up[0] * n[1] - up[1] * n[0],
    ]);
    let bitangent = [
        n[1] * tangent[2] - n[2] * tangent[1],
        n[2] * tangent[0] - n[0] * tangent[2],
        n[0] * tangent[1] - n[1] * tangent[0],
    ];
    [
        tangent[0] * h[0] + bitangent[0] * h[2] + n[0] * h[1],
        tangent[1] * h[0] + bitangent[1] * h[2] + n[1] * h[1],
        tangent[2] * h[0] + bitangent[2] * h[2] + n[2] * h[1],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-8);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn sample_equirect(pixels: &[[f32; 3]], dir: [f32; 3], w: u32, h: u32) -> [f32; 3] {
    let u = (dir[2].atan2(dir[0]) / (2.0 * PI) + 0.5).clamp(0.0, 1.0);
    let v = (dir[1].acos() / PI).clamp(0.0, 1.0);
    let ix = ((u * w as f32).floor() as u32).min(w - 1);
    let iy = ((v * h as f32).floor() as u32).min(h - 1);
    pixels[(ix + iy * w) as usize]
}
