use crate::Result;
use crate::pipeline::{BakeContext, BakeOutput, ImportedData, Importer};
use i3_io::texture::{TEXTURE_ASSET_TYPE, TextureFormat, TextureHeader};
use image::GenericImageView;
use intel_tex_2;
use std::path::{Path, PathBuf};
use uuid::Uuid;

pub struct ImageImporter {
    options: TextureImportOptions,
}

#[derive(Debug, Clone, Copy)]
pub struct TextureImportOptions {
    pub is_srgb: bool,
    pub generate_mips: bool,
    pub format: TextureFormat,
}

impl Default for TextureImportOptions {
    fn default() -> Self {
        Self {
            is_srgb: true,
            generate_mips: true,
            format: TextureFormat::BC7_SRGB,
        }
    }
}

impl ImageImporter {
    pub fn new(options: TextureImportOptions) -> Self {
        Self { options }
    }

    /// Convert sRGB u8 to linear f32
    fn srgb_to_linear(v: f32) -> f32 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Convert linear f32 to sRGB f32
    fn linear_to_srgb(v: f32) -> f32 {
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }

    pub fn import_memory(
        &self,
        buffer: &[u8],
        source_path: &Path,
    ) -> Result<Box<dyn ImportedData>> {
        let img = image::load_from_memory(buffer)
            .map_err(|e| crate::BakerError::Plugin(e.to_string()))?;
        Ok(Box::new(ImageImportedData {
            img,
            source_path: source_path.to_path_buf(),
        }))
    }
}

impl Importer for ImageImporter {
    fn name(&self) -> &str {
        "image"
    }

    fn source_extensions(&self) -> &[&str] {
        &["png", "jpg", "jpeg", "tga", "bmp", "exr"]
    }

    fn import(&self, source_path: &Path) -> Result<Box<dyn ImportedData>> {
        let img = image::open(source_path).map_err(|e| crate::BakerError::Plugin(e.to_string()))?;
        Ok(Box::new(ImageImportedData {
            img,
            source_path: source_path.to_path_buf(),
        }))
    }

    fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let imported = data
            .as_any()
            .downcast_ref::<ImageImportedData>()
            .ok_or_else(|| crate::BakerError::Pipeline("Invalid imported data type".to_string()))?;

        let (width, height) = imported.img.dimensions();

        let mut current_mip = {
            let mut rgba_f32 = Vec::with_capacity((width * height * 4) as usize);
            if self.options.is_srgb {
                for pixel in imported.img.to_rgba8().pixels() {
                    rgba_f32.push(Self::srgb_to_linear(pixel[0] as f32 / 255.0));
                    rgba_f32.push(Self::srgb_to_linear(pixel[1] as f32 / 255.0));
                    rgba_f32.push(Self::srgb_to_linear(pixel[2] as f32 / 255.0));
                    rgba_f32.push(pixel[3] as f32 / 255.0);
                }
            } else {
                for pixel in imported.img.to_rgba8().pixels() {
                    rgba_f32.push(pixel[0] as f32 / 255.0);
                    rgba_f32.push(pixel[1] as f32 / 255.0);
                    rgba_f32.push(pixel[2] as f32 / 255.0);
                    rgba_f32.push(pixel[3] as f32 / 255.0);
                }
            }
            rgba_f32
        };

        let mut mip_width = width;
        let mut mip_height = height;
        let mut all_pixel_data = Vec::new();
        let mut mip_count = 0;

        loop {
            let is_target_srgb = matches!(
                self.options.format,
                TextureFormat::R8G8B8A8_SRGB
                    | TextureFormat::BC1_RGB_SRGB
                    | TextureFormat::BC1_RGBA_SRGB
                    | TextureFormat::BC3_SRGB
                    | TextureFormat::BC7_SRGB
            );

            let rgba_u8: Vec<u8> = if is_target_srgb {
                current_mip
                    .chunks_exact(4)
                    .flat_map(|c| {
                        [
                            (Self::linear_to_srgb(c[0]).clamp(0.0, 1.0) * 255.0) as u8,
                            (Self::linear_to_srgb(c[1]).clamp(0.0, 1.0) * 255.0) as u8,
                            (Self::linear_to_srgb(c[2]).clamp(0.0, 1.0) * 255.0) as u8,
                            (c[3].clamp(0.0, 1.0) * 255.0) as u8,
                        ]
                    })
                    .collect()
            } else {
                current_mip
                    .iter()
                    .map(|&f| (f.clamp(0.0, 1.0) * 255.0) as u8)
                    .collect()
            };

            let surface = intel_tex_2::RgbaSurface {
                width: mip_width,
                height: mip_height,
                stride: mip_width * 4,
                data: &rgba_u8,
            };

            let compressed = match self.options.format {
                TextureFormat::BC1_RGB_UNORM | TextureFormat::BC1_RGB_SRGB => {
                    intel_tex_2::bc1::compress_blocks(&surface)
                }
                TextureFormat::BC3_UNORM | TextureFormat::BC3_SRGB => {
                    intel_tex_2::bc3::compress_blocks(&surface)
                }
                TextureFormat::BC4_UNORM | TextureFormat::BC4_SNORM => {
                    intel_tex_2::bc4::compress_blocks(&surface)
                }
                TextureFormat::BC5_UNORM | TextureFormat::BC5_SNORM => {
                    intel_tex_2::bc5::compress_blocks(&surface)
                }
                TextureFormat::BC7_UNORM | TextureFormat::BC7_SRGB => {
                    intel_tex_2::bc7::compress_blocks(
                        &intel_tex_2::bc7::opaque_ultra_fast_settings(),
                        &surface,
                    )
                }
                _ => rgba_u8,
            };

            all_pixel_data.extend_from_slice(&compressed);
            mip_count += 1;

            if !self.options.generate_mips || (mip_width <= 4 || mip_height <= 4) {
                if mip_width == 1 && mip_height == 1 {
                    break;
                }
            }

            let next_width = (mip_width / 2).max(1);
            let next_height = (mip_height / 2).max(1);

            let mut next_mip = Vec::with_capacity((next_width * next_height * 4) as usize);
            for y in 0..next_height {
                for x in 0..next_width {
                    for c in 0..4 {
                        let mut sum = 0.0;
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let sx = (x * 2 + dx).min(mip_width - 1);
                                let sy = (y * 2 + dy).min(mip_height - 1);
                                sum += current_mip[((sy * mip_width + sx) * 4 + c) as usize];
                            }
                        }
                        next_mip.push(sum / 4.0);
                    }
                }
            }

            current_mip = next_mip;
            mip_width = next_width;
            mip_height = next_height;

            if mip_width == 0 || mip_height == 0 {
                break;
            }
        }

        let header = TextureHeader {
            width,
            height,
            depth: 1,
            mip_levels: mip_count,
            array_layers: 1,
            format: self.options.format as u32,
            data_size: all_pixel_data.len() as u64,
        };

        let mut final_data = bytemuck::bytes_of(&header).to_vec();
        final_data.extend_from_slice(&all_pixel_data);

        let asset_id = Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            imported.source_path.to_string_lossy().as_bytes(),
        );

        Ok(vec![BakeOutput {
            asset_id,
            asset_type: TEXTURE_ASSET_TYPE,
            data: final_data,
            name: "texture".to_string(),
        }])
    }
}

struct ImageImportedData {
    img: image::DynamicImage,
    source_path: PathBuf,
}

impl ImportedData for ImageImportedData {
    fn source_path(&self) -> &Path {
        &self.source_path
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
