use crate::Result;
use crate::pipeline::{BakeContext, BakeOutput, ImportedData, Importer};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use std::path::{Path, PathBuf};

use super::image_importer::{ImageImportedData, ImageImporter, TextureImportOptions, TextureSemantic};

pub struct MaterialSetImporter {
    pub name: String,
    pub albedo: Option<PathBuf>,
    pub normal: Option<PathBuf>,
    pub roughness: Option<PathBuf>,
    pub metallic: Option<PathBuf>,
    pub occlusion: Option<PathBuf>,
}

pub struct MaterialSetImportedData {
    pub albedo_img: Option<DynamicImage>,
    pub normal_img: Option<DynamicImage>,
    pub orm_img: Option<DynamicImage>,
    pub source_path: PathBuf,
}

impl ImportedData for MaterialSetImportedData {
    fn source_path(&self) -> &Path {
        &self.source_path
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Importer for MaterialSetImporter {
    fn name(&self) -> &str {
        "material_set"
    }

    fn source_extensions(&self) -> &[&str] {
        &[]
    }

    fn import(&self, source_path: &Path) -> Result<Box<dyn ImportedData>> {
        let load_img = |p: Option<&PathBuf>| -> Option<DynamicImage> {
            p.and_then(|path| image::open(path).ok())
        };

        let albedo_img = load_img(self.albedo.as_ref());
        let normal_img = load_img(self.normal.as_ref());

        let rough = load_img(self.roughness.as_ref());
        let metal = load_img(self.metallic.as_ref());
        let ao = load_img(self.occlusion.as_ref());

        let mut orm_img = None;
        if rough.is_some() || metal.is_some() || ao.is_some() {
            let mut width = 0;
            let mut height = 0;

            for img in [&rough, &metal, &ao].into_iter().flatten() {
                let (w, h) = img.dimensions();
                if width == 0 {
                    width = w;
                    height = h;
                } else if width != w || height != h {
                    return Err(crate::BakerError::Pipeline(format!("ORM maps for material '{}' have different dimensions", self.name)));
                }
            }

            if width > 0 && height > 0 {
                let mut out_img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);
                for y in 0..height {
                    for x in 0..width {
                        let r = ao.as_ref().map(|i| i.get_pixel(x, y)[0]).unwrap_or(255);
                        let g = rough.as_ref().map(|i| i.get_pixel(x, y)[0]).unwrap_or(255);
                        let b = metal.as_ref().map(|i| i.get_pixel(x, y)[0]).unwrap_or(0);
                        out_img.put_pixel(x, y, Rgba([r, g, b, 255]));
                    }
                }
                orm_img = Some(DynamicImage::ImageRgba8(out_img));
            }
        }

        Ok(Box::new(MaterialSetImportedData {
            albedo_img,
            normal_img,
            orm_img,
            source_path: source_path.to_path_buf(),
        }))
    }

    fn extract(&self, data: &dyn ImportedData, ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let imported = data
            .as_any()
            .downcast_ref::<MaterialSetImportedData>()
            .ok_or_else(|| crate::BakerError::Pipeline("Invalid imported data type".to_string()))?;

        let mut outputs = Vec::new();

        let mut extract_map = |img: Option<&DynamicImage>, semantic: TextureSemantic, suffix: &str| {
            if let Some(i) = img {
                let mut virtual_path = imported.source_path.clone();
                virtual_path.set_file_name(format!("{}_{}.png", self.name, suffix));

                let img_data = ImageImportedData {
                    img: i.clone(),
                    source_path: virtual_path,
                };

                let importer = ImageImporter::new(TextureImportOptions {
                    semantic,
                    generate_mips: true,
                    format: None,
                });

                if let Ok(mut out) = importer.extract(&img_data, ctx) {
                    outputs.append(&mut out);
                }
            }
        };

        extract_map(imported.albedo_img.as_ref(), TextureSemantic::Albedo, "albedo");
        extract_map(imported.normal_img.as_ref(), TextureSemantic::Normal, "normal");
        extract_map(imported.orm_img.as_ref(), TextureSemantic::MetallicRoughness, "orm");

        Ok(outputs)
    }
}
