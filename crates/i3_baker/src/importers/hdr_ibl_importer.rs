use std::path::{Path, PathBuf};
use uuid::Uuid;
use crate::pipeline::{ImportedData, Importer, BakeOutput, BakeContext};
use crate::Result;
use crate::importers::ibl_bake::*;
use std::f32::consts::PI;
use i3_io::ibl::{IblHeader, IBL_ASSET_TYPE};

pub struct HdrImportedData {
    pub source_path: PathBuf,
    pub pixels: Vec<[f32; 3]>,
    pub width: u32,
    pub height: u32,
}

impl ImportedData for HdrImportedData {
    fn source_path(&self) -> &Path { &self.source_path }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

pub struct HdrIblImporter {
    pub options: IblBakeOptions,
}

impl Default for HdrIblImporter {
    fn default() -> Self { Self { options: IblBakeOptions::default() } }
}

impl Importer for HdrIblImporter {
    fn name(&self) -> &str { "HdrIblImporter" }
    fn source_extensions(&self) -> &[&str] { &["hdr", "exr"] }

    fn import(&self, source_path: &Path) -> Result<Box<dyn ImportedData>> {
        let img = image::open(source_path)
            .map_err(|e| crate::BakerError::Pipeline(e.to_string()))?
            .into_rgb32f();
        let width = img.width();
        let height = img.height();
        let pixels: Vec<[f32; 3]> = img.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        Ok(Box::new(HdrImportedData { source_path: source_path.to_path_buf(), pixels, width, height }))
    }

    fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let hdr = data.as_any().downcast_ref::<HdrImportedData>().expect("Invalid imported data type");

        // Calcul du seuil solaire
        let sun_threshold = if self.options.extract_sun {
            compute_sun_threshold(&hdr.pixels)
        } else {
            f32::MAX
        };

        // Bake les 3 maps
        let lut_data   = bake_brdf_lut(256, 256, 1024);
        let irr_data   = bake_irradiance(&hdr.pixels, hdr.width, hdr.height, 64, sun_threshold);
        let pref_data  = bake_prefiltered(&hdr.pixels, hdr.width, hdr.height, 256, 6, 512, sun_threshold);

        // Extraction soleil — retourne IblSunData::zeroed() si threshold == f32::MAX
        let mut sun = extract_sun(&hdr.pixels, hdr.width, hdr.height, sun_threshold);

        // Calibration du ratio direct/ambient si demandé.
        // sun_dir_toward = direction "vers le soleil" (opposée à la convention renderer).
        if let Some(ratio) = self.options.sun_strength_ratio {
            let sun_dir_toward = [-sun.direction[0], -sun.direction[1], -sun.direction[2]];
            let irr = irradiance_lum_at(&hdr.pixels, hdr.width, hdr.height,
                                        sun_dir_toward, sun_threshold);
            // Sur surface Lambertienne face au soleil (NdotL=1) :
            //   direct = sun_intensity × albedo/π
            //   ambient = irr × albedo
            //   direct/ambient = ratio ⟹ sun_intensity = ratio × irr × π
            sun.intensity = ratio * irr * PI;
            tracing::info!("IBL sun calibration: irr_at_sun={:.4}, ratio={}, sun_intensity={:.2}",
                irr, ratio, sun.intensity);
        }

        sun.intensity *= self.options.intensity_scale;

        // Compression equirect
        let env_data = compress_env_bc6h(&hdr.pixels, hdr.width, hdr.height);

        // Header
        let header = IblHeader {
            lut_width: 256, lut_height: 256,
            lut_format: 10, // R16G16_SFLOAT
            lut_data_size: lut_data.len() as u32,
            irr_width: 64, irr_height: 64,
            irr_format: 11, // R11G11B10_UFLOAT
            irr_data_size: irr_data.len() as u32,
            pref_width: 256, pref_height: 256,
            pref_format: 11,
            pref_mip_levels: 6,
            pref_data_size: pref_data.len() as u32,
            env_width: hdr.width, env_height: hdr.height,
            env_format: 12, // BC6H_UF16
            env_data_size: env_data.len() as u32,
            intensity_scale: self.options.intensity_scale,
            _pad: [0; 3],
            sun,
        };

        // Sérialisation
        let mut payload = bytemuck::bytes_of(&header).to_vec();
        payload.extend_from_slice(&lut_data);
        payload.extend_from_slice(&irr_data);
        payload.extend_from_slice(&pref_data);
        payload.extend_from_slice(&env_data);

        let asset_id = Uuid::new_v5(
            &Uuid::NAMESPACE_URL,
            hdr.source_path.to_string_lossy().as_bytes(),
        );

        Ok(vec![BakeOutput {
            asset_id,
            asset_type: IBL_ASSET_TYPE,
            data: payload,
            name: hdr.source_path.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
        }])
    }
}
