use uuid::{Uuid, uuid};

pub const IBL_ASSET_TYPE: Uuid = uuid!("4a7b1c2d-3e4f-5a6b-7c8d-9e0f1a2b3c4d");

/// Données soleil extraites automatiquement du HDR.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IblSunData {
    pub direction: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
    pub _pad: f32,
}

/// Header sérialisé en tête du BakeOutput IBL.
/// Layout binaire : IblHeader + brdf_lut_data + irradiance_data + prefiltered_data + env_data
/// L'asset est self-contained : aucune référence UUID externe nécessaire au runtime.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IblHeader {
    // BRDF LUT : R16G16_SFLOAT, 256x256, 1 mip
    pub lut_width: u32,
    pub lut_height: u32,
    pub lut_format: u32,       // TextureFormat::R16G16_SFLOAT as u32 = 10
    pub lut_data_size: u32,

    // Irradiance equirectangular : R11G11B10_UFLOAT, 128x64, 1 mip
    pub irr_width: u32,
    pub irr_height: u32,
    pub irr_format: u32,       // TextureFormat::R11G11B10_UFLOAT as u32 = 11
    pub irr_data_size: u32,

    // Pre-filtered equirectangular : R11G11B10_UFLOAT, 512x256, 6 mips
    pub pref_width: u32,
    pub pref_height: u32,
    pub pref_format: u32,      // TextureFormat::R11G11B10_UFLOAT as u32 = 11
    pub pref_mip_levels: u32,
    pub pref_data_size: u32,

    // Equirect HDR compressé : BC6H_UF16, résolution originale, 1 mip
    pub env_width: u32,
    pub env_height: u32,
    pub env_format: u32,       // TextureFormat::BC6H_UF16 as u32 = 12
    pub env_data_size: u32,

    /// Facteur de conversion HDR brut → unités physiques (lux).
    /// Appliqué à sun_intensity au bake. Le renderer multiplie l'IBL ambient par cette valeur.
    pub intensity_scale: f32,

    pub _pad: [u32; 3],

    pub sun: IblSunData,
}

pub struct IblAsset {
    pub header: IblHeader,
    pub data: Vec<u8>, // brdf_lut || irradiance || prefiltered || env_equirect (concaténés)
}

impl crate::asset::Asset for IblAsset {
    const ASSET_TYPE_ID: [u8; 16] = *IBL_ASSET_TYPE.as_bytes();

    fn load(_header: &crate::AssetHeader, data: &[u8]) -> crate::error::Result<Self> {
        let hdr_size = std::mem::size_of::<IblHeader>();
        if data.len() < hdr_size {
            return Err(crate::error::IoError::Generic("IBL data too small for header".into()));
        }
        let ibl_header: IblHeader = bytemuck::pod_read_unaligned(&data[..hdr_size]);
        Ok(Self { header: ibl_header, data: data[hdr_size..].to_vec() })
    }
}
