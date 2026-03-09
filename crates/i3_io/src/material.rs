use bytemuck::{Pod, Zeroable};
use uuid::{Uuid, uuid};

/// UUID for material assets
pub const MATERIAL_ASSET_TYPE: Uuid = uuid!("b2a5e4f1-9686-4f54-91be-547f89e62903");

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialHeader {
    /// albedo texture UUID
    pub albedo_texture: [u8; 16],
    /// normal map UUID
    pub normal_texture: [u8; 16],
    /// metallic-roughness map UUID
    pub metallic_roughness_texture: [u8; 16],
    /// emissive map UUID
    pub emissive_texture: [u8; 16],

    /// base color factor
    pub base_color_factor: [f32; 4],
    /// metallic factor
    pub metallic_factor: f32,
    /// roughness factor
    pub roughness_factor: f32,
    /// emissive factor
    pub emissive_factor: [f32; 3],

    pub alpha_cutoff: f32,
    pub _padding: [u8; 20], // Pad to 128 bytes or similar?
}

impl MaterialHeader {
    pub const MAGIC: [u8; 4] = *b"I3MT";
}

/// A parsed Material asset ready for rendering.
#[derive(Debug, Clone)]
pub struct MaterialAsset {
    pub header: Option<MaterialHeader>,
}

impl crate::asset::Asset for MaterialAsset {
    const ASSET_TYPE_ID: [u8; 16] = *MATERIAL_ASSET_TYPE.as_bytes();

    fn load(_header: &crate::AssetHeader, data: &[u8]) -> crate::Result<Self> {
        if data.len() < std::mem::size_of::<MaterialHeader>() {
            return Err(crate::IoError::Generic(
                "Material asset too small".to_string(),
            ));
        }

        // Since the header is Pod, we can just cast it
        let header =
            *bytemuck::from_bytes::<MaterialHeader>(&data[..std::mem::size_of::<MaterialHeader>()]);

        Ok(Self {
            header: Some(header),
        })
    }
}
