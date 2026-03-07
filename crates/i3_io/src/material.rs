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
