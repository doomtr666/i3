use uuid::{Uuid, uuid};

/// UUID for texture assets
pub const TEXTURE_ASSET_TYPE: Uuid = uuid!("c7a5e4f1-9686-4f54-91be-547f89e62902");

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    Undefined = 0,
    R8G8B8A8_UNORM = 1,
    R8G8B8A8_SRGB = 2,
    BC1_RGB_UNORM = 101,
    BC1_RGB_SRGB = 102,
    BC1_RGBA_UNORM = 103,
    BC1_RGBA_SRGB = 104,
    BC3_UNORM = 105,
    BC3_SRGB = 106,
    BC4_UNORM = 107,
    BC4_SNORM = 108,
    BC5_UNORM = 109,
    BC5_SNORM = 110,
    BC7_UNORM = 111,
    BC7_SRGB = 112,
    R16G16_SFLOAT = 10,
    R11G11B10_UFLOAT = 11,
    BC6H_UF16 = 12,
    R16G16B16A16_UNORM = 14,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextureHeader {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub format: u32,    // Raw value of TextureFormat
    pub data_size: u64, // Size of all mips/layers combined
}

impl TextureHeader {
    pub const MAGIC: [u8; 4] = *b"I3TX";
}

pub struct TextureAsset {
    pub header: TextureHeader,
    pub data: Vec<u8>,
}

impl crate::asset::Asset for TextureAsset {
    const ASSET_TYPE_ID: [u8; 16] = *TEXTURE_ASSET_TYPE.as_bytes();

    fn load(_header: &crate::AssetHeader, data: &[u8]) -> crate::error::Result<Self> {
        if data.len() < std::mem::size_of::<TextureHeader>() {
            return Err(crate::error::IoError::Generic("Invalid formatting".into()));
        }

        let header_size = std::mem::size_of::<TextureHeader>();
        let tex_header: TextureHeader = bytemuck::pod_read_unaligned(&data[..header_size]);

        let tex_data = data[std::mem::size_of::<TextureHeader>()..].to_vec();

        Ok(Self {
            header: tex_header,
            data: tex_data,
        })
    }
}
