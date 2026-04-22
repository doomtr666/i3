use bytemuck::{Pod, Zeroable};
use uuid::{Uuid, uuid};

pub const NOISE_ASSET_TYPE: Uuid = uuid!("7f3a1b2c-4d5e-6f70-8192-a3b4c5d6e7f8");

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NoiseAssetHeader {
    pub width:     u32,
    pub height:    u32,
    pub channels:  u32,  // 4 = RGBA
    pub format:    u32,  // TextureFormat discriminant (R16G16B16A16_UNORM = 14)
    pub seed:      u64,
    pub algorithm: u32,  // 0 = WhiteNoise, 1 = BlueNoise
    pub sigma:     f32,  // blue-noise filter width in normalised freq [0,1]
    pub data_size: u64,
    pub _pad:      [u8; 24],
}
// Total: 4+4+4+4+8+4+4+8+24 = 64 bytes

pub struct NoiseAsset {
    pub header: NoiseAssetHeader,
    pub data:   Vec<u8>,  // width × height × 4 channels × 2 bytes (u16 LE, UNORM)
}

impl crate::asset::Asset for NoiseAsset {
    const ASSET_TYPE_ID: [u8; 16] = *NOISE_ASSET_TYPE.as_bytes();

    fn load(_header: &crate::AssetHeader, data: &[u8]) -> crate::error::Result<Self> {
        let header_size = std::mem::size_of::<NoiseAssetHeader>();
        if data.len() < header_size {
            return Err(crate::error::IoError::Generic("NoiseAsset: data too short".into()));
        }
        let header: NoiseAssetHeader = bytemuck::pod_read_unaligned(&data[..header_size]);
        let pixels = data[header_size..].to_vec();
        Ok(Self { header, data: pixels })
    }
}
