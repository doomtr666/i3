use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod asset;
pub mod error;
pub mod mesh;
pub mod scene_asset;
pub mod vfs;

pub use error::{IoError, Result};

/// Fixed-size header for all assets stored in .i3b bundles.
/// Total size: 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AssetHeader {
    pub magic: u64,             // 0..8   0x4933415353455400 ("I3ASSET\0")
    pub version: u32,           // 8..12  Current: 1
    pub compression: u32,       // 12..16 0: None, 1: Zstd, 2: GDeflate
    pub data_offset: u64,       // 16..24 Offset from start of .i3b
    pub data_size: u64,         // 24..32 Size of the blob (after compression)
    pub uncompressed_size: u64, // 32..40
    pub asset_type: [u8; 16],   // 40..56 Uuid
    pub _reserved: [u8; 8],     // 56..64 Padding to 64 bytes
}

impl AssetHeader {
    pub const MAGIC: u64 = 0x4933415353455400;
    pub const VERSION: u32 = 1;

    pub fn new(asset_type: Uuid, data_offset: u64, data_size: u64) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            compression: 0,
            data_offset,
            data_size,
            uncompressed_size: data_size,
            asset_type: asset_type.into_bytes(),
            _reserved: [0; 8],
        }
    }

    pub fn is_valid(&self) -> bool {
        self.magic == Self::MAGIC
    }
}

/// Catalog entry for an asset in a bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub asset_type: [u8; 16],
    pub offset: u64,
    pub size: u64,
    pub compression: u32,
    pub uncompressed_size: u64,
}

pub mod prelude;
