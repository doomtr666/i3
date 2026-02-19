use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod asset;
pub mod error;
pub mod vfs;

pub use error::{IoError, Result};

/// Fixed-size header for all assets stored in .i3b bundles.
/// Repr(C) and 16-byte aligned for direct mapping and DirectStorage compatibility.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AssetHeader {
    pub magic: [u8; 8],       // "i3ASSET\0"
    pub version: u32,         // Current: 1
    pub asset_type: [u8; 16], // Uuid
    pub data_offset: u64,     // Offset from start of .i3b (aligned to 64KB)
    pub data_size: u64,       // Size of the blob (after compression)
    pub compression: u32,     // 0: None, 1: Zstd, 2: GDeflate
    pub uncompressed_size: u64,
    pub _reserved: [u8; 8], // For 64-byte padding & alignment
}

impl AssetHeader {
    pub const MAGIC: [u8; 8] = *b"i3ASSET\0";
    pub const VERSION: u32 = 1;

    pub fn new(asset_type: Uuid, data_offset: u64, data_size: u64) -> Self {
        Self {
            magic: Self::MAGIC,
            version: Self::VERSION,
            asset_type: asset_type.into_bytes(),
            data_offset,
            data_size,
            compression: 0,
            uncompressed_size: data_size,
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

pub mod prelude {
    pub use crate::AssetHeader;
    pub use crate::asset::{Asset, AssetHandle};
    pub use crate::error::{IoError, Result};
    pub use crate::vfs::{Vfs, VfsFile};
}
