use uuid::Uuid;

pub mod asset;
pub mod error;
pub mod material;
pub mod mesh;
pub mod pipeline_asset;
pub mod scene_asset;
pub mod texture;
pub mod vfs;

pub use error::{IoError, Result};

/// Fixed-size header for all assets stored in .i3b bundles.
/// Total size: 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default, PartialEq)]
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

/// Catalog file header for .i3c files.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CatalogHeader {
    pub magic: u64,   // 0x4933434154000000 ("I3CAT\0")
    pub version: u32, // Current: 1
    pub count: u32,   // Number of entries following the header
}

impl CatalogHeader {
    pub const MAGIC: u64 = 0x4933434154000000;
    pub const VERSION: u32 = 1;
}

/// Catalog entry for an asset in a bundle.
/// Fixed size (128 bytes) for O(1) mmap casting.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CatalogEntry {
    pub asset_id: [u8; 16],     // 0..16
    pub asset_type: [u8; 16],   // 16..32
    pub offset: u64,            // 32..40
    pub size: u64,              // 40..48
    pub uncompressed_size: u64, // 48..56
    pub compression: u32,       // 56..60
    pub _padding: u32,          // 60..64
    pub name: [u8; 64],         // 64..128
}

impl CatalogEntry {
    pub fn name(&self) -> &str {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(64);
        std::str::from_utf8(&self.name[..end]).unwrap_or("unknown")
    }
}

pub mod prelude;
