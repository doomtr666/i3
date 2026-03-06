use crate::Result;
use crate::pipeline::BakeOutput;
use i3_io::{AssetHeader, CatalogEntry};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

pub struct BundleWriter {
    blob_file: std::fs::File,
    catalog: HashMap<String, CatalogEntry>,
    current_offset: u64,
}

impl BundleWriter {
    pub fn new(blob_path: impl AsRef<Path>) -> Result<Self> {
        let blob_file =
            std::fs::File::create(blob_path).map_err(|e| crate::error::BakerError::Os {
                path: Path::new("blob").to_path_buf(),
                source: e,
            })?;

        Ok(Self {
            blob_file,
            catalog: HashMap::new(),
            current_offset: 0,
        })
    }

    pub fn add_bake_output(&mut self, output: &BakeOutput) -> Result<()> {
        let header = AssetHeader::new(output.asset_type, 0, output.data.len() as u64);
        self.add_asset(&output.name, &header, &output.data)
    }

    pub fn add_asset(&mut self, id: &str, header: &AssetHeader, data: &[u8]) -> Result<()> {
        // Align each asset start on 64KB for DirectStorage/mmap performance
        let padding = (65536 - (self.current_offset % 65536)) % 65536;
        if padding > 0 {
            let p = vec![0u8; padding as usize];
            self.blob_file
                .write_all(&p)
                .map_err(|e| crate::error::BakerError::Os {
                    path: Path::new("blob").to_path_buf(),
                    source: e,
                })?;
            self.current_offset += padding;
        }

        let start_offset = self.current_offset;

        // Update header with the correct offset within the blob
        let mut final_header = *header;
        final_header.data_offset = start_offset;

        // Write header (binary direct via bytemuck)
        let header_bytes: &[u8] = bytemuck::bytes_of(&final_header);
        assert_eq!(header_bytes.len(), 64);

        self.blob_file
            .write_all(header_bytes)
            .map_err(|e| crate::error::BakerError::Os {
                path: Path::new("blob").to_path_buf(),
                source: e,
            })?;
        self.current_offset += header_bytes.len() as u64;

        // Write data
        self.blob_file
            .write_all(data)
            .map_err(|e| crate::error::BakerError::Os {
                path: Path::new("blob").to_path_buf(),
                source: e,
            })?;
        self.current_offset += data.len() as u64;

        // Add to catalog
        self.catalog.insert(
            id.to_string(),
            CatalogEntry {
                asset_type: final_header.asset_type,
                offset: start_offset,
                size: (self.current_offset - start_offset),
                compression: final_header.compression,
                uncompressed_size: final_header.uncompressed_size,
            },
        );

        Ok(())
    }

    pub fn finish(self, catalog_path: impl AsRef<Path>) -> Result<()> {
        let catalog_bytes = bincode::serialize(&self.catalog)
            .map_err(|e| i3_io::IoError::Generic(e.to_string()))?;

        std::fs::write(catalog_path, catalog_bytes).map_err(|e| crate::error::BakerError::Os {
            path: Path::new("catalog").to_path_buf(),
            source: e,
        })?;

        Ok(())
    }
}
