use crate::{CatalogEntry, Result};
use std::io::{Read, Seek};
use std::path::Path;
use std::sync::Arc;

pub trait VfsFile: Read + Seek + Send + Sync {
    fn size(&self) -> u64;
    /// Allows zero-copy mmap access if supported by the backend.
    fn as_slice(&self) -> Option<&[u8]>;
}

pub trait VfsBackend: Send + Sync {
    fn exists(&self, path: &Path) -> bool;
    fn open(&self, path: &Path) -> Result<Box<dyn VfsFile>>;
    fn open_by_uuid(&self, _uuid: &uuid::Uuid) -> Option<Result<Box<dyn VfsFile>>> {
        None
    }
}

pub struct Vfs {
    backends: Vec<Box<dyn VfsBackend>>,
}

impl Vfs {
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    pub fn mount(&mut self, backend: Box<dyn VfsBackend>) {
        self.backends.push(backend);
    }

    pub fn exists(&self, path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        self.backends.iter().any(|b| b.exists(path))
    }

    pub fn open(&self, path: impl AsRef<Path>) -> Result<Box<dyn VfsFile>> {
        let path = path.as_ref();
        for backend in &self.backends {
            if backend.exists(path) {
                return backend.open(path);
            }
        }
        Err(crate::error::IoError::NotFound(path.to_path_buf()))
    }

    pub fn open_by_uuid(&self, uuid: &uuid::Uuid) -> Result<Box<dyn VfsFile>> {
        for backend in &self.backends {
            if let Some(result) = backend.open_by_uuid(uuid) {
                return result;
            }
        }
        Err(crate::error::IoError::Generic(format!(
            "Asset UUID {} not found in any backend",
            uuid
        )))
    }
}

pub struct PhysicalFile {
    file: std::fs::File,
    mmap: Option<memmap2::Mmap>,
}

impl Read for PhysicalFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }
}

impl Seek for PhysicalFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.file.seek(pos)
    }
}

impl VfsFile for PhysicalFile {
    fn size(&self) -> u64 {
        self.file.metadata().map(|m| m.len()).unwrap_or(0)
    }

    fn as_slice(&self) -> Option<&[u8]> {
        self.mmap.as_ref().map(|m| &m[..])
    }
}

pub struct PhysicalBackend {
    root: std::path::PathBuf,
}

impl PhysicalBackend {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }
}

impl VfsBackend for PhysicalBackend {
    fn exists(&self, path: &Path) -> bool {
        self.root.join(path).exists()
    }

    fn open(&self, path: &Path) -> Result<Box<dyn VfsFile>> {
        let full_path = self.root.join(path);
        let file = std::fs::File::open(&full_path).map_err(|e| crate::error::IoError::Os {
            path: full_path.clone(),
            message: e.to_string(),
        })?;

        let mmap = unsafe { memmap2::Mmap::map(&file).ok() };

        Ok(Box::new(PhysicalFile { file, mmap }))
    }
}

pub struct BundleFile {
    mmap: Arc<memmap2::Mmap>,
    offset: u64,
    size: u64,
    pos: u64,
}

impl Read for BundleFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let remaining = self.size - self.pos;
        if remaining == 0 {
            return Ok(0);
        }
        let to_read = (buf.len() as u64).min(remaining) as usize;
        let start = (self.offset + self.pos) as usize;
        buf[..to_read].copy_from_slice(&self.mmap[start..start + to_read]);
        self.pos += to_read as u64;
        Ok(to_read)
    }
}

impl Seek for BundleFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            std::io::SeekFrom::Start(p) => p as i64,
            std::io::SeekFrom::End(p) => self.size as i64 + p,
            std::io::SeekFrom::Current(p) => self.pos as i64 + p,
        };
        if new_pos < 0 || new_pos > self.size as i64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid seek",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

impl VfsFile for BundleFile {
    fn size(&self) -> u64 {
        self.size
    }

    fn as_slice(&self) -> Option<&[u8]> {
        let start = (self.offset) as usize;
        let end = (self.offset + self.size) as usize;
        Some(&self.mmap[start..end])
    }
}

pub struct BundleBackend {
    by_name: std::collections::HashMap<String, CatalogEntry>,
    by_uuid: std::collections::HashMap<uuid::Uuid, CatalogEntry>,
    blob_mmap: Arc<memmap2::Mmap>,
}

impl BundleBackend {
    pub fn mount(catalog_path: impl AsRef<Path>, blob_path: impl AsRef<Path>) -> Result<Self> {
        let catalog_file =
            std::fs::File::open(&catalog_path).map_err(|e| crate::error::IoError::Os {
                path: catalog_path.as_ref().to_path_buf(),
                message: e.to_string(),
            })?;

        let catalog_mmap = unsafe {
            memmap2::Mmap::map(&catalog_file).map_err(|e| crate::error::IoError::Os {
                path: catalog_path.as_ref().to_path_buf(),
                message: e.to_string(),
            })?
        };

        let header_size = std::mem::size_of::<crate::CatalogHeader>();
        if catalog_mmap.len() < header_size {
            return Err(crate::error::IoError::InvalidData {
                message: "Catalog too small for header".to_string(),
            });
        }

        let header: &crate::CatalogHeader = bytemuck::from_bytes(&catalog_mmap[..header_size]);
        if header.magic != crate::CatalogHeader::MAGIC {
            return Err(crate::error::IoError::InvalidData {
                message: "Invalid catalog magic".to_string(),
            });
        }

        let entries_size = header.count as usize * std::mem::size_of::<CatalogEntry>();
        if catalog_mmap.len() < header_size + entries_size {
            return Err(crate::error::IoError::InvalidData {
                message: "Catalog truncated".to_string(),
            });
        }

        let entries: &[CatalogEntry] =
            bytemuck::cast_slice(&catalog_mmap[header_size..header_size + entries_size]);

        let mut by_name = std::collections::HashMap::with_capacity(header.count as usize);
        let mut by_uuid = std::collections::HashMap::with_capacity(header.count as usize);

        for entry in entries {
            by_name.insert(entry.name().to_string(), *entry);
            by_uuid.insert(uuid::Uuid::from_bytes(entry.asset_id), *entry);
        }

        let blob_file = std::fs::File::open(&blob_path).map_err(|e| crate::error::IoError::Os {
            path: blob_path.as_ref().to_path_buf(),
            message: e.to_string(),
        })?;

        let blob_mmap = unsafe {
            memmap2::Mmap::map(&blob_file).map_err(|e| crate::error::IoError::Os {
                path: blob_path.as_ref().to_path_buf(),
                message: e.to_string(),
            })?
        };

        Ok(Self {
            by_name,
            by_uuid,
            blob_mmap: Arc::new(blob_mmap),
        })
    }

    pub fn open_by_uuid(&self, id: &uuid::Uuid) -> Result<Box<dyn crate::vfs::VfsFile>> {
        let entry = self.by_uuid.get(id).ok_or_else(|| {
            crate::error::IoError::Generic(format!("Asset UUID {} not found", id))
        })?;

        Ok(Box::new(BundleFile {
            mmap: self.blob_mmap.clone(),
            offset: entry.offset,
            size: entry.size,
            pos: 0,
        }))
    }
}

impl VfsBackend for BundleBackend {
    fn exists(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        self.by_name.contains_key(path_str.as_ref())
    }

    fn open(&self, path: &Path) -> Result<Box<dyn VfsFile>> {
        let path_str = path.to_string_lossy();
        let entry = self
            .by_name
            .get(path_str.as_ref())
            .ok_or_else(|| crate::error::IoError::NotFound(path.to_path_buf()))?;

        Ok(Box::new(BundleFile {
            mmap: self.blob_mmap.clone(),
            offset: entry.offset,
            size: entry.size,
            pos: 0,
        }))
    }

    fn open_by_uuid(&self, id: &uuid::Uuid) -> Option<Result<Box<dyn VfsFile>>> {
        match self.by_uuid.get(id) {
            Some(entry) => Some(Ok(Box::new(BundleFile {
                mmap: self.blob_mmap.clone(),
                offset: entry.offset,
                size: entry.size,
                pos: 0,
            }))),
            None => None,
        }
    }
}
