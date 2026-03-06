use crate::{AssetHeader, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

pub const ASSET_STATE_UNLOADED: u8 = 0;
pub const ASSET_STATE_LOADING: u8 = 1;
pub const ASSET_STATE_LOADED: u8 = 2;
pub const ASSET_STATE_ERROR: u8 = 3;

pub trait Asset: Sized + Send + Sync + 'static {
    const ASSET_TYPE_ID: [u8; 16];
    fn load(header: &AssetHeader, data: &[u8]) -> Result<Self>;
}

pub struct AssetInner<T> {
    pub state: AtomicU8,
    pub asset: Option<T>,
}

pub struct AssetHandle<T> {
    pub(crate) inner: Arc<AssetInner<T>>,
}

impl<T> Clone for AssetHandle<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> AssetHandle<T> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AssetInner {
                state: AtomicU8::new(ASSET_STATE_UNLOADED),
                asset: None,
            }),
        }
    }

    pub fn state(&self) -> u8 {
        self.inner.state.load(Ordering::Acquire)
    }

    pub fn is_loaded(&self) -> bool {
        self.state() == ASSET_STATE_LOADED
    }

    pub fn get(&self) -> Option<&T> {
        if self.is_loaded() {
            self.inner.asset.as_ref()
        } else {
            None
        }
    }
}

pub struct AssetLoader {
    vfs: Arc<crate::vfs::Vfs>,
}

impl AssetLoader {
    pub fn new(vfs: Arc<crate::vfs::Vfs>) -> Self {
        Self { vfs }
    }

    pub fn load<T: Asset>(&self, path: impl AsRef<std::path::Path>) -> AssetHandle<T> {
        let handle = AssetHandle::new();
        let path = path.as_ref().to_path_buf();
        let vfs = self.vfs.clone();
        let handle_clone = handle.clone();

        rayon::spawn(move || {
            handle_clone
                .inner
                .state
                .store(ASSET_STATE_LOADING, Ordering::Release);

            let result = (|| -> Result<T> {
                let mut vfs_file = vfs.open(&path)?;

                if let Some(slice) = vfs_file.as_slice() {
                    // Fast path: Zero-copy via bytemuck
                    let header_size = std::mem::size_of::<AssetHeader>();
                    if slice.len() < header_size {
                        return Err(crate::IoError::InvalidData {
                            message: "Asset too small for header".to_string(),
                        });
                    }

                    let header: &AssetHeader = bytemuck::from_bytes(&slice[..header_size]);
                    let data = &slice[header_size..];

                    T::load(header, data)
                } else {
                    // Fallback: Read to memory
                    let mut full_data = Vec::new();
                    vfs_file.read_to_end(&mut full_data).map_err(|e| {
                        crate::error::IoError::Os {
                            path: path.clone(),
                            source: e,
                        }
                    })?;

                    let header_size = std::mem::size_of::<AssetHeader>();
                    if full_data.len() < header_size {
                        return Err(crate::IoError::InvalidData {
                            message: "Asset too small for header".to_string(),
                        });
                    }

                    let header: &AssetHeader = bytemuck::from_bytes(&full_data[..header_size]);
                    let data = &full_data[header_size..];

                    T::load(header, data)
                }
            })();

            match result {
                Ok(asset) => {
                    // Safe because we are the only ones writing to 'asset' while in LOADING state
                    let inner =
                        unsafe { &mut *(Arc::as_ptr(&handle_clone.inner) as *mut AssetInner<T>) };
                    inner.asset = Some(asset);
                    handle_clone
                        .inner
                        .state
                        .store(ASSET_STATE_LOADED, Ordering::Release);
                }
                Err(e) => {
                    tracing::error!("Failed to load asset {}: {}", path.display(), e);
                    handle_clone
                        .inner
                        .state
                        .store(ASSET_STATE_ERROR, Ordering::Release);
                }
            }
        });

        handle
    }
}
