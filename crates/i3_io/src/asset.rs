use crate::{AssetHeader, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use uuid::Uuid;

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
                let mut data = Vec::new();

                if let Some(slice) = vfs_file.as_slice() {
                    // Fast path: Zero-copy
                    data = slice.to_vec(); // Simplified for now, real implementation would use slices
                } else {
                    vfs_file
                        .read_to_end(&mut data)
                        .map_err(|e| crate::error::IoError::Os {
                            path: path.clone(),
                            source: e,
                        })?;
                }

                // Header validation would go here
                // For now, assume raw data is the asset
                // In a real i3 asset, we'd parse the AssetHeader first

                // Constructing an empty header for the load trait
                let header = crate::AssetHeader::new(
                    Uuid::from_bytes(T::ASSET_TYPE_ID),
                    0,
                    data.len() as u64,
                );

                T::load(&header, &data)
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
