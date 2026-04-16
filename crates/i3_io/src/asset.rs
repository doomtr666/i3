use crate::{AssetHeader, Result};
use std::sync::Arc;
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
    pub sync: std::sync::Mutex<(u8, Option<std::result::Result<Arc<T>, crate::IoError>>)>,
    pub condvar: std::sync::Condvar,
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
                sync: std::sync::Mutex::new((ASSET_STATE_UNLOADED, None)),
                condvar: std::sync::Condvar::new(),
            }),
        }
    }

    pub fn state(&self) -> u8 {
        self.inner.sync.lock().unwrap().0
    }

    pub fn is_loaded(&self) -> bool {
        self.state() == ASSET_STATE_LOADED
    }

    pub fn get(&self) -> Option<Arc<T>> {
        let lock = self.inner.sync.lock().unwrap();
        if lock.0 == ASSET_STATE_LOADED {
            match lock.1.as_ref().unwrap() {
                Ok(asset) => Some(asset.clone()),
                Err(_) => None,
            }
        } else {
            None
        }
    }

    pub fn wait_loaded(&self) -> Result<Arc<T>> {
        let mut lock = self.inner.sync.lock().unwrap();
        lock = self
            .inner
            .condvar
            .wait_while(lock, |(state, _)| {
                *state == ASSET_STATE_LOADING || *state == ASSET_STATE_UNLOADED
            })
            .unwrap();

        if lock.0 == ASSET_STATE_LOADED {
            match lock.1.as_ref().unwrap() {
                Ok(asset) => Ok(asset.clone()),
                Err(e) => Err(e.clone()),
            }
        } else {
            match lock.1.as_ref() {
                Some(Err(e)) => Err(e.clone()),
                _ => Err(crate::IoError::Generic(
                    "Asset failed to load (Inconsistent state)".to_string(),
                )),
            }
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

    pub fn vfs(&self) -> Arc<crate::vfs::Vfs> {
        self.vfs.clone()
    }

    pub fn list_assets<T: Asset>(&self) -> Vec<String> {
        self.vfs.list_by_type(&T::ASSET_TYPE_ID)
    }

    pub fn load<T: Asset>(&self, path: impl AsRef<std::path::Path>) -> AssetHandle<T> {
        let handle = AssetHandle::new();
        let path = path.as_ref().to_path_buf();
        let vfs = self.vfs.clone();
        let handle_clone = handle.clone();
        let start_time = std::time::Instant::now();

        rayon::spawn(move || {
            {
                let mut lock = handle_clone.inner.sync.lock().unwrap();
                lock.0 = ASSET_STATE_LOADING;
            }

            let result = (|| -> Result<T> {
                let vfs_file = vfs.open(&path)?;
                Self::load_internal(vfs_file)
            })();

            Self::finalize_load(handle_clone, result, &path.to_string_lossy(), start_time);
        });

        handle
    }

    pub fn load_by_uuid<T: Asset>(&self, uuid: &Uuid) -> Result<AssetHandle<T>> {
        let handle = AssetHandle::new();
        let vfs = self.vfs.clone();
        let handle_clone = handle.clone();
        let uuid = *uuid;
        let start_time = std::time::Instant::now();

        rayon::spawn(move || {
            {
                let mut lock = handle_clone.inner.sync.lock().unwrap();
                lock.0 = ASSET_STATE_LOADING;
            }

            let result = (|| -> Result<T> {
                let vfs_file = vfs.open_by_uuid(&uuid)?;
                Self::load_internal(vfs_file)
            })();

            Self::finalize_load(handle_clone, result, &uuid.to_string(), start_time);
        });

        Ok(handle)
    }

    fn load_internal<T: Asset>(mut vfs_file: Box<dyn crate::vfs::VfsFile>) -> Result<T> {
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
            if let Err(e) = vfs_file.read_to_end(&mut full_data) {
                return Err(crate::error::IoError::Generic(format!(
                    "Failed to read asset data: {}",
                    e
                )));
            }

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
    }

    fn finalize_load<T>(
        handle: AssetHandle<T>,
        result: Result<T>,
        identity: &str,
        start_time: std::time::Instant,
    ) {
        let (new_state, log_err) = match &result {
            Ok(_) => {
                let duration = start_time.elapsed();
                tracing::debug!(
                    "Loaded asset {} in {}ms",
                    identity,
                    duration.as_secs_f32() * 1000.0
                );
                (ASSET_STATE_LOADED, None)
            }
            Err(e) => (
                ASSET_STATE_ERROR,
                Some(format!("Failed to load asset {}: {}", identity, e)),
            ),
        };

        {
            let mut lock = handle.inner.sync.lock().unwrap();
            lock.0 = new_state;
            lock.1 = Some(result.map(Arc::new));
        }

        handle.inner.condvar.notify_all();

        if let Some(err) = log_err {
            tracing::error!("{}", err);
        }
    }
}
