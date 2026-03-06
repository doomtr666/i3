use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IoError {
    #[error("IO error at {path}: {source}")]
    Os {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("VFS path not found: {0}")]
    NotFound(PathBuf),

    #[error("Invalid asset header magic")]
    InvalidMagic,

    #[error("Unsupported asset version: {0}")]
    UnsupportedVersion(u32),

    #[error("Catalog deserialization failed: {0}")]
    CatalogError(String),

    #[error("Asset type mismatch. Expected {expected}, found {found}")]
    TypeMismatch {
        expected: uuid::Uuid,
        found: uuid::Uuid,
    },

    #[error("Memory alignment error: expected 64KB")]
    AlignmentError,

    #[error("Invalid data: {message}")]
    InvalidData { message: String },

    #[error("Generic error: {0}")]
    Generic(String),
}

pub type Result<T> = std::result::Result<T, IoError>;
