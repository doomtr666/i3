use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BakerError {
    #[error("IO error at {path}: {source}")]
    Os {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Plugin error: {0}")]
    Plugin(String),

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("i3_io error: {0}")]
    Io(#[from] i3_io::IoError),

    #[error("Pipeline error: {0}")]
    Pipeline(String),
}

pub type Result<T> = std::result::Result<T, BakerError>;
