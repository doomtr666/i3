pub mod error;
pub mod importers;
pub mod pipeline;
pub mod writer;

pub use error::{BakerError, Result};

pub mod prelude {
    pub use crate::error::{BakerError, Result};
    pub use crate::pipeline::{
        BakeContext, BakeOutput, Extractor, ImportedData, Importer,
    };
}
