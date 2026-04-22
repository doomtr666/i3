pub mod error;
pub mod importers;
pub mod manifest;
pub mod pipeline;
pub mod writer;

pub use error::{BakerError, Result};

pub use importers::HdrIblImporter;
pub use importers::ibl_bake::IblBakeOptions;
pub use importers::{NoiseImporter, NoiseManifestEntry, NoiseAlgorithm};
pub use manifest::ManifestBaker;

pub mod prelude {
    pub use crate::error::{BakerError, Result};
    pub use crate::pipeline::{
        BakeContext, BakeOutput, BundleBaker, Extractor, ImportedData, Importer,
    };
    pub use crate::importers::HdrIblImporter;
    pub use crate::importers::ibl_bake::IblBakeOptions;
    pub use crate::manifest::ManifestBaker;
}
