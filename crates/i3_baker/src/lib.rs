pub mod error;
pub mod pipeline;
pub mod registry;
pub mod writer;

pub use error::{BakerError, Result};

pub mod prelude {
    pub use crate::error::{BakerError, Result};
    pub use crate::pipeline::{AssetPlugin, BakeContext, BakeResult, PipelineNode};
}
