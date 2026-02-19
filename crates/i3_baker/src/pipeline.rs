use crate::Result;
use std::path::{Path, PathBuf};

pub struct BakeContext {
    pub source_path: PathBuf,
    pub output_dir: PathBuf,
    pub dependencies: Vec<PathBuf>,
}

impl BakeContext {
    pub fn new(source: impl AsRef<Path>, output: impl AsRef<Path>) -> Self {
        Self {
            source_path: source.as_ref().to_path_buf(),
            output_dir: output.as_ref().to_path_buf(),
            dependencies: Vec::new(),
        }
    }

    pub fn add_dependency(&mut self, path: impl AsRef<Path>) {
        self.dependencies.push(path.as_ref().to_path_buf());
    }
}

pub struct BakeResult {
    pub blob: Vec<u8>,
    pub secondary_outputs: Vec<PathBuf>,
}

pub trait AssetPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn source_extension(&self) -> &str;
    fn output_extension(&self) -> &str;
    fn bake(&self, context: &mut BakeContext) -> Result<BakeResult>;
}

pub struct PipelineNode {
    pub plugin: Box<dyn AssetPlugin>,
    pub children: Vec<PipelineNode>,
}

impl PipelineNode {
    pub fn new(plugin: Box<dyn AssetPlugin>) -> Self {
        Self {
            plugin,
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, node: PipelineNode) {
        self.children.push(node);
    }
}
