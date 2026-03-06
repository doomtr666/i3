//! Baker pipeline architecture.
//!
//! Defines the Importer → Extractor model for asset baking:
//! - Importers parse source files (e.g., AssimpImporter for glTF/FBX/OBJ)
//! - Extractors produce typed outputs from imported data (e.g., MeshExtractor, SceneExtractor)

use crate::Result;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Context passed to importers and extractors during baking.
pub struct BakeContext {
    /// Path to the source file being baked.
    pub source_path: PathBuf,
    /// Directory for output files.
    pub output_dir: PathBuf,
    /// Additional files that this bake depends on (for incremental baking).
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

/// A single baked asset output.
/// Produced by an Extractor, consumed by BundleWriter.
pub struct BakeOutput {
    /// Unique identifier for this asset (used for references).
    pub asset_id: Uuid,
    /// Asset type UUID (identifies the format, e.g., i3mesh, i3scene).
    pub asset_type: Uuid,
    /// Binary payload (the asset data, ready to write).
    pub data: Vec<u8>,
    /// Human-readable name for logging/debugging.
    pub name: String,
}

/// Intermediate data produced by an importer.
/// Each importer defines its own concrete type implementing this trait.
pub trait ImportedData: Send + Sync {
    /// Returns the source file path that was imported.
    fn source_path(&self) -> &Path;

    /// Support for downcasting to concrete types.
    /// Required for extractors to access importer-specific data.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// An importer reads a source file format and produces parsed data.
///
/// Importers are responsible for:
/// 1. Parsing the source file (e.g., glTF via Assimp)
/// 2. Producing an intermediate representation (ImportedData)
/// 3. Coordinating extractors to produce final outputs
pub trait Importer: Send + Sync {
    /// Importer name (for logging/debug).
    fn name(&self) -> &str;

    /// Supported source file extensions (e.g., &["gltf", "glb", "fbx", "obj"]).
    fn source_extensions(&self) -> &[&str];

    /// Parse a source file and produce intermediate data.
    /// Called once per source file.
    fn import(&self, source_path: &Path) -> Result<Box<dyn ImportedData>>;

    /// Run all extractors on the imported data, producing baked assets.
    /// The default implementation calls each registered extractor.
    fn extract(&self, data: &dyn ImportedData, ctx: &BakeContext) -> Result<Vec<BakeOutput>>;
}

/// An extractor produces a specific output type from imported data.
///
/// Extractors are registered with an importer and operate on the parsed
/// intermediate representation. Multiple extractors can run on the same
/// imported data (e.g., MeshExtractor + SceneExtractor from a glTF file).
pub trait Extractor: Send + Sync {
    /// Extractor name (for logging/debug).
    fn name(&self) -> &str;

    /// Asset type UUID this extractor produces.
    fn output_type(&self) -> Uuid;

    /// Extract assets from imported data.
    /// Returns zero or more baked outputs (e.g., multiple meshes from one file).
    fn extract(&self, data: &dyn ImportedData, ctx: &BakeContext) -> Result<Vec<BakeOutput>>;
}

/// Legacy trait kept for compatibility during migration.
/// Will be removed once all plugins are converted to Importer/Extractor.
#[deprecated(note = "Use Importer and Extractor traits instead")]
#[allow(deprecated)]
pub trait AssetPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn source_extension(&self) -> &str;
    fn output_extension(&self) -> &str;
    fn bake(&self, context: &mut BakeContext) -> Result<BakeResult>;
}

/// Legacy result type kept for compatibility.
#[deprecated(note = "Use BakeOutput instead")]
#[allow(deprecated)]
pub struct BakeResult {
    pub blob: Vec<u8>,
    pub secondary_outputs: Vec<PathBuf>,
}

/// Pipeline node for building import pipelines.
pub struct PipelineNode {
    pub importer: Box<dyn Importer>,
    pub extractors: Vec<Box<dyn Extractor>>,
}

impl PipelineNode {
    pub fn new(importer: Box<dyn Importer>) -> Self {
        Self {
            importer,
            extractors: Vec::new(),
        }
    }

    pub fn add_extractor(&mut self, extractor: Box<dyn Extractor>) {
        self.extractors.push(extractor);
    }
}
