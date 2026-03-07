//! Baker pipeline architecture.
//!
//! Defines the Importer → Extractor model for asset baking:
//! - Importers parse source files (e.g., AssimpImporter for glTF/FBX/OBJ)
//! - Extractors produce typed outputs from imported data (e.g., MeshExtractor, SceneExtractor)

use crate::Result;
use crate::writer::BundleWriter;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::Instant;
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

struct AssetJob {
    source_path: PathBuf,
    importer: Box<dyn Importer>,
}

/// A high-level declarative Baker for asset bundles.
///
/// Handles:
/// - Cargo `rerun-if-changed` tracking.
/// - Incremental build logic (mtime checks).
/// - Parallel asset baking using Rayon.
pub struct BundleBaker {
    bundle_name: String,
    output_dir: PathBuf,
    assets: Vec<AssetJob>,
}

impl BundleBaker {
    /// Create a new baker for the specified bundle.
    /// Derived paths use `CARGO_MANIFEST_DIR` and `OUT_DIR`.
    pub fn new(bundle_name: impl Into<String>) -> Result<Self> {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .map_err(|_| crate::BakerError::Pipeline("CARGO_MANIFEST_DIR not found".to_string()))?;
        let _out_dir = std::env::var("OUT_DIR")
            .map_err(|_| crate::BakerError::Pipeline("OUT_DIR not found".to_string()))?;

        let manifest_path = PathBuf::from(&manifest_dir);
        let output_dir = manifest_path.join("assets");
        if !output_dir.exists() {
            std::fs::create_dir_all(&output_dir).map_err(|e| crate::BakerError::Os {
                path: output_dir.clone(),
                source: e,
            })?;
        }

        Ok(Self {
            bundle_name: bundle_name.into(),
            output_dir,
            assets: Vec::new(),
        })
    }

    /// Register an asset to be baked into the bundle.
    pub fn add_asset(
        mut self,
        source_path: impl AsRef<Path>,
        importer: impl Importer + 'static,
    ) -> Self {
        self.assets.push(AssetJob {
            source_path: source_path.as_ref().to_path_buf(),
            importer: Box::new(importer),
        });
        self
    }

    /// Execute the baking process.
    pub fn execute(self) -> Result<()> {
        println!("cargo:rerun-if-changed=build.rs");
        let blob_path = self.output_dir.join(format!("{}.i3b", self.bundle_name));
        let catalog_path = self.output_dir.join(format!("{}.i3c", self.bundle_name));

        // Track outputs and inputs for Cargo
        println!("cargo:rerun-if-changed={}", catalog_path.display());
        for asset in &self.assets {
            println!("cargo:rerun-if-changed={}", asset.source_path.display());
            if !asset.source_path.exists() {
                println!(
                    "cargo:warning=Asset source NOT FOUND: {:?}",
                    asset.source_path
                );
            }
        }

        // Global mtime check for incremental baking
        let mut needs_bake = !catalog_path.exists() || !blob_path.exists();
        if !needs_bake {
            let output_metadata =
                std::fs::metadata(&catalog_path).map_err(|e| crate::BakerError::Os {
                    path: catalog_path.clone(),
                    source: e,
                })?;
            let output_mtime = output_metadata
                .modified()
                .map_err(|e| crate::BakerError::Os {
                    path: catalog_path.clone(),
                    source: e,
                })?;

            for asset in &self.assets {
                if asset.source_path.exists() {
                    let metadata = std::fs::metadata(&asset.source_path).map_err(|e| {
                        crate::BakerError::Os {
                            path: asset.source_path.clone(),
                            source: e,
                        }
                    })?;
                    if metadata.modified().map_err(|e| crate::BakerError::Os {
                        path: asset.source_path.clone(),
                        source: e,
                    })? > output_mtime
                    {
                        needs_bake = true;
                        break;
                    }
                }
            }
        }

        if needs_bake {
            println!(
                "Baking bundle '{}' with {} assets (parallel)...",
                self.bundle_name,
                self.assets.len()
            );
            let total_start = Instant::now();

            // Parallel bake using Rayon
            let results: Result<Vec<Vec<BakeOutput>>> = self
                .assets
                .into_par_iter()
                .map(|job| {
                    let start = Instant::now();
                    println!("  Baking {:?}...", job.source_path);
                    let ctx = BakeContext::new(&job.source_path, &self.output_dir);
                    let imported = job.importer.import(&job.source_path)?;
                    let outputs = job.importer.extract(imported.as_ref(), &ctx)?;
                    println!(
                        "  Finished {:?} in {:.2?}.",
                        job.source_path,
                        start.elapsed()
                    );
                    Ok(outputs)
                })
                .collect();

            let all_outputs = results?;

            // Write results (Writer is not thread-safe, so we collect and write sequentially)
            let mut writer = BundleWriter::new(&blob_path)?;
            for outputs in all_outputs {
                for output in outputs {
                    writer.add_bake_output(&output)?;
                }
            }
            writer.finish(&catalog_path)?;

            println!(
                "Bundle '{}' bake complete in {:.2?}.",
                self.bundle_name,
                total_start.elapsed()
            );
        }

        Ok(())
    }
}

/// Legacy trait kept for compatibility during migration.
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
