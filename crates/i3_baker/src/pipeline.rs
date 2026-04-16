//! Baker pipeline architecture.
//!
//! Defines the Importer → Extractor model for asset baking:
//! - Importers parse source files (e.g., AssimpImporter for glTF/FBX/OBJ)
//! - Extractors produce typed outputs from imported data (e.g., MeshExtractor, SceneExtractor)
//!
//! Context passed to importers and extractors during baking.

use crate::Result;
use crate::writer::BundleWriter;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Instant, UNIX_EPOCH};
use uuid::Uuid;

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
#[derive(Debug, Clone)]
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

    /// Returns a list of additional dependencies for a source file.
    /// Used for incremental baking (mtime checks).
    fn get_dependencies(&self, _source_path: &Path) -> Result<Vec<PathBuf>> {
        Ok(Vec::new())
    }
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
    importer:    Box<dyn Importer>,
    /// Serialized bake options — included in the cache key so that changing
    /// options (e.g. IBL intensity_scale) invalidates the cache even when
    /// the source file hasn't changed on disk.
    config_key:  Vec<u8>,
}

/// A high-level declarative Baker for asset bundles.
///
/// Handles:
/// - Cargo `rerun-if-changed` tracking.
/// - Incremental build logic (mtime checks).
/// - Parallel asset baking using Rayon.
// ─────────────────────────────────────────────────────────────────────────────
// Per-asset cache format (postcard-serialised)
// ─────────────────────────────────────────────────────────────────────────────
// Cache format — split into two files to keep the validity check fast:
//
//   {stem}.meta  — tiny: source_mtime + dep mtimes + config_key
//   {stem}.cache — large: the actual baked output blobs
//
// The hot path (all assets unchanged) only reads .meta files.
// The .cache files are only opened when the bundle needs reassembly.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct CacheMeta {
    source_mtime: u64,
    deps:         Vec<(String, u64)>,
    config_key:   Vec<u8>,
}

#[derive(Serialize, Deserialize)]
struct CachedOutput {
    asset_id:   [u8; 16],
    asset_type: [u8; 16],
    name:       String,
    data:       Vec<u8>,
}

fn mtime_secs(path: &Path) -> Option<u64> {
    std::fs::metadata(path).ok().and_then(|m| {
        m.modified().ok().and_then(|t| {
            t.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
        })
    })
}

fn meta_path(cache_path: &Path) -> PathBuf {
    cache_path.with_extension("meta")
}

fn load_meta(cache_path: &Path) -> Option<CacheMeta> {
    let bytes = std::fs::read(meta_path(cache_path)).ok()?;
    postcard::from_bytes(&bytes).ok()
}

fn load_outputs(cache_path: &Path) -> Option<Vec<CachedOutput>> {
    let bytes = std::fs::read(cache_path).ok()?;
    postcard::from_bytes(&bytes).ok()
}

fn save_cache(cache_path: &Path, meta: &CacheMeta, outputs: &[BakeOutput]) -> Result<()> {
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| crate::BakerError::Os {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    let meta_bytes = postcard::to_allocvec(meta).map_err(|e| {
        crate::BakerError::Pipeline(format!("cache meta serialise: {}", e))
    })?;
    std::fs::write(meta_path(cache_path), &meta_bytes).map_err(|e| crate::BakerError::Os {
        path: meta_path(cache_path),
        source: e,
    })?;
    let cached: Vec<CachedOutput> = outputs.iter().map(|o| CachedOutput {
        asset_id:   *o.asset_id.as_bytes(),
        asset_type: *o.asset_type.as_bytes(),
        name:       o.name.clone(),
        data:       o.data.clone(),
    }).collect();
    let data_bytes = postcard::to_allocvec(&cached).map_err(|e| {
        crate::BakerError::Pipeline(format!("cache data serialise: {}", e))
    })?;
    std::fs::write(cache_path, &data_bytes).map_err(|e| crate::BakerError::Os {
        path: cache_path.to_path_buf(),
        source: e,
    })
}

fn meta_is_valid(meta: &CacheMeta, source: &Path, deps: &[PathBuf], config_key: &[u8]) -> bool {
    if meta.config_key != config_key {
        return false;
    }
    let src_mtime = match mtime_secs(source) {
        Some(t) => t,
        None    => return false,
    };
    if src_mtime != meta.source_mtime {
        return false;
    }
    if deps.len() != meta.deps.len() {
        return false;
    }
    for (dep_str, stored_mtime) in &meta.deps {
        match mtime_secs(Path::new(dep_str)) {
            Some(t) if t == *stored_mtime => {}
            _ => return false,
        }
    }
    true
}

fn outputs_from_cached(cached: Vec<CachedOutput>) -> Vec<BakeOutput> {
    cached.into_iter().map(|o| BakeOutput {
        asset_id:   Uuid::from_bytes(o.asset_id),
        asset_type: Uuid::from_bytes(o.asset_type),
        name:       o.name,
        data:       o.data,
    }).collect()
}

fn make_meta(source: &Path, deps: &[PathBuf], config_key: &[u8]) -> CacheMeta {
    CacheMeta {
        source_mtime: mtime_secs(source).unwrap_or(0),
        deps: deps.iter()
            .map(|p| (p.to_string_lossy().into_owned(), mtime_secs(p).unwrap_or(0)))
            .collect(),
        config_key: config_key.to_vec(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────

pub struct BundleBaker {
    bundle_name: String,
    output_dir:  PathBuf,
    assets:      Vec<AssetJob>,
    /// When Some, per-asset incremental caching is enabled.
    cache_dir:   Option<PathBuf>,
}

impl BundleBaker {
    /// Create a new baker for the specified bundle.
    /// Derived paths use `CARGO_MANIFEST_DIR` and `OUT_DIR`.
    pub fn new(bundle_name: impl Into<String>) -> Result<Self> {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .map_err(|_| crate::BakerError::Pipeline("CARGO_MANIFEST_DIR not found".to_string()))?;

        let manifest_path = PathBuf::from(&manifest_dir);
        let output_dir = manifest_path.join("assets");

        Ok(Self {
            bundle_name: bundle_name.into(),
            output_dir,
            assets:    Vec::new(),
            cache_dir: None,
        })
    }

    /// Create a baker with an explicit output directory (used by `ManifestBaker`).
    pub fn new_with_output(
        bundle_name: impl Into<String>,
        output_dir:  impl AsRef<Path>,
    ) -> Self {
        Self {
            bundle_name: bundle_name.into(),
            output_dir:  output_dir.as_ref().to_path_buf(),
            assets:      Vec::new(),
            cache_dir:   None,
        }
    }

    /// Set an explicit output directory.
    pub fn with_output_dir(mut self, path: impl AsRef<Path>) -> Self {
        self.output_dir = path.as_ref().to_path_buf();
        self
    }

    /// Enable per-asset incremental caching in `cache_dir`.
    /// The baker will create `{cache_dir}/{bundle_name}/{asset_stem}.cache` files
    /// and only recompile assets whose source or dependencies have changed.
    pub fn with_cache_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.cache_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Register an asset to be baked into the bundle.
    pub fn add_asset(
        mut self,
        source_path: impl AsRef<Path>,
        importer: impl Importer + 'static,
    ) -> Self {
        self.assets.push(AssetJob {
            source_path: source_path.as_ref().to_path_buf(),
            importer:    Box::new(importer),
            config_key:  vec![],
        });
        self
    }

    /// Register an asset with a config fingerprint for cache invalidation.
    ///
    /// `config_key` should be a serialized representation of the bake options.
    /// The cache is invalidated whenever `config_key` changes, even if the
    /// source file mtime has not changed (e.g. IBL `intensity_scale` change).
    pub fn add_asset_keyed(
        mut self,
        source_path: impl AsRef<Path>,
        importer:    impl Importer + 'static,
        config_key:  Vec<u8>,
    ) -> Self {
        self.assets.push(AssetJob {
            source_path: source_path.as_ref().to_path_buf(),
            importer:    Box::new(importer),
            config_key,
        });
        self
    }

    /// Recursively scan a directory and add all pipelines (.i3p files) found.
    pub fn add_pipelines(mut self, directory: impl AsRef<Path>) -> Result<Self> {
        let dir = directory.as_ref();
        if !dir.exists() {
            return Err(crate::BakerError::Pipeline(format!(
                "Directory not found: {:?}",
                dir
            )));
        }

        let importer = crate::importers::PipelineImporter::new();

        for entry in std::fs::read_dir(dir).map_err(|e| crate::BakerError::Os {
            path: dir.to_path_buf(),
            source: e,
        })? {
            let entry = entry.map_err(|e| crate::BakerError::Os {
                path: dir.to_path_buf(),
                source: e,
            })?;
            let path = entry.path();

            if path.is_dir() {
                self = self.add_pipelines(&path)?;
            } else if path.extension().map_or(false, |ext| ext == "i3p") {
                self.assets.push(AssetJob {
                    source_path: path,
                    importer:    Box::new(importer),
                    config_key:  vec![],
                });
            }
        }

        Ok(self)
    }

    /// Recursively scan a directory and add all images found with the given options.
    pub fn add_images(
        mut self,
        directory: impl AsRef<Path>,
        options: crate::importers::image_importer::TextureImportOptions,
    ) -> Result<Self> {
        let dir = directory.as_ref();
        if !dir.exists() {
            return Err(crate::BakerError::Pipeline(format!(
                "Directory not found: {:?}",
                dir
            )));
        }

        let importer = crate::importers::ImageImporter::new(options);
        let extensions = importer.source_extensions();

        for entry in std::fs::read_dir(dir).map_err(|e| crate::BakerError::Os {
            path: dir.to_path_buf(),
            source: e,
        })? {
            let entry = entry.map_err(|e| crate::BakerError::Os {
                path: dir.to_path_buf(),
                source: e,
            })?;
            let path = entry.path();

            if path.is_dir() {
                self = self.add_images(&path, options)?;
            } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if extensions.iter().any(|&e| e == ext) {
                    self.assets.push(AssetJob {
                        source_path: path,
                        importer:    Box::new(importer),
                        config_key:  vec![],
                    });
                }
            }
        }

        Ok(self)
    }

    /// Register an HDR environment map to be baked as an IBL asset.
    pub fn add_hdr_ibl(
        self,
        source_path: impl AsRef<Path>,
        options: crate::importers::ibl_bake::IblBakeOptions,
    ) -> Self {
        self.add_asset(
            source_path,
            crate::importers::HdrIblImporter { options },
        )
    }

    /// Execute the baking process.
    ///
    /// When `with_cache_dir()` was set, uses per-asset incremental caching:
    /// only assets whose source/dependencies changed since the last bake
    /// are recompiled; the rest are loaded from the cache.
    ///
    /// Without a cache dir, falls back to all-or-nothing bundle-level mtime check.
    pub fn execute(self) -> Result<()> {
        let blob_path    = self.output_dir.join(format!("{}.i3b", self.bundle_name));
        let catalog_path = self.output_dir.join(format!("{}.i3c", self.bundle_name));

        // Cargo change-tracking
        println!("cargo:rerun-if-changed={}", catalog_path.display());
        for asset in &self.assets {
            println!("cargo:rerun-if-changed={}", asset.source_path.display());
            if !asset.source_path.exists() {
                println!("cargo:warning=Asset source NOT FOUND: {:?}", asset.source_path);
            }
        }

        if !self.output_dir.exists() {
            std::fs::create_dir_all(&self.output_dir).map_err(|e| crate::BakerError::Os {
                path: self.output_dir.clone(),
                source: e,
            })?;
        }

        let force = std::env::var("FORCE_BAKE").is_ok();

        // ── Per-asset incremental path ────────────────────────────────────────
        if let Some(ref cache_base) = self.cache_dir {
            let cache_base = cache_base.clone();
            return self.execute_incremental(&blob_path, &catalog_path, &cache_base, force);
        }

        // ── Legacy all-or-nothing path ────────────────────────────────────────
        let mut needs_bake = !catalog_path.exists() || !blob_path.exists() || force;

        if !needs_bake {
            let output_mtime = std::fs::metadata(&catalog_path)
                .and_then(|m| m.modified())
                .map_err(|e| crate::BakerError::Os {
                    path: catalog_path.clone(),
                    source: e,
                })?;

            // Rebuild if build.rs itself changed
            if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
                let build_rs = Path::new(&manifest_dir).join("build.rs");
                if build_rs.exists() {
                    if let Some(m) = std::fs::metadata(&build_rs).ok() {
                        if let Ok(mtime) = m.modified() {
                            if mtime > output_mtime {
                                needs_bake = true;
                            }
                        }
                    }
                }
            }

            if !needs_bake {
                'outer: for asset in &self.assets {
                    let mut all_deps = vec![asset.source_path.clone()];
                    if let Ok(deps) = asset.importer.get_dependencies(&asset.source_path) {
                        all_deps.extend(deps);
                    }
                    for dep in &all_deps {
                        if dep.exists() {
                            let mtime = std::fs::metadata(dep)
                                .and_then(|m| m.modified())
                                .map_err(|e| crate::BakerError::Os {
                                    path: dep.clone(),
                                    source: e,
                                })?;
                            if mtime > output_mtime {
                                needs_bake = true;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        if needs_bake {
            self.bake_all_and_write(&blob_path, &catalog_path)?;
        }

        Ok(())
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn execute_incremental(
        self,
        blob_path:    &Path,
        catalog_path: &Path,
        cache_base:   &Path,
        force:        bool,
    ) -> Result<()> {
        let bundle_cache_dir = cache_base.join(&self.bundle_name);
        let total_assets     = self.assets.len();
        let total_start      = Instant::now();

        // Pre-compute cache paths (indexed, parallel-safe).
        let cache_paths: Vec<PathBuf> = self.assets.iter().map(|job| {
            let stem = job.source_path.file_stem().unwrap_or_default().to_string_lossy();
            bundle_cache_dir.join(format!("{}.cache", stem))
        }).collect();

        // ── Phase 1: meta-only validity check (no output data loaded) ─────────
        let check_start = Instant::now();

        let valid: Vec<bool> = self.assets.iter().zip(cache_paths.iter())
            .map(|(job, cache_path)| {
                if force { return false; }
                let deps = job.importer.get_dependencies(&job.source_path).unwrap_or_default();
                load_meta(cache_path)
                    .map(|m| meta_is_valid(&m, &job.source_path, &deps, &job.config_key))
                    .unwrap_or(false)
            })
            .collect();

        let check_elapsed = check_start.elapsed();
        let rebaked_count = valid.iter().filter(|&&v| !v).count();
        let bundle_missing = !catalog_path.exists() || !blob_path.exists();

        // ── Fast path ─────────────────────────────────────────────────────────
        if rebaked_count == 0 && !bundle_missing && !force {
            println!(
                "cargo:warning=[i3_baker] '{}': all {} assets up-to-date (cache check: {:.2?}).",
                self.bundle_name, total_assets, check_elapsed
            );
            return Ok(());
        }

        // ── Phase 2: bake misses in parallel ──────────────────────────────────
        println!(
            "cargo:warning=[i3_baker] '{}': {}/{} rebaked, assembling bundle...",
            self.bundle_name, rebaked_count, total_assets
        );

        // Bake only the cache-miss jobs, keyed by their original index.
        let miss_indices: Vec<usize> = valid.iter().enumerate()
            .filter(|&(_, v)| !v)
            .map(|(i, _)| i)
            .collect();

        let baked_outputs: Result<Vec<(usize, Vec<BakeOutput>)>> = miss_indices
            .par_iter()
            .map(|&idx| {
                let job        = &self.assets[idx];
                let cache_path = &cache_paths[idx];
                let stem       = job.source_path.file_stem()
                    .unwrap_or_default().to_string_lossy();

                let start = Instant::now();
                println!(
                    "cargo:warning=[i3_baker] [{}/{}] Baking {}...",
                    idx + 1, total_assets, stem
                );
                let ctx      = BakeContext::new(&job.source_path, &self.output_dir);
                let imported = job.importer.import(&job.source_path)?;
                let outputs  = job.importer.extract(imported.as_ref(), &ctx)?;
                println!(
                    "cargo:warning=[i3_baker] [{}/{}] Finished {} in {:.2?}.",
                    idx + 1, total_assets, stem, start.elapsed()
                );

                let deps = job.importer.get_dependencies(&job.source_path).unwrap_or_default();
                let meta = make_meta(&job.source_path, &deps, &job.config_key);
                if let Err(e) = save_cache(cache_path, &meta, &outputs) {
                    println!("cargo:warning=[i3_baker] cache write failed for {}: {}", stem, e);
                }

                Ok((idx, outputs))
            })
            .collect();

        let baked_map: std::collections::HashMap<usize, Vec<BakeOutput>> =
            baked_outputs?.into_iter().collect();

        // ── Phase 3: assemble bundle (hits load from .cache, misses from RAM) ──
        let write_start = Instant::now();
        let mut writer = BundleWriter::new(blob_path)?;

        for (idx, cache_path) in cache_paths.iter().enumerate() {
            let outputs = if valid[idx] {
                // Cache hit: load output data from disk now (only needed for assembly).
                let cached = load_outputs(cache_path).ok_or_else(|| {
                    crate::BakerError::Pipeline(format!("cache data missing: {:?}", cache_path))
                })?;
                outputs_from_cached(cached)
            } else {
                // Cache miss: already baked in phase 2.
                baked_map.get(&idx)
                    .ok_or_else(|| crate::BakerError::Pipeline(
                        format!("missing bake result for index {}", idx)
                    ))?
                    .clone()
            };
            for output in outputs {
                writer.add_bake_output(&output)?;
            }
        }

        writer.finish(catalog_path)?;
        println!(
            "cargo:warning=[i3_baker] Bundle '{}' done in {:.2?} (cache check: {:.2?}, bundle write: {:.2?}).",
            self.bundle_name, total_start.elapsed(), check_elapsed, write_start.elapsed()
        );

        Ok(())
    }

    fn bake_all_and_write(self, blob_path: &Path, catalog_path: &Path) -> Result<()> {
        println!(
            "cargo:warning=[i3_baker] Baking bundle '{}' with {} assets (parallel)...",
            self.bundle_name,
            self.assets.len()
        );
        let total_start  = Instant::now();
        let total_assets = self.assets.len();

        let results: Result<Vec<Vec<BakeOutput>>> = self
            .assets
            .into_par_iter()
            .enumerate()
            .map(|(idx, job)| {
                let start      = Instant::now();
                let asset_name = job.source_path.file_name().unwrap_or_default().to_string_lossy();
                println!(
                    "cargo:warning=[i3_baker] [{}/{}] Baking {}...",
                    idx + 1, total_assets, asset_name
                );
                let ctx      = BakeContext::new(&job.source_path, &self.output_dir);
                let imported = job.importer.import(&job.source_path)?;
                let outputs  = job.importer.extract(imported.as_ref(), &ctx)?;
                println!(
                    "cargo:warning=[i3_baker] [{}/{}] Finished {} in {:.2?}.",
                    idx + 1, total_assets, asset_name, start.elapsed()
                );
                Ok(outputs)
            })
            .collect();

        let all_outputs = results?;
        let write_start = Instant::now();
        let mut writer  = BundleWriter::new(blob_path)?;
        for outputs in all_outputs {
            for output in outputs {
                writer.add_bake_output(&output)?;
            }
        }
        writer.finish(catalog_path)?;

        println!(
            "cargo:warning=[i3_baker] Bundle '{}' bake complete in {:.2?} (bundle write: {:.2?}).",
            self.bundle_name,
            total_start.elapsed(),
            write_start.elapsed()
        );
        Ok(())
    }
}
