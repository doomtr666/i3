//! RON-based declarative bake manifest.
//!
//! `ManifestBaker` reads a `.bake.ron` file that lists all assets to bake,
//! then delegates to `BundleBaker` with per-asset caching enabled.

use crate::importers::ibl_bake::IblBakeOptions;
use crate::pipeline::BundleBaker;
use crate::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};

// postcard is used to fingerprint IBL options for cache invalidation

// ─── Manifest structs ─────────────────────────────────────────────────────────

/// Declarative bake manifest — the RON-parsed representation of a `.bake.ron` file.
///
/// All paths are **relative to the manifest file's directory**.
#[derive(Debug, Deserialize)]
pub struct BakeManifest {
    pub bundle_name: String,

    /// Directories to scan recursively for `.i3p` pipeline files.
    #[serde(default)]
    pub pipelines_dirs: Vec<PathBuf>,

    /// Individual `.i3p` pipeline files (in addition to scanned dirs).
    #[serde(default)]
    pub pipeline_files: Vec<PathBuf>,

    /// HDR environment maps to bake as IBL assets.
    #[serde(default)]
    pub ibl: Vec<IblManifestEntry>,

    /// 3D scene/mesh files to bake via AssimpImporter (glTF, glb, fbx, obj, …).
    #[serde(default)]
    pub meshes: Vec<PathBuf>,
}

/// One IBL entry in the manifest.
#[derive(Debug, Deserialize)]
pub struct IblManifestEntry {
    /// Path to the `.hdr` or `.exr` source (relative to manifest dir).
    pub source: PathBuf,

    /// Bake options — defaults to `IblBakeOptions::default()` if omitted.
    #[serde(default)]
    pub options: IblBakeOptions,
}

// ─── ManifestBaker ────────────────────────────────────────────────────────────

/// High-level baker that reads a `BakeManifest` from a RON file and executes
/// an incremental per-asset bake with file-system caching.
pub struct ManifestBaker {
    manifest_path: PathBuf,
    output_dir:    Option<PathBuf>,
}

impl ManifestBaker {
    /// Create a `ManifestBaker` for the given `.bake.ron` file.
    pub fn from_file(path: impl Into<PathBuf>) -> Self {
        Self {
            manifest_path: path.into(),
            output_dir:    None,
        }
    }

    /// Override the output directory (where `.i3b` / `.i3c` are written).
    pub fn with_output_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.output_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Execute the manifest: read RON, resolve paths, run `BundleBaker` with caching.
    pub fn execute(self) -> Result<()> {
        // ── 1. Parse manifest ────────────────────────────────────────────────
        let manifest_path = self.manifest_path.canonicalize().unwrap_or(self.manifest_path.clone());
        println!("cargo:rerun-if-changed={}", manifest_path.display());

        let content = std::fs::read_to_string(&manifest_path).map_err(|e| {
            crate::BakerError::Pipeline(format!(
                "Failed to read manifest {:?}: {}",
                manifest_path, e
            ))
        })?;

        let manifest: BakeManifest = ron::from_str(&content).map_err(|e| {
            crate::BakerError::Pipeline(format!(
                "Failed to parse manifest {:?}: {}",
                manifest_path, e
            ))
        })?;

        // ── 2. Resolve base dir (manifest file's parent) ─────────────────────
        let base_dir = manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        let resolve = |p: &Path| -> PathBuf {
            if p.is_absolute() { p.to_path_buf() } else { base_dir.join(p) }
        };

        // ── 3. Determine output & cache dirs ─────────────────────────────────
        let output_dir = match self.output_dir {
            Some(d) => d,
            None => {
                // Fallback: alongside the manifest (only useful outside build.rs)
                base_dir.clone()
            }
        };

        let cache_dir = output_dir.join(".bake_cache");

        // ── 4. Build BundleBaker with cache ───────────────────────────────────
        let mut baker = BundleBaker::new_with_output(&manifest.bundle_name, &output_dir)
            .with_cache_dir(cache_dir);

        // Pipelines from directories
        for dir in &manifest.pipelines_dirs {
            let abs = resolve(dir);
            if abs.exists() {
                baker = baker.add_pipelines(&abs)?;
            } else {
                println!(
                    "cargo:warning=[i3_baker] manifest: pipelines_dir not found: {}",
                    abs.display()
                );
            }
        }

        // Individual pipeline files
        for file in &manifest.pipeline_files {
            let abs = resolve(file);
            baker = baker.add_asset(
                &abs,
                crate::importers::PipelineImporter::new(),
            );
        }

        // Mesh / scene assets
        for mesh in &manifest.meshes {
            let abs = resolve(mesh);
            if abs.exists() {
                baker = baker.add_asset(&abs, crate::importers::AssimpImporter::new());
            } else {
                println!(
                    "cargo:warning=[i3_baker] manifest: mesh not found: {}",
                    abs.display()
                );
            }
        }

        // IBL assets — use add_asset_keyed so that changing options (intensity_scale,
        // sun_strength_ratio, etc.) in the manifest invalidates the per-asset cache.
        for entry in manifest.ibl {
            let abs = resolve(&entry.source);
            let config_key = postcard::to_allocvec(&entry.options).unwrap_or_default();
            baker = baker.add_asset_keyed(
                abs,
                crate::importers::HdrIblImporter { options: entry.options },
                config_key,
            );
        }

        baker.execute()
    }
}
