//! Source file scanner.
//!
//! Recursively scans source directories and associates files with importers.

use crate::Result;
use crate::pipeline::Importer;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A source file discovered by the scanner.
pub struct SourceFile {
    /// Absolute path to the source file.
    pub path: PathBuf,
    /// Relative path from the source root (used for asset IDs).
    pub relative_path: PathBuf,
    /// The importer that will handle this file.
    pub importer_index: usize,
}

/// Scanner discovers source files and associates them with importers.
pub struct Scanner {
    /// Registered importers.
    importers: Vec<Box<dyn Importer>>,
    /// Maps file extensions to importer indices.
    extension_map: HashMap<String, usize>,
}

impl Scanner {
    /// Create a new scanner with no importers.
    pub fn new() -> Self {
        Self {
            importers: Vec::new(),
            extension_map: HashMap::new(),
        }
    }

    /// Register an importer.
    /// The importer will be associated with all its supported extensions.
    pub fn register_importer(&mut self, importer: Box<dyn Importer>) {
        let index = self.importers.len();
        for ext in importer.source_extensions() {
            self.extension_map.insert(ext.to_lowercase(), index);
        }
        self.importers.push(importer);
    }

    /// Get a registered importer by index.
    pub fn get_importer(&self, index: usize) -> Option<&dyn Importer> {
        self.importers.get(index).map(|b| b.as_ref())
    }

    /// Get the importer for a file extension.
    pub fn get_importer_for_extension(&self, ext: &str) -> Option<(usize, &dyn Importer)> {
        let ext_lower = ext.to_lowercase();
        self.extension_map
            .get(&ext_lower)
            .and_then(|&idx| self.importers.get(idx).map(|i| (idx, i.as_ref())))
    }

    /// Scan a source directory recursively.
    /// Returns all source files that have a registered importer.
    pub fn scan_directory(&self, source_root: &Path) -> Result<Vec<SourceFile>> {
        let mut files = Vec::new();
        self.scan_directory_recursive(source_root, source_root, &mut files)?;
        Ok(files)
    }

    fn scan_directory_recursive(
        &self,
        current_dir: &Path,
        source_root: &Path,
        files: &mut Vec<SourceFile>,
    ) -> Result<()> {
        let entries = std::fs::read_dir(current_dir).map_err(|e| crate::BakerError::Os {
            path: current_dir.to_path_buf(),
            source: e,
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| crate::BakerError::Os {
                path: current_dir.to_path_buf(),
                source: e,
            })?;

            let path = entry.path();

            // Skip hidden files and directories
            if let Some(name) = path.file_name() {
                if name.to_string_lossy().starts_with('.') {
                    continue;
                }
            }

            if path.is_dir() {
                // Recurse into subdirectories
                self.scan_directory_recursive(&path, source_root, files)?;
            } else if path.is_file() {
                // Check if we have an importer for this extension
                if let Some(ext) = path.extension() {
                    if let Some((importer_index, _)) =
                        self.get_importer_for_extension(&ext.to_string_lossy())
                    {
                        let relative_path = path.strip_prefix(source_root).unwrap_or(&path);
                        files.push(SourceFile {
                            path: path.clone(),
                            relative_path: relative_path.to_path_buf(),
                            importer_index,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the number of registered importers.
    pub fn importer_count(&self) -> usize {
        self.importers.len()
    }
}

impl Default for Scanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyImporter;

    impl Importer for DummyImporter {
        fn name(&self) -> &str {
            "dummy"
        }

        fn source_extensions(&self) -> &[&str] {
            &["gltf", "glb"]
        }

        fn import(&self, _source_path: &Path) -> Result<Box<dyn crate::pipeline::ImportedData>> {
            unimplemented!()
        }

        fn extract(
            &self,
            _data: &dyn crate::pipeline::ImportedData,
            _ctx: &crate::pipeline::BakeContext,
        ) -> Result<Vec<crate::pipeline::BakeOutput>> {
            unimplemented!()
        }
    }

    #[test]
    fn test_scanner_registration() {
        let mut scanner = Scanner::new();
        scanner.register_importer(Box::new(DummyImporter));

        assert_eq!(scanner.importer_count(), 1);
        assert!(scanner.get_importer_for_extension("gltf").is_some());
        assert!(scanner.get_importer_for_extension("glb").is_some());
        assert!(scanner.get_importer_for_extension("fbx").is_none());
    }

    #[test]
    fn test_extension_case_insensitive() {
        let mut scanner = Scanner::new();
        scanner.register_importer(Box::new(DummyImporter));

        assert!(scanner.get_importer_for_extension("GLTF").is_some());
        assert!(scanner.get_importer_for_extension("Gltf").is_some());
    }
}
