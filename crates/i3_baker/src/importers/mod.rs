//! Asset importers for various source formats.
//!
//! This module contains importers that parse source files and produce
//! intermediate representations for extractors.

mod assimp_importer;

pub use assimp_importer::{AssimpImporter, AssimpScene};
