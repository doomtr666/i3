/// Shared build utilities for workspace build.rs scripts
///
/// This module provides common functions for managing native dependencies

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Returns the path to the third_party/libs directory
pub fn get_libs_dir() -> PathBuf {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = find_workspace_root(&manifest_dir);
    workspace_root.join("third_party").join("libs")
}

/// Finds the workspace root by traversing parent directories
fn find_workspace_root(start: &str) -> PathBuf {
    let mut current = PathBuf::from(start);
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            // Check if it's a workspace
            if let Ok(content) = fs::read_to_string(&cargo_toml) {
                if content.contains("[workspace]") {
                    return current;
                }
            }
        }

        if !current.pop() {
            panic!("Cannot find workspace root");
        }
    }
}

/// Copies a native DLL to the binary output directory
///
/// # Arguments
/// * `lib_name` - Library name (e.g., "sdl2")
/// * `dll_name` - DLL file name (e.g., "SDL2.dll")
///
/// # Example
/// ```no_run
/// copy_dll_to_output("sdl2", "SDL2.dll");
/// ```
#[allow(dead_code)]
pub fn copy_dll_to_output(lib_name: &str, dll_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let libs_dir = get_libs_dir();

    // Try multiple possible DLL locations
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let arch_subdir = match target_arch.as_str() {
        "x86_64" => "x64",
        "x86" => "x86",
        arch => arch,
    };

    let possible_paths = vec![
        libs_dir.join(lib_name).join("lib").join(&arch_subdir).join(dll_name),
        libs_dir.join(lib_name).join("bin").join(&arch_subdir).join(dll_name),
        libs_dir.join(lib_name).join("bin").join(dll_name),
        libs_dir.join(lib_name).join("lib").join(dll_name),
    ];

    for dll_path in possible_paths {
        if dll_path.exists() {
            return copy_dll_to_output_from_path(&dll_path, dll_name);
        }
    }

    eprintln!("⚠️  DLL not found: {}", dll_name);
    eprintln!("    Run: cargo run --manifest-path third_party/Cargo.toml");
    Err(format!("Missing DLL: {}", dll_name).into())
}

/// Copies all DLLs from a library to the output directory
pub fn copy_all_dlls(lib_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let libs_dir = get_libs_dir();

    // Try multiple possible DLL locations
    let possible_dirs = vec![
        libs_dir.join(lib_name).join("bin"),
        libs_dir.join(lib_name).join("lib"),
    ];

    // Also try architecture-specific directories
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let arch_subdir = match target_arch.as_str() {
        "x86_64" => "x64",
        "x86" => "x86",
        arch => arch,
    };

    let mut arch_dirs = vec![
        libs_dir.join(lib_name).join("bin").join(&arch_subdir),
        libs_dir.join(lib_name).join("lib").join(&arch_subdir),
    ];

    // Check arch-specific directories first
    arch_dirs.extend(possible_dirs);

    for bin_dir in arch_dirs {
        if !bin_dir.exists() {
            continue;
        }

        for entry in fs::read_dir(&bin_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("dll") {
                if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
                    copy_dll_to_output_from_path(&path, file_name)?;
                }
            }
        }

        return Ok(()); // Found a valid directory, we're done
    }

    Err(format!("No DLL directory found for library: {}", lib_name).into())
}

/// Internal helper to copy a DLL from a specific path
fn copy_dll_to_output_from_path(dll_path: &Path, dll_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    if !dll_path.exists() {
        return Err(format!("Missing DLL: {}", dll_name).into());
    }

    let target_dir = get_target_dir()?;
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let output_dir = target_dir.join(profile);

    fs::create_dir_all(&output_dir)?;
    let dest_path = output_dir.join(dll_name);

    fs::copy(dll_path, &dest_path)?;
    println!("cargo:rerun-if-changed={}", dll_path.display());

    Ok(())
}

/// Configures linking paths for a native library
///
/// # Arguments
/// * `lib_name` - Library name (e.g., "sdl2")
/// * `link_libs` - List of library names to link (e.g., &["SDL2"])
pub fn setup_native_lib(lib_name: &str, link_libs: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let libs_dir = get_libs_dir();
    let lib_dir = libs_dir.join(lib_name).join("lib");

    // Determine architecture
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let arch_subdir = match target_arch.as_str() {
        "x86_64" => "x64",
        "x86" => "x86",
        arch => arch,
    };

    let lib_path = lib_dir.join(arch_subdir);

    if !lib_path.exists() {
        // Try without architecture subdirectory
        if !lib_dir.exists() {
            return Err(format!("Lib directory not found: {}", lib_dir.display()).into());
        }
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    } else {
        println!("cargo:rustc-link-search=native={}", lib_path.display());
    }

    // Link libraries
    for lib in link_libs {
        println!("cargo:rustc-link-lib=dylib={}", lib);
    }

    // Copy DLLs
    copy_all_dlls(lib_name)?;

    Ok(())
}

/// Finds the workspace target directory
fn get_target_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Use CARGO_TARGET_DIR if defined
    if let Ok(target_dir) = env::var("CARGO_TARGET_DIR") {
        return Ok(PathBuf::from(target_dir));
    }

    // Otherwise, look in the workspace
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = find_workspace_root(&manifest_dir);
    Ok(workspace_root.join("target"))
}
