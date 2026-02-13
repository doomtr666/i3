#!/usr/bin/env rust-script
//! Script to download native dependencies (SDL2, etc.)
//!
//! Usage: cargo run --manifest-path third_party/Cargo.toml
//! Or: rust-script third_party/download.rs

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const SDL2_VERSION: &str = "2.30.0";
const SDL2_URL: &str =
    "https://github.com/libsdl-org/SDL/releases/download/release-2.30.0/SDL2-devel-2.30.0-VC.zip";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Downloading native dependencies...");
    println!();

    // Check for Vulkan SDK
    check_vulkan_sdk()?;

    let third_party = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let libs_dir = third_party.join("libs");

    // Create necessary directories
    fs::create_dir_all(&libs_dir)?;

    // Download SDL2
    download_sdl2(&libs_dir)?;

    println!("All dependencies are ready!");
    Ok(())
}

fn check_vulkan_sdk() -> Result<(), Box<dyn std::error::Error>> {
    match env::var("VULKAN_SDK") {
        Ok(sdk_path) => {
            let sdk_path = PathBuf::from(&sdk_path);
            if sdk_path.exists() {
                println!("Vulkan SDK found: {}", sdk_path.display());
            } else {
                println!(
                    "Warning: VULKAN_SDK is set but directory does not exist: {}",
                    sdk_path.display()
                );
                println!("   The project may fail to build.");
            }
            Ok(())
        }
        Err(_) => {
            eprintln!();
            eprintln!("ERROR: VULKAN_SDK environment variable is not set!");
            eprintln!();
            eprintln!("   This project requires the Vulkan SDK to build.");
            eprintln!();
            eprintln!("   Please install the Vulkan SDK:");
            eprintln!("   - Windows: https://vulkan.lunarg.com/sdk/home#windows");
            eprintln!("   - Linux:   https://vulkan.lunarg.com/sdk/home#linux");
            eprintln!();
            eprintln!("   After installation, restart your terminal/IDE to pick up the");
            eprintln!("   VULKAN_SDK environment variable.");
            eprintln!();
            Err("VULKAN_SDK not found".into())
        }
    }
}

fn download_sdl2(libs_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let sdl2_dir = libs_dir.join("sdl2");

    // Check if SDL2 already exists (check for lib directory which is always present)
    if sdl2_dir.join("lib").exists() {
        println!("SDL2 already present in {}", sdl2_dir.display());
        return Ok(());
    }

    println!("Downloading SDL2 v{}...", SDL2_VERSION);

    // Download the ZIP file
    let zip_path = libs_dir.join("sdl2.zip");
    download_file(SDL2_URL, &zip_path)?;

    // Extract the ZIP
    println!("Extracting SDL2...");
    extract_zip(&zip_path, libs_dir)?;

    // Rename extracted directory
    let extracted_dir = libs_dir.join(format!("SDL2-{}", SDL2_VERSION));
    if extracted_dir.exists() {
        // Remove destination if it exists (in case of partial extraction)
        if sdl2_dir.exists() {
            fs::remove_dir_all(&sdl2_dir)?;
        }
        fs::rename(&extracted_dir, &sdl2_dir)?;
    }

    // Cleanup
    if zip_path.exists() {
        fs::remove_file(&zip_path)?;
    }

    println!("✓ SDL2 installed in {}", sdl2_dir.display());
    Ok(())
}

fn download_file(url: &str, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Use curl or powershell depending on availability
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;

        let status = Command::new("powershell")
            .arg("-Command")
            .arg(format!(
                "Invoke-WebRequest -Uri '{}' -OutFile '{}'",
                url,
                dest.display()
            ))
            .status()?;

        if !status.success() {
            return Err("Download failed".into());
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        use std::process::Command;

        let status = Command::new("curl")
            .arg("-L")
            .arg("-o")
            .arg(dest)
            .arg(url)
            .status()?;

        if !status.success() {
            return Err("Download failed".into());
        }
    }

    Ok(())
}

fn extract_zip(zip_path: &Path, dest_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;

        let status = Command::new("powershell")
            .arg("-Command")
            .arg(format!(
                "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                zip_path.display(),
                dest_dir.display()
            ))
            .status()?;

        if !status.success() {
            return Err("Extraction failed".into());
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        use std::process::Command;

        let status = Command::new("unzip")
            .arg("-q")
            .arg(zip_path)
            .arg("-d")
            .arg(dest_dir)
            .status()?;

        if !status.success() {
            return Err("Extraction failed".into());
        }
    }

    Ok(())
}
