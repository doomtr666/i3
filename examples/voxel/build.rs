use i3_baker::prelude::*;
use std::path::Path;

fn main() {
    if let Err(e) = run() {
        eprintln!("[voxel build] error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir      = std::env::var("OUT_DIR").unwrap();

    let target_dir = Path::new(&out_dir)
        .parent().unwrap()
        .parent().unwrap()
        .parent().unwrap();

    let workspace_root = Path::new(&manifest_dir)
        .parent().unwrap() // examples/
        .parent().unwrap(); // workspace root

    // Compile assets
    println!("cargo:rerun-if-changed=assets/voxel.bake.ron");
    println!("cargo:rerun-if-changed=assets/pipelines");
    println!("cargo:rerun-if-changed=assets/shaders");

    ManifestBaker::from_file(
        Path::new(&manifest_dir).join("assets/voxel.bake.ron"),
    )
    .with_output_dir(target_dir)
    .execute()?;

    // Copy renderer shaders to target directory so they can be loaded at runtime
    let shader_src = workspace_root.join("crates/i3_renderer/assets/shaders");
    let shader_dst = target_dir.join("shaders");
    copy_dir(&shader_src, &shader_dst)
        .map_err(|e| i3_baker::BakerError::Pipeline(e.to_string()))?;

    // Also copy voxel-specific shaders
    let voxel_shader_src = Path::new(&manifest_dir).join("assets/shaders");
    copy_dir(&voxel_shader_src, &shader_dst)
        .map_err(|e| i3_baker::BakerError::Pipeline(e.to_string()))?;

    Ok(())
}

fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    if !dst.exists() {
        std::fs::create_dir_all(dst)?;
    }
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty    = entry.file_type()?;
        if ty.is_dir() {
            copy_dir(&entry.path(), &dst.join(entry.file_name()))?;
        } else {
            std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
        }
    }
    Ok(())
}

