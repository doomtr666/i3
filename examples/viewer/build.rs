use i3_baker::prelude::*;
use std::path::Path;

fn main() {
    if let Err(e) = run() {
        eprintln!("[i3_baker] viewer build error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir      = std::env::var("OUT_DIR").unwrap();

    // target/debug or target/release
    let target_dir = Path::new(&out_dir)
        .parent().unwrap()
        .parent().unwrap()
        .parent().unwrap();

    println!("cargo:rerun-if-changed=viewer_scenes.bake.ron");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PROFILING");

    let profiling = std::env::var("CARGO_FEATURE_PROFILING").is_ok();

    ManifestBaker::from_file(
        Path::new(&manifest_dir).join("viewer_scenes.bake.ron"),
    )
    .with_output_dir(target_dir)
    .with_shader_debug_info(profiling)
    .execute()?;

    // Copy shaders to target directory.
    // NOTE: We do NOT emit cargo:rerun-if-changed for the shader dir here — shaders are runtime
    // assets and don't affect Rust compilation. Watching them would force a full crate recompile
    // on every shader edit. i3_renderer's build.rs is responsible for shader change tracking.
    let workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();
    let shader_src = workspace_root.join("crates/i3_renderer/assets/shaders");
    let shader_dst = target_dir.join("shaders");
    copy_dir(&shader_src, &shader_dst)
        .map_err(|e| BakerError::Pipeline(format!("shader copy: {}", e)))?;

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
