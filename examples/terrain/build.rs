use i3_baker::prelude::*;
use std::path::Path;

fn main() {
    if let Err(e) = run() {
        for line in e.to_string().split('\n') {
            println!("cargo:warning={}", line);
        }
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let profile_dir = Path::new(&out_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    println!("cargo:rerun-if-changed=assets/terrain.bake.ron");
    println!("cargo:rerun-if-changed=assets/pipelines");
    println!("cargo:rerun-if-changed=assets/shaders");
    println!("cargo:rerun-if-changed=../../crates/i3_voxel/assets/pipelines");
    println!("cargo:rerun-if-changed=../../crates/i3_voxel/assets/shaders");

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PROFILING");
    let profiling = std::env::var("CARGO_FEATURE_PROFILING").is_ok();

    ManifestBaker::from_file(Path::new(&manifest_dir).join("assets/terrain.bake.ron"))
        .with_output_dir(profile_dir)
        .with_shader_debug_info(profiling)
        .execute()?;

    Ok(())
}
