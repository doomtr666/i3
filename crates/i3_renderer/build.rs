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
    let out_dir      = std::env::var("OUT_DIR").unwrap();

    // target/debug or target/release
    let profile_dir = Path::new(&out_dir)
        .parent().unwrap()
        .parent().unwrap()
        .parent().unwrap();

    // Watch only the bake manifest and pipeline/IBL assets — NOT shaders.
    // Shaders are runtime-loaded and don't affect Rust compilation. Watching the full assets/
    // directory would force a crate recompile on every shader edit.
    println!("cargo:rerun-if-changed=assets/system.bake.ron");
    println!("cargo:rerun-if-changed=assets/pipelines");
    println!("cargo:rerun-if-changed=../../i3_egui/assets/pipelines");

    ManifestBaker::from_file(
        Path::new(&manifest_dir).join("assets/system.bake.ron"),
    )
    .with_output_dir(profile_dir)
    .execute()?;

    Ok(())
}
