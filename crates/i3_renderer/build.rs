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
    let manifest_path = Path::new(&manifest_dir);
    
    // Assets are collocated in the crate
    let input_dir = manifest_path.join("assets/pipelines");
    
    // Also include egui assets from crates/i3_egui
    let workspace_root = manifest_path.parent().unwrap().parent().unwrap();
    let egui_assets = workspace_root.join("crates/i3_egui/assets/pipelines");
    
    // Output bundle to the target directory (debug/release)
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);
    
    // We want the final artifacts to be in the profile folder (target/debug or target/release)
    let profile_dir = out_path.parent().unwrap().parent().unwrap().parent().unwrap();
    let output_dir = profile_dir;

    println!("cargo:rerun-if-changed=assets");
    println!("cargo:rerun-if-changed={}", egui_assets.display());
    
    let hdri_path = workspace_root.join("assets/hdri/horn-koppe_spring_1k.hdr");
    println!("cargo:rerun-if-changed={}", hdri_path.display());
    
    BundleBaker::new("system")?
        .with_output_dir(output_dir)
        .add_pipelines(input_dir)?
        .add_pipelines(egui_assets)?
        .add_hdr_ibl(hdri_path, IblBakeOptions::default())
        .execute()?;

    Ok(())
}
