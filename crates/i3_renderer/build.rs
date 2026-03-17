use i3_baker::prelude::*;
use std::path::Path;

fn main() -> Result<()> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_path = Path::new(&manifest_dir);
    
    // Assets are collocated in the crate
    let input_dir = manifest_path.join("assets/pipelines");
    
    // Also include egui assets from crates/i3_egui
    let workspace_root = manifest_path.parent().unwrap().parent().unwrap();
    let egui_assets = workspace_root.join("crates/i3_egui/assets/pipelines");
    
    // Output bundle to the workspace assets folder (at root)
    let output_dir = workspace_root.join("assets");

    println!("cargo:rerun-if-changed=assets");
    println!("cargo:rerun-if-changed={}", egui_assets.display());
    
    BundleBaker::new("system")?
        .with_output_dir(output_dir)
        .add_pipelines(input_dir)?
        .add_pipelines(egui_assets)?
        .execute()?;

    Ok(())
}
