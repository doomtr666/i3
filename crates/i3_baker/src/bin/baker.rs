use i3_baker::prelude::*;
use std::path::Path;

fn main() -> i3_baker::Result<()> {
    let renderer_pipelines = Path::new("crates/i3_renderer/assets/pipelines");
    let egui_pipelines = Path::new("crates/i3_egui/assets/pipelines");
    let output_dir = Path::new("assets");
    
    println!("Baking system bundle...");
    
    BundleBaker::new("system")?
        .with_output_dir(output_dir)
        .add_pipelines(renderer_pipelines)?
        .add_pipelines(egui_pipelines)?
        .execute()?;
    
    println!("Done!");
    Ok(())
}
