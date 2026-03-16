use i3_baker::prelude::*;
use i3_baker::importers::pipeline_importer::PipelineImporter;
use i3_baker::writer::BundleWriter;
use std::path::Path;

fn main() -> i3_baker::Result<()> {
    let input_dir = Path::new("assets/system/pipelines");
    let output_dir = Path::new("assets");
    
    println!("Baking pipelines from {}...", input_dir.display());
    
    BundleBaker::new("system")?
        .with_output_dir(output_dir)
        .add_pipelines(input_dir)?
        .execute()?;
    
    println!("Done!");
    Ok(())
}
