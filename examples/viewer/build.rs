use i3_baker::importers::AssimpImporter;
use i3_baker::pipeline::{BakeContext, Importer};
use i3_baker::writer::BundleWriter;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../../assets/DamagedHelmet.glb");
    println!("cargo:rerun-if-changed=../../assets/Sponza/glTF/Sponza.gltf");
    println!("cargo:rerun-if-changed=../../assets/Sponza/glTF/Sponza.bin");

    let out_dir = std::env::var("OUT_DIR")?;
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;

    let assets_dir = Path::new(&manifest_dir).join("assets");
    if !assets_dir.exists() {
        std::fs::create_dir_all(&assets_dir)?;
    }

    let blob_path = assets_dir.join("viewer_scenes.i3b");
    let catalog_path = assets_dir.join("viewer_scenes.i3c");

    let importer = AssimpImporter::new();
    let mut writer = BundleWriter::new(&blob_path)?;

    // 1. Bake DamagedHelmet
    let helmet_src = Path::new(&manifest_dir).join("../../assets/DamagedHelmet.glb");
    let ctx_helmet = BakeContext::new(&helmet_src, &out_dir);
    let data_helmet = importer.import(&helmet_src)?;
    let outputs_helmet = importer.extract(data_helmet.as_ref(), &ctx_helmet)?;
    for output in &outputs_helmet {
        writer.add_bake_output(output)?;
    }

    // 2. Bake Sponza
    let sponza_src = Path::new(&manifest_dir).join("../../assets/Sponza/glTF/Sponza.gltf");
    if sponza_src.exists() {
        let ctx_sponza = BakeContext::new(&sponza_src, &out_dir);
        let data_sponza = importer.import(&sponza_src)?;
        let outputs_sponza = importer.extract(data_sponza.as_ref(), &ctx_sponza)?;
        for output in &outputs_sponza {
            writer.add_bake_output(output)?;
        }
    } else {
        println!(
            "cargo:warning=Sponza.gltf not found at {:?}, skipping Sponza bake",
            sponza_src
        );
    }

    writer.finish(&catalog_path)?;

    Ok(())
}
