use i3_baker::importers::AssimpImporter;
use i3_baker::pipeline::BundleBaker;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    let manifest_path = Path::new(&manifest_dir);

    let helmet_src = manifest_path.join("../../assets/DamagedHelmet.glb");
    let sponza_src = manifest_path.join("../../assets/Sponza/glTF/Sponza.gltf");

    BundleBaker::new("viewer_scenes")?
        .add_asset(&helmet_src, AssimpImporter::new())
        .add_asset(&sponza_src, AssimpImporter::new())
        .execute()?;

    Ok(())
}
