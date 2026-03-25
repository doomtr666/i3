use i3_baker::importers::AssimpImporter;
use i3_baker::pipeline::BundleBaker;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    let manifest_path = Path::new(&manifest_dir);

    // Find the actual target directory (e.g. target/debug) from OUT_DIR
    // OUT_DIR = .../target/debug/build/viewer-hash/out
    let out_dir = std::env::var("OUT_DIR")?;
    let target_dir = Path::new(&out_dir)
        .parent()
        .unwrap() // viewer-hash
        .parent()
        .unwrap() // build
        .parent()
        .unwrap(); // debug/release

    let helmet_src = manifest_path
        .join("../../assets/gltf-sample-assets/Models/DamagedHelmet/glTF-Binary/DamagedHelmet.glb");
    let sponza_src =
        manifest_path.join("../../assets/gltf-sample-assets/Models/Sponza/glTF/Sponza.gltf");
    let bistro_ext_src = manifest_path.join("../../assets/Bistro_v5_2/BistroExterior.fbx");
    let bistro_int_src = manifest_path.join("../../assets/Bistro_v5_2/BistroInterior.fbx");
    let normal_tangent_src = manifest_path.join("../../assets/gltf-sample-assets/Models/NormalTangentTest/glTF/NormalTangentTest.gltf");
    let normal_tangent_mirror_src = manifest_path.join("../../assets/gltf-sample-assets/Models/NormalTangentMirrorTest/glTF/NormalTangentMirrorTest.gltf");

    BundleBaker::new("viewer_scenes")?
        .with_output_dir(target_dir)
        .add_asset(&helmet_src, AssimpImporter::new())
        .add_asset(&sponza_src, AssimpImporter::new())
        .add_asset(&bistro_ext_src, AssimpImporter::new())
        .add_asset(&bistro_int_src, AssimpImporter::new())
        .add_asset(&normal_tangent_src, AssimpImporter::new())
        .add_asset(&normal_tangent_mirror_src, AssimpImporter::new())
        .execute()?;

    // Copy shaders to target directory
    let shader_src = manifest_path.join("../../crates/i3_renderer/assets/shaders");
    let shader_dst = target_dir.join("shaders");
    copy_dir(&shader_src, &shader_dst)?;

    // Tell cargo to rerun if assets or shaders change
    println!("cargo:rerun-if-changed=../../assets");
    println!("cargo:rerun-if-changed=../../crates/i3_renderer/assets/shaders");

    Ok(())
}

fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    if !dst.exists() {
        std::fs::create_dir_all(dst)?;
    }
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir(&entry.path(), &dst.join(entry.file_name()))?;
        } else {
            std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
        }
    }
    Ok(())
}
