use i3_baker::importers::AssimpImporter;
use i3_baker::pipeline::{BakeContext, Importer};
use i3_baker::writer::BundleWriter;
use i3_io::AssetHeader;
use i3_io::asset::Asset;
use i3_io::mesh::{MESH_ASSET_TYPE, MeshAsset};
use i3_io::scene_asset::{SCENE_ASSET_TYPE, SceneAsset};
use i3_io::vfs::{BundleBackend, Vfs};

#[test]
fn test_full_baking_pipeline() {
    let temp_dir = std::env::temp_dir().join("i3_baker_test_full");
    if temp_dir.exists() {
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
    std::fs::create_dir_all(&temp_dir).unwrap();

    // Create a minimal valid OBJ file for testing
    let sample_obj = temp_dir.join("cube.obj");
    std::fs::write(&sample_obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
    let source_path = sample_obj;

    let blob_path = temp_dir.join("test_assets.i3b");
    let catalog_path = temp_dir.join("test_assets.i3c");

    // 1. Run Baker Pipeline
    let importer = AssimpImporter::new();
    let ctx = BakeContext::new(&source_path, &temp_dir);

    let imported_data = importer.import(&source_path).expect("Failed to import OBJ");
    let outputs = importer
        .extract(imported_data.as_ref(), &ctx)
        .expect("Failed to extract assets");

    assert!(
        !outputs.is_empty(),
        "Should have extracted at least one asset"
    );

    let mut writer = BundleWriter::new(&blob_path).expect("Failed to create BundleWriter");
    for output in &outputs {
        writer
            .add_bake_output(output)
            .expect("Failed to add bake output");
    }
    writer
        .finish(&catalog_path)
        .expect("Failed to finish bundle");

    // 2. Verify with VFS and i3_io
    let backend = BundleBackend::mount(&catalog_path, &blob_path).expect("Failed to mount bundle");
    let vfs = Vfs::new();
    vfs.mount(Box::new(backend));

    let mut mesh_found = false;
    let mut scene_found = false;

    for output in &outputs {
        let file = vfs.open(&output.name).expect("Failed to open asset in VFS");
        let data = file.as_slice().expect("Failed to get file slice");

        let header_size = std::mem::size_of::<AssetHeader>();
        let header: &AssetHeader = bytemuck::from_bytes(&data[..header_size]);
        let asset_data = &data[header_size..];

        if output.asset_type == MESH_ASSET_TYPE {
            let mesh = MeshAsset::load(header, asset_data).expect("Failed to load MeshAsset");
            assert!(mesh.header.vertex_count > 0);
            mesh_found = true;
        } else if output.asset_type == SCENE_ASSET_TYPE {
            let scene = SceneAsset::load(header, asset_data).expect("Failed to load SceneAsset");
            assert!(scene.header.object_count > 0);
            scene_found = true;
        }
    }

    assert!(mesh_found, "No mesh asset was baked/verified");
    assert!(scene_found, "No scene asset was baked/verified");
}
