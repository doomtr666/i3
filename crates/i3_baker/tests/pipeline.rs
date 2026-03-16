use i3_baker::pipeline::BundleBaker;
use i3_baker::importers::pipeline_importer::PipelineImporter;
use i3_io::prelude::*;
use std::path::PathBuf;

#[test]
fn test_pipeline_baking_roundtrip() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let asset_dir = manifest_dir.join("tests/assets");
    let output_dir = manifest_dir.join("target/test_assets");

    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir).unwrap();
    }

    // 1. Bake the pipeline
    let baker = BundleBaker::new("test_bundle")
        .unwrap()
        .with_output_dir(&output_dir)
        .add_asset(asset_dir.join("test_pipeline.i3p"), PipelineImporter);

    baker.execute().unwrap();

    // 2. Verify files exist
    let blob_path = output_dir.join("test_bundle.i3b");
    let catalog_path = output_dir.join("test_bundle.i3c");
    assert!(blob_path.exists());
    assert!(catalog_path.exists());

    // 3. Load the baked asset via VFS/Bundle
    // For simplicity in test, we read the catalog manually
    let catalog_data = std::fs::read(&catalog_path).unwrap();
    let _header: i3_io::CatalogHeader = bytemuck::pod_read_unaligned(&catalog_data[..std::mem::size_of::<i3_io::CatalogHeader>()]);
    
    let entry_size = std::mem::size_of::<i3_io::CatalogEntry>();
    let entry_data = &catalog_data[std::mem::size_of::<i3_io::CatalogHeader>()..std::mem::size_of::<i3_io::CatalogHeader>() + entry_size];
    let entry: i3_io::CatalogEntry = bytemuck::pod_read_unaligned(entry_data);

    assert_eq!(entry.name(), "test_pipeline");

    let blob_data = std::fs::read(&blob_path).unwrap();
    // CatalogEntry.offset points to the start of AssetHeader, 
    // we need to skip 64 bytes to get to the PipelineHeader/Blob
    let header_size = 64; 
    let asset_data = &blob_data[(entry.offset as usize + header_size)..(entry.offset + entry.size) as usize];

    // 4. Load as PipelineAsset
    let asset_header = i3_io::AssetHeader {
        asset_type: entry.asset_type,
        ..Default::default()
    };
    println!("Asset data size: {}", asset_data.len());
    println!("First 16 bytes of asset data: {:?}", &asset_data[..16]);
    
    let pipeline = PipelineAsset::load(&asset_header, asset_data).unwrap();

    println!("Loaded pipeline type: {:?}", pipeline.type_info);
    assert_eq!(pipeline.type_info, PipelineType::COMPUTE);
    assert!(pipeline.bytecode.len() > 0);
    assert!(pipeline.reflection_data.len() > 0);
}
