use i3_baker::Result;
use i3_baker::importers::AssimpImporter;
use i3_baker::pipeline::BakeContext;
use i3_baker::scanner::Scanner;
use i3_baker::writer::BundleWriter;
use std::path::Path;
use tracing::{info, warn};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Parse CLI arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!("Usage: i3_baker <source_dir> <output_dir>");
        println!("\nBakes 3D assets (glTF, FBX, OBJ, etc.) into .i3b bundles.");
        return Ok(());
    }

    let source_dir = Path::new(&args[1]);
    let output_dir = Path::new(&args[2]);

    // Validate paths
    if !source_dir.exists() {
        return Err(i3_baker::BakerError::Plugin(format!(
            "Source directory does not exist: {}",
            source_dir.display()
        )));
    }

    // Create output directory if needed
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir).map_err(|e| i3_baker::BakerError::Os {
            path: output_dir.to_path_buf(),
            source: e,
        })?;
    }

    info!(
        "Baking assets from {} to {}...",
        source_dir.display(),
        output_dir.display()
    );

    // Set up scanner with importers
    let mut scanner = Scanner::new();
    scanner.register_importer(Box::new(AssimpImporter::new()));

    // Scan source directory
    let source_files = scanner.scan_directory(source_dir)?;
    info!("Found {} source files to process", source_files.len());

    if source_files.is_empty() {
        warn!("No supported source files found.");
        return Ok(());
    }

    // Create bundle writer
    let blob_path = output_dir.join("assets.i3b");
    let catalog_path = output_dir.join("assets.i3c");
    let mut writer = BundleWriter::new(&blob_path)?;

    // Process each source file
    let mut total_outputs = 0;
    for source_file in &source_files {
        info!("Processing: {}", source_file.path.display());

        // Get the importer for this file
        let importer = scanner.get_importer(source_file.importer_index).unwrap();

        // Create bake context
        let ctx = BakeContext::new(&source_file.path, output_dir);

        // Import the file
        match importer.import(&source_file.path) {
            Ok(imported_data) => {
                // Extract outputs
                match importer.extract(imported_data.as_ref(), &ctx) {
                    Ok(outputs) => {
                        for output in &outputs {
                            info!("  -> {} ({} bytes)", output.name, output.data.len());

                            // Add to bundle using the BakeOutput directly
                            writer.add_bake_output(output)?;
                            total_outputs += 1;
                        }
                    }
                    Err(e) => {
                        warn!("  Failed to extract: {}", e);
                    }
                }
            }
            Err(e) => {
                warn!("  Failed to import: {}", e);
            }
        }
    }

    // Finalize bundle
    writer.finish(&catalog_path)?;

    info!(
        "Baking complete: {} assets written to {}",
        total_outputs,
        blob_path.display()
    );

    Ok(())
}
