use clap::Parser;
use i3_io::{CatalogEntry, CatalogHeader};
use prettytable::{format, row, Table};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the .i3c catalog file
    #[arg(short, long)]
    catalog: PathBuf,

    /// Filter by asset name (substring)
    #[arg(short, long)]
    filter: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut file = File::open(&args.catalog)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    if buffer.len() < std::mem::size_of::<CatalogHeader>() {
        return Err("File too small for catalog header".into());
    }

    let header: CatalogHeader =
        *bytemuck::from_bytes(&buffer[0..std::mem::size_of::<CatalogHeader>()]);

    if header.magic != CatalogHeader::MAGIC {
        return Err("Invalid catalog magic".into());
    }

    println!("Bundle Catalog: {:?}", args.catalog);
    println!("Version: {}", header.version);
    println!("Entries: {}", header.count);
    println!();

    let entry_size = std::mem::size_of::<CatalogEntry>();
    let entries_start = std::mem::size_of::<CatalogHeader>();
    let entries_end = entries_start + (header.count as usize * entry_size);

    if buffer.len() < entries_end {
        return Err("File too small for catalog entries".into());
    }

    let entries: &[CatalogEntry] = bytemuck::cast_slice(&buffer[entries_start..entries_end]);

    let mut table = Table::new();
    table.set_format(*format::consts::FORMAT_NO_BORDER_LINE_SEPARATOR);
    table.set_titles(row!["Name", "UUID", "Type", "Size", "Offset", "Comp"]);

    let mut total_size = 0u64;
    let mut filtered_count = 0;

    for entry in entries {
        let name = entry.name();

        if let Some(ref filter) = args.filter {
            if !name.to_lowercase().contains(&filter.to_lowercase()) {
                continue;
            }
        }

        let asset_id = Uuid::from_bytes(entry.asset_id);
        let asset_type = Uuid::from_bytes(entry.asset_type);

        let type_name = if asset_type == i3_io::mesh::MESH_ASSET_TYPE {
            "Mesh"
        } else if asset_type == i3_io::scene_asset::SCENE_ASSET_TYPE {
            "Scene"
        } else {
            "Unknown"
        };

        let comp_name = match entry.compression {
            0 => "None",
            1 => "Zstd",
            2 => "GDefl",
            _ => "???",
        };

        table.add_row(row![
            name,
            asset_id,
            type_name,
            format_size(entry.size),
            format!("0x{:X}", entry.offset),
            comp_name
        ]);

        total_size += entry.size;
        filtered_count += 1;
    }

    table.printstd();

    println!();
    println!("Total Assets: {}/{}", filtered_count, header.count);
    println!("Total Size:   {}", format_size(total_size));

    Ok(())
}

fn format_size(size: u64) -> String {
    if size < 1024 {
        format!("{} B", size)
    } else if size < 1024 * 1024 {
        format!("{:.2} KB", size as f32 / 1024.0)
    } else {
        format!("{:.2} MB", size as f32 / (1024.0 * 1024.0))
    }
}
