use serde_json::Value;
use std::io::{self, BufRead};

/// Small utility to parse cargo --message-format=json and print a concise report.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let reader: Box<dyn BufRead> = if args.len() > 1 {
        Box::new(io::BufReader::new(std::fs::File::open(&args[1])?))
    } else {
        Box::new(io::BufReader::new(io::stdin()))
    };

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        // Cargo output might contain non-JSON lines (e.g. from build scripts)
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if msg["reason"] == "compiler-message" {
            let message = &msg["message"];
            let level = message["level"].as_str().unwrap_or("unknown");

            // We only care about errors and warnings
            if level != "error" && level != "warning" {
                continue;
            }

            let _rendered = message["rendered"].as_str().unwrap_or("");
            let spans = message["spans"].as_array();

            if let Some(spans) = spans {
                for span in spans {
                    if span["is_primary"].as_bool().unwrap_or(false) {
                        let file = span["file_name"].as_str().unwrap_or("?");
                        let line_start = span["line_start"].as_u64().unwrap_or(0);
                        let col_start = span["column_start"].as_u64().unwrap_or(0);

                        println!(
                            "--- [{}] {}:{}:{} ---",
                            level.to_uppercase(),
                            file,
                            line_start,
                            col_start
                        );
                        println!("{}", message["message"].as_str().unwrap_or(""));

                        // Print contextual snippet if available
                        if let Some(text) = span["text"].as_array() {
                            for t in text {
                                if let Some(line_text) = t["text"].as_str() {
                                    println!("  | {}", line_text.trim_end());
                                }
                            }
                        }
                    }
                }
            }

            // Check for suggested fixes/children
            if let Some(children) = message["children"].as_array() {
                for child in children {
                    if let Some(msg) = child["message"].as_str() {
                        if msg.contains("help:") || msg.contains("suggestion:") {
                            println!("  >> {}", msg);
                        }
                    }
                }
            }
            println!();
        }
    }
    Ok(())
}
