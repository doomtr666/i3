use regex::Regex;
use std::collections::HashSet;
use std::io::{self, BufRead};

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let werror = args.iter().any(|arg| arg == "--werror" || arg == "-Werror");

    let vuid_re = Regex::new(r"VUID-(?P<vuid>[^\]\s]+)").unwrap();
    let panic_re = Regex::new(r"thread '.*' panicked at '(?P<msg>.*)',").unwrap();
    let reader = io::BufReader::new(io::stdin());

    println!("\n--- [I3FX VULKAN DIAGNOSTICS SESSION] ---\n");
    if werror {
        println!("[!] -Werror mode active: all validation warnings are treated as fatal errors.\n");
    }

    let mut found_validation_error = false;
    let mut found_validation_warning = false;
    let mut found_panic = false;
    let mut last_vuid = String::new();
    let mut seen_vuids = HashSet::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        // 1. Check for Validation Hits
        if line.contains("VALIDATION") || line.contains("Validation Error") {
            let is_error = line.contains("ERROR") || line.contains("Validation Error");
            if is_error {
                found_validation_error = true;
            } else {
                found_validation_warning = true;
            }

            let vuid = vuid_re
                .captures(&line)
                .and_then(|cap| cap.name("vuid"))
                .map(|m| m.as_str().to_string())
                .unwrap_or_else(|| "Unknown".to_string());

            if vuid != last_vuid {
                let tag = if is_error {
                    "VULKAN VALIDATION ERROR"
                } else {
                    "VULKAN VALIDATION WARNING"
                };
                println!("\n--- [{}] ---", tag);
                println!("VUID: {}", vuid);
                if vuid != "Unknown" {
                    println!(
                        "Doc: https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#{}",
                        vuid
                    );
                }
                println!("Message: {}", trimmed);
                last_vuid = vuid.clone();
                seen_vuids.insert(vuid);
            }
            continue;
        }

        // 2. Check for Rust Panics
        if line.contains("panicked at") {
            found_panic = true;
            let panic_msg = panic_re
                .captures(&line)
                .and_then(|cap| cap.name("msg"))
                .map(|m| m.as_str())
                .unwrap_or(trimmed);

            println!("\n!!! [RUST PANIC DETECTED] !!!");
            println!("Message: {}", panic_msg);
            continue;
        }

        // 3. Filtered Passthrough
        if line.contains("ERROR")
            || line.contains("WARN")
            || line.contains("Frame Stats")
            || line.contains("Starting")
        {
            println!("{}", trimmed);
        }
    }

    let mut success = !found_panic && !found_validation_error;
    if werror && found_validation_warning {
        success = false;
    }

    println!("\n--- [SESSION SUMMARY] ---");
    if found_panic {
        println!("RESULT: CRASHED (Panic)");
    } else if !success {
        println!("RESULT: FAILED (Validation Issues detected)");
        println!("Unique VUIDs triggered: {}", seen_vuids.len());
    } else {
        println!("RESULT: SUCCESS (Zero Validation Errors)");
    }
    println!("--------------------------\n");

    if !success {
        std::process::exit(1);
    }

    Ok(())
}
