use regex::Regex;
use std::collections::HashMap;
use std::io::{self, BufRead};

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let werror = args.iter().any(|arg| arg == "--werror" || arg == "-Werror");
    let verbose = args.iter().any(|arg| arg == "--verbose" || arg == "-v");

    // Regex to extract VUID
    let vuid_re = Regex::new(r"VUID-(?P<vuid>[^\]\s\)]+)").unwrap();
    let panic_re = Regex::new(r"thread '.*' panicked at '(?P<msg>.*)',").unwrap();

    // stdin reader
    let reader = io::BufReader::new(io::stdin());

    if verbose {
        println!("\n--- [I3FX VULKAN DIAGNOSTICS SESSION] ---\n");
    }

    if werror && verbose {
        println!("[!] -Werror mode active: all validation warnings are treated as fatal errors.\n");
    }

    let mut found_validation_error = false;
    let mut found_validation_warning = false;
    let mut found_panic = false;

    // Track stats
    let mut vuid_counts: HashMap<String, usize> = HashMap::new();
    let mut vuid_messages: HashMap<String, String> = HashMap::new();
    let mut panic_messages: Vec<String> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        // 0. Extract VUID if present on ANY line
        let vuid_match = vuid_re
            .captures(&line)
            .and_then(|cap| cap.name("vuid"))
            .map(|m| m.as_str().to_string());

        // 1. Check for Validation Hits
        if line.contains("VALIDATION") || line.contains("Validation Error") || vuid_match.is_some()
        {
            let is_error = line.contains("ERROR") || line.contains("Validation Error");
            // If it's just a continuation line with VUID, assume it belongs to the recent context or treat as error
            // Simplification: validaton messages usually have "ERROR" or "WARNING" in the header line.
            // If we found a VUID but no explicit tag, we assume it's part of an issue.

            if is_error || found_validation_error {
                // sticky error state? No, let's keep it simple.
                // If the line has "ERROR", it's an error.
                if is_error {
                    found_validation_error = true;
                } else if !found_validation_error
                    && (line.contains("WARN") || found_validation_warning)
                {
                    found_validation_warning = true;
                }
                // If we just found a VUID on a random line, we count it.
            }

            // Update stats if we found a NEW VUID
            if let Some(vuid) = vuid_match {
                let count = vuid_counts.entry(vuid.clone()).or_insert(0);
                *count += 1;

                if *count == 1 {
                    vuid_messages.insert(vuid.clone(), trimmed.to_string());
                    // Print header
                    println!("\n--- [VULKAN VALIDATION ISSUE] ---");
                    println!("VUID: {}", vuid);
                    println!(
                        "Doc: https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#{}",
                        vuid
                    );
                    println!("Message: {}", trimmed);
                }
            } else if line.contains("VALIDATION") {
                // It's a header line without VUID (yet).
                // We might want to print it if verbose, or if it's the start of a new unknown error.
                // But simply, if we don't have a VUID, we wait.
            }
            // If count > 1, we suppress output (deduplication)
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

            panic_messages.push(panic_msg.to_string());

            println!("\n!!! [RUST PANIC DETECTED] !!!");
            println!("Message: {}", panic_msg);
            continue;
        }

        // 3. Filtered Passthrough of other important logs
        // Only if verbose
        if verbose {
            if line.contains("ERROR")
                || line.contains("WARN")
                || line.contains("INFO")
                || line.contains("Frame Stats")
                || line.contains("DEBUG")
            {
                println!("{}", trimmed);
            }
        }
    }

    let mut success = !found_panic && !found_validation_error;
    if werror && found_validation_warning {
        success = false;
    }

    if verbose || !success {
        println!("\n--- [SESSION SUMMARY] ---");
    }

    if found_panic {
        println!("RESULT: CRASHED (Panic)");
        for msg in &panic_messages {
            println!(" - {}", msg);
        }
    } else if !success {
        println!("RESULT: FAILED (Validation Issues detected)");
        println!("Unique VUIDs triggered: {}", vuid_counts.len());
        println!("\n{:<40} | {:<3} | {:<40}", "VUID", "N", "Snippet");
        println!("{:-<40}-|-{:-<3}-|-{:-<40}", "", "", "");

        for (vuid, count) in &vuid_counts {
            let full_msg = vuid_messages.get(vuid).map(|s| s.as_str()).unwrap_or("");
            // Strip newlines/tabs from snippet
            let clean_msg = full_msg
                .replace('\n', " ")
                .replace('\r', "")
                .replace('\t', " ");
            let snippet: String = clean_msg.chars().take(40).collect();
            println!("{:<40} | {:<3} | {}...", vuid, count, snippet);
        }
    } else if verbose {
        println!("RESULT: SUCCESS (Zero Validation Errors)");
    }

    if verbose || !success {
        println!("--------------------------\n");
    }

    if !success {
        std::process::exit(1);
    }

    Ok(())
}
