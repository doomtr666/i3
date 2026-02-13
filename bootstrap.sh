#!/usr/bin/env bash
# Bootstrap script to download native dependencies
# Run this script before building the project for the first time

set -e

echo "i3 Native Dependencies Bootstrap"
echo ""

# Check if cargo is available
if ! command -v cargo &> /dev/null; then
    echo "Error: cargo not found in PATH"
    echo "   Please install Rust from https://rustup.rs/"
    exit 1
fi

echo "Downloading native dependencies (SDL2, etc.)..."
echo ""

# Run the download script
cargo run --manifest-path third_party/Cargo.toml

echo ""
echo "Bootstrap completed successfully!"
echo "   You can now build the project with: cargo build"
