#!/bin/bash
# i3-cargo.sh - Proxy command to run cargo with automated AI-friendly diagnostics
TEMP_FILE=$(mktemp)
trap 'rm -f "$TEMP_FILE"' EXIT

cargo "$@" --message-format=json 2>/dev/null > "$TEMP_FILE"
cargo run --quiet --manifest-path tools/rust_diagnostics/Cargo.toml -- "$TEMP_FILE"
