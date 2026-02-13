---
name: rust_diagnostics
description: Efficiently diagnose and report Rust compilation errors using machine-readable output.
---

# Rust Diagnostics Skill

This skill provides a structured way to handle Rust compilation errors, avoiding the noise of raw terminal output.

## Instructions

1.  **Always use the proxy**: Run the `i3-cargo` wrapper instead of raw `cargo`.
2.  **Diagnostics**: The tool will automatically parse the output and provide a concise report.
3.  **Cross-platform**:
    - Windows: `.\tools\i3-cargo.ps1 <command> [args]`
    - Linux: `./tools/i3-cargo.sh <command> [args]`
4.  **Review**: Examine the report for errors/warnings.

## Scripts
- `tools/i3-cargo.ps1`: The Windows proxy script.
- `tools/i3-cargo.sh`: The Linux proxy script.
- `tools/rust_diagnostics/src/main.rs`: The core diagnostic parser.

## Examples

### Selective Check
```powershell
.\tools\i3-cargo.ps1 check -p i3_gfx
```

### Running Tests
```powershell
.\tools\i3-cargo.ps1 test
```

### Full Build
```powershell
.\tools\i3-cargo.ps1 build
```
