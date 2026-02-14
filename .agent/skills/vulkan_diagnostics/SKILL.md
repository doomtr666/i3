---
name: vulkan_diagnostics
description: Efficiently diagnose and report Vulkan validation errors during runtime.
---

# Vulkan Diagnostics Skill

This skill provides a structured way to handle Vulkan validation layer errors, avoiding the noise of raw validation logs and providing direct links to specifications.

## Instructions

1.  **Always use the launcher**: Run the `i3-vrun` script instead of raw `cargo run`.
2.  **Diagnostics**: The tool will automatically parse the output and provide a concise report when `VALIDATION` messages are detected.
3.  **Cross-platform**:
    - Windows: `.\tools\i3-vrun.ps1 <example_name>`
4.  **Review**: Examine the report for VUIDs (Vulkan Usage IDs) and follow the links to the official Khronos specification.

## Scripts
- `tools/i3-vrun.ps1`: The Windows launcher script.
- `tools/vulkan_diagnostics/src/main.rs`: The core diagnostic parser.

## Examples

### Running Triangle Demo
```powershell
.\tools\i3-vrun.ps1 draw_triangle
```
