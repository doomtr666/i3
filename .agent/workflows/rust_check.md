---
description: Unified workflow for checking Rust code using structured diagnostics.
---

# Rust Check Workflow

Use this workflow to quickly identify and fix compilation errors.

1.  **Run Check**: Execute cargo check and capture JSON output.
// turbo
2.  **Diagnose**: Use the `rust_diagnostics` skill to parse the output.
    ```powershell
    .\tools\i3-cargo.ps1 check
    ```
3.  **Review**: Examine the concise report and fix identified issues.
4.  **Repeat**: Run the check again to verify the fix.
