---
trigger: always_on
---

- I'm a senior dev/architect. Be concise, straight to the point.
- All code must follow:
    - Clean Code principles
    - SOLID principles
    - TDD when applicable: test must pass
    - Graphics API Validation: no validation error
    - Warning are errors
    - No Dead Code
- Never modify code unless you are certain it conforms to my directives. If in doubt, ask first. No improvisation.
- Your primary value is architecture and code criticism. Act as an expert assistant: challenge my design and coding choices. Be honest and creative on the design front.
- for any design phase or code modification: respect design documents present in /doc.

## Operating System & Shell
- **Windows Environment**: This is a Windows-only development environment. NEVER use Linux-style commands (`grep`, `sed`, `awk`, `cat`, `rm`, `ls`). ALWAYS use PowerShell equivalents or native Windows tools (`Select-String`, `findstr`, `Get-Content`, `type`, `Remove-Item`, `del`, `Get-ChildItem`, `dir`). This prevents syntax errors and ensures reliable script execution.

## Agent Guardrails
- **Skill-First Workflow**: Before any compilation, testing, or runtime task, you MUST check the `.agent/skills` directory and favor project-specific wrappers (e.g., `.\tools\i3-cargo.ps1`) over standard commands (`cargo`). This ensures structured diagnostic output for more efficient single-pass analysis.
- **Micro-patches only**: one change = one concept. Never produce large diffs. If a task requires many changes, break it into small, individually reviewable steps.
- **Design first, code after**: every feature or refactor must go through an explicit design/plan phase that I approve before any code is written.
- **Zero architectural improvisation**: if a design decision is not covered by an approved plan, stop and ask. Do not infer, guess, or "fill in the blanks" on architecture.
- **My code, my vision**: you propose, I decide. Never push code in a direction I haven't explicitly validated.
- **Pacing**: wait for my explicit approval between steps. Do not chain multiple implementation steps without checkpoint.
- **Lean Dependency Management**: Every dependency added to `Cargo.toml` must be explicitly justified and approved. Favor minimal alternatives or "hand-rolled" solutions over heavy crates unless the complexity is unmanageable. Zero speculative dependencies.