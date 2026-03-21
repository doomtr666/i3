# i3 Engine — High Level Design (HLD)

This document outlines the master architecture of the i3 engine, focusing on the project structure and the Hardware Rendering Interface (HRI) decoupling.

## 1. Project Organization (Rust 2024)

The engine is structured as a Rust 2024 workspace. This ensures strict isolation between the high-level rendering logic and the hardware-specific implementations.

### Workspace Structure

```text
i3/
├── crates/
│   ├── i3_gfx/             # Core Engine & Frame Graph (Agnostic)
│   ├── i3_vulkan_backend/   # Vulkan 1.3 Implementation
│   ├── i3_dx12_backend/     # DX12 Implementation (Future)
│   ├── i3_null_backend/     # Logging & Validation Oracle
│   ├── i3_slang/           # Slang compiler wrapper & reflection
│   ├── i3_renderer/        # Default Render Pipeline (Deferred Clustered)
│   ├── i3_baker/           # Asset Baking Toolchain
│   ├── i3_io/              # VFS & Asset Loading Logic
│   ├── i3_bundle/          # Asset Bundle Format & Packing
│   └── i3_egui/            # Egui integration
├── doc/                    # Architecture & Design Docs
│   ├── engine_hld.md       # This file
│   └── frame_graph_design.md
├── examples/               # Usage examples (Viewer, etc.)
├── third_party/            # Native dependencies & build support
│   ├── libs/               # Downloaded binaries (gitignored)
│   └── build-support/      # Shared build scripts
└── tools/                  # Internal development tools (Diagnostics)
```

## 2. Core Components

### Error Management
The engine standardizes on `thiserror` for internal error definitions. This ensure structured, type-safe errors with minimal boilerplate, facilitating robust validation (especially in the Frame Graph and NullBackend).

### Diagnostics & Tracing
The engine uses `tracing` for instrumenting execution. This provides:
- Structured logs with fields (not just strings).
- Hierarchy (Spans) to track pass execution across frames.
- Low-overhead profiling integration.

### `i3_gfx` (The Brain)
- **Frame Graph**: Manages the `Declare -> Compile -> Execute` pipeline.
- **Resource Management**: Handles logical resources (`ResourceId`), lifetimes, and aliasing.
- **Graph Compiler**: Resolves synchronization, barriers, and multi-queue assignments.
- **RenderBackend**: Agnostic abstraction for hardware (Vulkan, DX12, etc.).

## Ergonomics & Best Practices

### The Prelude Pattern
To avoid "import hell" and provide a frictionless experience for users of the engine, every major crate (`i3_xxx`) must provide a `prelude` module.
- **Location**: `pub mod prelude` in `src/lib.rs`.
- **Content**:
    - Essential Traits (those required to use methods on types).
    - Core Handles (`ImageHandle`, `BufferHandle`).
    - Main entry points (`FrameGraph`, `SlangCompiler`).
- **Goal**: A user should be able to do `use i3_gfx::prelude::*;` and have 90% of what they need for common tasks.

### `i3_xx_backend` (The Muscle)
- **RenderBackend Implementation**: Translates logical graph commands into native API calls (`vkCmdXXX`).
- **Memory Management**: Implements physical memory pools and aliasing.
- **Synchronization**: Translates logical transitions into native barriers (`VkImageMemoryBarrier2`, etc.).
- **Submission**: Handles asynchronous GPU submission and timeline semaphore tracking.

### `i3_slang` (Shader Intelligence)
- **Slang Wrapper**: Rust bindings for the Slang compiler.
- **Reflection**: Extracts pipeline layouts, resource bindings, and vertex layouts from shader source.
- **Hot Reload**: Monitors and re-compiles shaders at runtime.

### `i3_io` (Asset Intelligence)
- **Virtual File System (VFS)**: Unified access to raw files and `.i3b` asset bundles.
- **Zero-Copy Loading**: Optimized binary layouts designed for `mmap` and direct CPU/GPU access.
- **Future-Proof**: Architecture considerations for DirectStorage and hardware decompression.

### `i3_baker` (Asset Factory)
- **Incremental Builds**: Dependency tracking to only re-bake what has changed.
- **Parallel Processing**: Multi-threaded baking of assets.
- **Plugin-System**: Extensible architecture for processing different asset types (Mesh, Texture, Slang).

## 3. Hardware Rendering Interface (HRI)

The boundary between `i3_gfx` and the backends is a set of Rust traits.

- **`HriBackend`**: Factory for resources and submission control.
- **`PassContext`**: Agnostic command recorder passed to render passes.
- **`NullBackend`**: A specialized implementation used for CI validation and graph visualization.

## 4. Window Management

To avoid abstraction leaks, the **Window is a backend service**.

- High-level code requests a "Surface" or "Window" from the HRI.
- The `i3_vulkan_backend` handles the native window creation (via SDL2/winit) and the corresponding `VkSurfaceKHR` and `VkSwapchainKHR`.
- This ensures that platform-specific requirements (e.g., Win32 `HWND`, Wayland surfaces) remain completely hidden from the Frame Graph.

## 5. Testing Conventions

To keep the source code concise and avoid "pollution" from test noise, we adopt the **Separate Sub-module** pattern for unit tests.

### Unit Tests
Instead of mixing `#[cfg(test)] mod tests` at the bottom of the source file, we use a sibling file:
- `src/graph/types.rs` (Pure code)
- `src/graph/types.tests.rs` (Unit tests)

In `types.rs`, we only include the tests via a conditional module:
```rust
// types.rs
pub struct ResourceId(u32);
// ...

#[cfg(test)]
#[path = "types.tests.rs"]
mod tests;
```

### Integration Tests
Stored in the `tests/` directory at the crate root. These tests interact with the crate as an external consumer (only accessing public APIs).

### Null-Backend Tests
A specialized category of integration tests. They run complex frame-graph scenarios against the `i3_null_backend` to validate the logical output (barriers, aliasing) in a deterministic way.

## 6. Setup & Dependencies

The engine requires several native dependencies that are managed through a automated bootstrap process.

### Prerequisites
- **Rust**: Latest stable (targeting Rust 2024 edition).
- **Vulkan SDK**: Must be installed and the `VULKAN_SDK` environment variable set.
- **Git**: For version control.

### Initial Setup
Run the bootstrap script from the repository root:
- **Windows**: `.\bootstrap.ps1`
- **Linux/macOS**: `./bootstrap.sh`

### Native Dependency Management
Native libraries (like SDL2) are stored in `third_party/libs/`. 

- **Bootstrap System**: A dedicated Rust tool in `third_party/` handles downloading and extracting binaries to avoid committing large binary blobs to the repo.
- **Build Support**: The `i3_build_support` module (in `third_party/build-support`) provides utilities for `build.rs` scripts to automatically link native libraries and copy necessary DLLs to the `target/` directory during build.
