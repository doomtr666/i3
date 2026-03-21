# i3 Engine

A high-performance, data-oriented rendering engine built in Rust 2024, focusing on modern Vulkan features and a decouplable architecture.

> [!NOTE]
> This is currently an **educational playground** and a hobbyist project for architectural experimentation. It serves as a personal "sandbox" for testing advanced rendering techniques in Rust.

## Core Pillars

- **Modern Vulkan 1.3+**: Built from the ground up to leverage `dynamic_rendering`, `synchronization2`, and timeline semaphores.
- **Frame Graph Architecture**: Automatic synchronization and memory aliasing. Pass authors focus on logic, the engine handles barriers.
- **Clustered Deferred Shading**: Support for thousands of lights via screen-space tiles and depth slicing.
- **GPU-Driven Pipeline**: Minimal CPU overhead with persistent GPU buffers and indirect drawing.
- **Zero-Copy Asset Pipeline**: Optimized `.i3b` bundles for direct mapping into GPU memory.

## Workspace Structure

The engine is organized into several specialized crates:

- `i3_gfx`: Core engine logic and the Frame Graph abstraction.
- `i3_vulkan_backend`: Hardware implementation for Vulkan 1.3.
- `i3_null_backend`: Logging & Validation Oracle for CI and testing.
- `i3_renderer`: The default render pipeline (Deferred Clustered Shading).
- `i3_baker`: Offline asset processing toolchain (glTF, Shaders, Textures).
- `i3_io`: Virtual File System (VFS) and asynchronous asset loading.
- `i3_slang`: Integration with the Slang shader compiler and reflection system.
- `i3_bundle`: Asset bundle format and packing utilities.
- `i3_egui`: Egui integration for tools and debugging.

## Documentation

Comprehensive design and architecture documents are available in the [`/doc`](./doc) directory:

- [**High Level Design (HLD)**](./doc/engine_hld.md): Overall architecture and workspace organization.
- [**Frame Graph Design**](./doc/frame_graph_design.md): In-depth look at the synchronization and execution model.
- [**Renderer Design**](./doc/renderer_design.md): Details on the Clustered Shading and GPU-driven architecture.
- [**Baker Architecture**](./doc/baker_design.md): Overview of the asset processing and bundle formats.
- [**Engine Conventions**](./doc/engine_conventions.md): Coordinate systems, units, and coding standards.

## Performance & Vision

The i3 engine is designed for efficiency and sets a solid foundation for future development:
- **Instant Loading**: The `i3_baker` and `i3_io` system can process and load the full **Lumberyard Bistro (Exterior)** scene in **under one second** on modern hardware.
- **Current State**: Focus is currently on core structure and robust import of complex scenes (Assimp/gltf). 
- **Minimal Rendering**: Implementation is currently limited to a high-performance **Deferred Clustered** path with basic **Tonemapping**. This is a functional baseline ("work in progress") that will be expanded with advanced features as the architecture matures.

## Getting Started

### Prerequisites
- **Rust**: Ensure you are using the latest stable Rust.
- **Vulkan SDK**: Required for development and validation.
- **Bootstrap**: Run `.\bootstrap.ps1` (Windows) to download native dependencies.
- **Assets**: Run `.\download-assets.ps1` (Windows) to clone the Khronos `glTF-Sample-Assets` repository and NVIDIA Bistro into `/assets`.

### Running the Viewer
```powershell
# 1. Download assets
.\download-assets.ps1

# 2. Run the viewer (will trigger asset baking on first run)
cargo run -p viewer
```

## The Solitary Craftsman

I am building i3 as a solo developer—a "solitary craftsman" tackling an ambitious project. To maintain this pace and complexity, I leverage **Antigravity**, an agentic AI coding assistant. 

The project's health and evolution are supported by specialized tools and custom coding rules found in the [`.agent`](./.agent) directory (`rules/` and `skills/`), which allow Antigravity to operate as a high-tier pair programmer, strictly adhering to the "Craftsman" vision.

## License
Licensed under [GPLv3](./LICENSE.txt).

