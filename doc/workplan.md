# i3 Engine -- Roadmap & Remaining Tasks

This document tracks the technical debt, design gaps, and upcoming features for the i3 engine.

---

## 1. Project Overview

The i3 engine is a Rust 2024 workspace targeting high-end desktop rendering with Vulkan 1.3. It implements a Frame Graph pattern with deferred clustered shading.

### Workspace Members

| Crate | Role | LOC approx | Maturity |
|---|---|---|---|
| `i3_gfx` | Frame Graph core, HRI abstraction | ~2500 | Functional |
| `i3_vulkan_backend` | Vulkan 1.3 implementation | ~5500 | Functional |
| `i3_null_backend` | Validation oracle | ~550 | Basic |
| `i3_slang` | Slang shader compiler wrapper | ~560 | Functional |
| `i3_renderer` | Deferred clustered shading | ~4000 | Functional but incomplete |
| `i3_io` | VFS, binary formats, asset loading | ~1200 | Functional |
| `i3_baker` | Asset baking pipeline | ~1600 | Functional |
| `i3_egui` | Egui UI integration layer | ~350 | MVP |
| `i3_bundle` | CLI bundle inspector | ~130 | Basic |
| `examples/` | draw_triangle, compute_mandelbrot, deferred_stress, viewer | ~1400 | Working |

### Dependency Graph

```mermaid
graph TD
    A[examples/viewer] --> B[i3_renderer]
    A --> C[i3_vulkan_backend]
    A --> D[i3_io]
    B --> E[i3_gfx]
    B --> F[i3_slang]
    B --> L[i3_egui]
    L --> E
    C --> E
    F --> E
    D --> G[memmap2 + bytemuck + uuid]
    H[i3_baker] --> D
    H --> I[russimp + image + intel_tex_2]
    J[i3_null_backend] --> E
    K[i3_bundle] --> D
```

---

## 2. Remaining Issues & Technical Debt

### 2.1 Core & Infrastructure (i3_gfx, i3_vulkan_backend, i3_io)

| ID | Component | Severity | Description |
|---|---|---|---|
| GFX-03 | i3_gfx | High | `compiler.rs` is too large (~1000 LOC). Split into symbol_table, node_storage, etc. |
| GFX-04 | i3_gfx | Medium | `consume_erased` panics on missing symbol; should return Result. |
| GFX-06 | i3_gfx | Medium | Memory aliasing (AliasingPlan) described in design but not implemented. |
| GFX-07 | i3_gfx | **DONE** | ~~Multi-queue support (async compute/transfer) not implemented.~~ Implemented: `QueueType`, `prefer_async`, `BatchStep`-based sub-batch splitting, timeline semaphore cross-queue sync, queue family ownership transfers. Working with 3 queues visible in Nsight, no validation errors. |
| GFX-08 | i3_gfx | Low | Dead node elimination not implemented. |
| GFX-MQ-01 | i3_vulkan_backend | High | `begin_frame` waits only on the **graphics** timeline semaphore before resetting all frame contexts. Compute and transfer pools are reset without waiting for those queues to finish — command buffers in flight can be invalidated. Fix: add `last_completion_value` per `QueueContext` and wait on each queue's timeline in `begin_frame`. (`submission.rs:114-196`) |
| GFX-MQ-02 | i3_vulkan_backend | Medium | `get_queue_family()` calls `.unwrap()` on `backend.compute` / `backend.transfer` unconditionally. Safe today only because `assign_queues` never routes to AsyncCompute when `capabilities.async_compute=false` — a fragile implicit invariant. Fix: return `Option<u32>` or fall back to `graphics_family`. (`sync.rs:34-40`) |
| GFX-MQ-03 | i3_vulkan_backend | Medium | `sanitize_stages` silently converts unsupported graphics stages (e.g. `FRAGMENT_SHADER`) to `TOP_OF_PIPE` on compute/transfer queues without logging. Weakens synchronization invisibly. Fix: add `tracing::warn!` when the fallback fires. (`sync.rs:327-342`) |
| IO-01 | i3_io | High | `AssetHandle::get()`/`wait_loaded()` return refs that may outlive the lock (potential UB). Use Arc<T>. |
| IO-03 | i3_io | Medium | Manual unsafe pointer cast in `texture.rs` load. Use match `bytemuck` patterns. |
| VK-03 | i3_vulkan_backend | Low | Format conversion audit needed for recent Vulkan additions. |

### 2.2 Renderer & Shading (i3_renderer)

| ID | Severity | Description |
|---|---|---|
| RN-02 | High | Normal mapping not utilized in deferred resolve. GBuffer normal lacks tangent-space map sampling. |
| RN-03 | Medium | Buffer sizes in `gpu_buffers.rs` are magic numbers. Derive from constants. |
| RN-04 | High | No GPU culling pass (GPUCull). Currently uses CPU-side draw commands. |
| RN-05 | High | No ZPrePass implemented. |
| RN-06 | Medium | No forward transparency pass. |
| RN-07 | Info | No RT support (BLAS/TLAS). Planned for future phases. |
| RN-09 | Low | `LightData` in `scene.rs` needs `repr(C)` padding check for GPU compatibility. |

### 2.3 Tools (i3_baker, i3_bundle, i3_egui)

| ID | Component | Severity | Description |
|---|---|---|
| BK-01 | i3_baker | Medium | Dead `PipelineNode` abstraction review. |
| BK-05 | i3_baker | Low | No tangent recalculation when Assimp metadata is missing. |
| BN-01 | i3_bundle | Medium | Show fragmentation info in bundle inspector (gaps, padding, overhead). |
| BN-02 | i3_bundle | Low | Missing `compact`/`defragment` command for optimized production bundles. |
| EG-I01 | i3_egui | Medium | Support user textures beyond the font atlas. |
| EG-I02 | i3_egui | Medium | Scissoring not implemented in `execute()`. |
| EG-I03 | i3_egui | Low | VB/IB re-allocated every frame. Use persistent or ring buffers. |

---

## 3. Action Plan: Upcoming Phases

### Phase 1: Safety & Foundation
- **[DONE]** Multi-queue async compute + transfer (GFX-07).
- **[TODO]** Fix GFX-MQ-01: per-queue `last_completion_value` and wait in `begin_frame`.
- **[TODO]** Fix GFX-MQ-02: safe `get_queue_family()` with fallback.
- **[TODO]** Fix GFX-MQ-03: log warn in `sanitize_stages` on fallback.
- Refactor `AssetHandle` accessors to return `Arc<T>` (Fix IO-01/IO-02).
- Clean up unsafe casts in `texture.rs`.
- Split large files (`compiler.rs`).
- Implement disk-based `VkPipelineCache` for faster cold starts.

### Phase 2: Advanced Rendering Features
- **P2.1: Normal Mapping**: Update GBuffer to include tangent/bitangent and sample normal maps in deferred resolve.
- **P2.2: ZPrePass**: Implement depth-only pass for early Z optimization.
- **P2.3: GPU-Driven Pipeline**: Implement compute-based frustum culling and `draw_indexed_indirect` support.
- **P2.4: Forward Transparency**: Add forward pass group for transparent objects.

### Phase 3: Hardware Evolution
- **P3.1: Ray Tracing Support**: Add BLAS/TLAS types to i3_gfx and implement backend logic for RT shadows/queries.
- **P3.2: Multi-GPU Selection**: Implement explicit GPU selection via config and CLI flags.

### Phase 4: Ergonomics & Polish
- **P4.1: Shading DSL**: High-level material description language that compiles to `.i3p` assets.
- **P4.2: Baker Progress**: Real-time progress reporting during long bakes (e.g., Sponza).
- **P4.3: Egui Polish**: Scissoring, DPI support, and multi-texture management.

---

## 4. Documentation & Quality
- Update `engine_hld.md` to reflect current workspace structure.
- Annotate all design documents with current implementation status.
- Reconcile testing conventions between documentation and code.
- Implement VFS unit tests and renderer-level NullBackend integration tests.

---

## 5. Documentation Gaps (frame_graph_design.md vs Code)

`doc/frame_graph_design.md` was written before the implementation and is partially outdated.
The following table summarizes the delta — a doc update pass is needed.

| Topic | Doc says | Code reality |
|---|---|---|
| `RenderPass::domain()` | Required method returning `PassDomain` | Removed — domain is auto-inferred by compiler from resource declarations |
| `RenderPass::declare()` | Method name | Renamed to `record()` in code |
| `RenderPass::prefer_async()` | Not mentioned | Present in trait; default = `true`; controls async queue routing |
| `CommandBatch` / `BatchStep` | Not mentioned | Core submission primitive; `BatchStep::{Command,Wait,Signal}` drives sub-batch splitting |
| Multi-queue sync | Timeline semaphores described abstractly | Concrete: ordered `BatchStep` steps, sub-batch splitting at `Signal` boundaries, per-queue `cpu_timeline` |
| Memory aliasing | Described as "from day one" | **Not yet implemented** (see GFX-06) |
| `PassContext` | Enum with `Gpu`/`Cpu` variants | Implemented as a trait |
| `PassBuilder::add_node()` | FnOnce closure API | Actual API: `add_pass(&mut dyn RenderPass)` / `add_owned_pass<P: RenderPass>` |
| GFX-07 status | "Multi-queue not implemented" | Fully implemented and working |
