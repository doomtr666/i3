# i3fx Frame Graph — Architecture Design

## Problem Statement

Explicit GPU APIs (Vulkan, DX12) require manual synchronization barriers between resource state transitions. This creates **implicit coupling** between render passes that should be independent. The Frame Graph solves this by making **the engine** responsible for synchronization, while the **pass author** focuses purely on rendering logic.

**3rd attempt.** Previous failures: deferred recording had awkward parallelization; secondary command buffers were too slow.

## State of the Art & Key References

| Reference | Year | Key Contribution |
|---|---|---|
| Frostbite FrameGraph (Wihlidal, GDC) | 2017 | Established Declare/Compile/Execute pattern, transient resources, memory aliasing |
| Granite Render Graph (themaister) | 2017 | Deep Vulkan implementation, CONCURRENT queue sharing, practical aliasing |
| `VK_KHR_dynamic_rendering` | 2021 | Eliminates VkRenderPass/VkFramebuffer objects. Implementation convenience, not structural. |
| Cyclic Render Graphs (Dolp, Vulkanised) | 2025 | Graph partitioning for cyclic dependencies (temporal reprojection, iterative denoising), SSA/Phi-node approach |

**Our design builds on Frostbite's core pattern.** `VK_KHR_dynamic_rendering` simplifies the backend implementation but is not architecturally structuring — the graph compiler could generate `VkRenderPass` objects at compile time regardless.

---

## Platform Scope & API Baseline

**Target:** High-end open source engine. The niche Godot doesn't cover.

| | Decision | Rationale |
|---|---|---|
| **Platforms** | Windows, Linux (console later) | Desktop high-end focus |
| **No mobile** | Deliberate | Avoids leveling down abstractions (wgpu trap) |
| **No macOS** | Deliberate | Metal is a dead-end for high-end; MoltenVK is a wrapper. Community PRs welcome, not a project goal. |
| **Primary API** | **Vulkan 1.3** | Open standard, covers desktop GPU since ~2018 |
| **Validation API** | **DX12** | Validates RHI decoupling, covers Windows without Vulkan |
| **No OpenGL** | Deliberate | Would pull every design decision downward. GL lacks explicit barriers, flexible compute, universal bindless. Time better spent on renderer. |

### Vulkan 1.3 Baseline — Key Features We Rely On

- `VK_KHR_dynamic_rendering` — no VkRenderPass/VkFramebuffer management
- `VK_KHR_synchronization2` — modern barrier API (`VkDependencyInfo`, split pipeline stages)
- Timeline semaphores — cross-queue synchronization
- Buffer device address — bindless buffer access
- Descriptor indexing — bindless textures/samplers

### Optional High-End Features (capability-gated)

- **Ray Tracing** (`VK_KHR_ray_tracing_pipeline`, `VK_KHR_acceleration_structure`)
- **Mesh Shaders** (`VK_EXT_mesh_shader`)
- **Hardware RT + Mesh Shaders** combo for GPU-driven rendering

## Design Principles

1. **Pass authors never touch barriers.** They declare what they use, the engine does the rest.
2. **Parallel by default.** Independent passes record in parallel on separate threads.
3. **Single recording pass.** No deferred → resolve → re-record. Declare lightweight, compile fast, record once.
4. **Multi-queue transparent.** Async compute/transfer supported natively; falls back silently on single-queue GPUs.
5. **Memory aliasing from day one.** Transient resources share memory when lifetimes don't overlap.
6. **Hierarchical Scoping.** Both CPU data and GPU resources live in a Scoped Symbol Table.

---

## The Node Invariant

> **The Graph is a tree of Nodes. A Node is either a leaf (Render Pass) or a branch (Scoped Node).**

1. **Atomic Leaf.** A Leaf (Pass) is an uninterruptible sequence of GPU commands.
2. **Recursive Branch.** A Branch (Group) encapsulates a sub-tree of nodes and manages a local symbol scope.
3. **No mid-pass state transitions.** If a pass needs a resource in two different states (e.g., compute-write then shader-read), that's **two passes**, not one.
4. **Resource usage is declared, not discovered.** The `declare()` call is the **complete** and **exhaustive** contract. The `execute()` call must not use any resource not declared.
5. **Global Scope Invariant.** Services (AssetLoader, ECS, Physics) live in a persistent **Global Scope** that outlives the frame.

### What this enables
- **Barrier resolution is purely a graph-level problem.** The compiler only needs to reason about transitions *between* nodes, never within them.
- **Parallelism is clean.** Any two passes without a data dependency can execute concurrently.
- **Separation of Initialization.** Passes boot up once using the **Global Scope**, avoiding per-frame lookups for static services.

### What this enables
- **Barrier resolution is purely a graph-level problem.** The compiler only needs to reason about transitions *between* nodes, never within them.
- **Parallelism is clean.** Any two passes without a data dependency can execute concurrently — no risk of hidden internal sync requirements.
- **Error detection.** A pass that violates this invariant (e.g., undeclared resource access) can be caught by validation layers or debug tooling.

### Examples

| Scenario | Valid? | Why |
|---|---|---|
| Pass reads texture A as SRV, draws to RT B | ✅ | Both resources in fixed state for the entire pass |
| Pass dispatches compute to generate texture A, then reads A as SRV | ❌ | A transitions mid-pass. Split into 2 passes. |
| Pass reads buffer A (vertex) and buffer B (index) | ✅ | Both in fixed state |
| Pass does N independent dispatches on UAV A (GENERAL) | ✅ | All dispatches share the same resource state, no ordering between them |
| Pass dispatch 1 writes UAV A, dispatch 2 reads the result | ❌ | Data dependency between dispatches requires a barrier → split into 2 passes |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Long-lived FrameGraph                            │
│                                                                             │
│  0. SETUP (Global Scope) ──►  1. INIT (One-time) ──►  2. FRAME (Per-frame)   │
│                                                                             │
│  Publish Services            Auto-config passes      RECORD ──► EXECUTE     │
│  (AssetLoader, etc.)         (Shaders, PSOs)         Build List Flatten/Run │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Record

The user builds an arborescent structure (Node Tree) by declaring passes and groups. CPU and GPU dependencies are handled via a **Scoped Symbol Table**.

### Scoped Symbol Table (The "Internal Compiler")
Inspired by compiler theory (SSA/Phi-nodes), the graph treats all dependencies as symbols.

- **Symbols**: `ImageHandle`, `BufferHandle`, `Camera`, `RenderSettings`.
- **Global Scope**: Persistent root scope for engine services (AssetLoader, Physics).
- **Frame Scope**: Root scope for per-frame resources (Backbuffer, GBuffer).
- **Publish**: Register a symbol in the current node's scope.
- **Consume**: Resolve a symbol by looking up the tree (all the way to Global Scope).
- **Acquire**: Special publisher for external swapchain resources.

```rust
pub trait RenderPass {
    fn name(&self) -> &str;
    
    /// Called once during graph initialization. 
    /// Can consume services from the Global Scope.
    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &GlobalScope);

    /// Define per-frame dependencies and state.
    fn record(&mut self, builder: &mut PassBuilder);

    /// Execute the node commands. Only valid for Leaf nodes.
    fn execute(&self, ctx: &mut dyn PassContext);
}
```

**PassBuilder API**:

```rust
impl PassBuilder {
    // --- Scoped Symbol Table ---
    /// Register a typed symbol in the current scope.
    fn publish<T: Send + Sync + 'static>(&mut self, name: &str, data: T);
    
    /// Resolve a typed symbol from the current or parent scope.
    fn consume<T: Send + Sync + 'static>(&mut self, name: &str) -> &T;

    // --- GPU Intents (GPU Leaf only) ---
    fn read_image(&mut self, res: ImageHandle, usage: ResourceUsage);
    fn write_image(&mut self, res: ImageHandle, usage: ResourceUsage);
    fn read_buffer(&mut self, res: BufferHandle, usage: ResourceUsage);
    fn write_buffer(&mut self, res: BufferHandle, usage: ResourceUsage);

    // --- Resource Management ---
    fn declare_image(&mut self, name: &str, desc: ImageDesc) -> ImageHandle;
    fn resolve_image(&mut self, name: &str) -> ImageHandle;
    fn acquire_backbuffer(&mut self, window: WindowHandle) -> ImageHandle;

    // --- Recursion (Node Tree) ---
    fn add_pass(&mut self, pass: &mut dyn RenderPass);
    fn add_owned_pass<P: RenderPass + 'static>(&mut self, pass: P);
}
```
```

**PassContext** (adapts based on domain):

```rust
pub enum PassContext<'a> {
    Gpu {
        cmd: &'a mut CommandRecorder,
        resources: &'a ResourceRegistry,
    },
    Cpu {
        resources: &'a ResourceRegistry,
    },
}
```

> [!IMPORTANT]
> **Open question:** Should `read`/`write` return a scoped handle that the pass uses in `execute()`, or is `ResourceId` sufficient everywhere? Scoped handles add safety but add complexity.

---

## Phase 2: Compile

Sequential, pure data computation. No GPU API calls. **Must be fast** (target: <100μs for ~100 passes).

### Steps:
1. **Tree Flattening** — recursively traverse the Node Tree to produce a linear execution candidate.
2. **Symbol Resolution** — resolve all `consume()` calls to their respective `publish()` or `acquire()` origins across scopes.
3. **Dependency Graph** — build DAG from symbol read/write declarations.
4. **Topological Sort** — determine final execution order.
5. **Dead Node Elimination** — symbols that are never consumed (and don't have side effects like `present`) are culled.
6. **Barrier Resolution** — for each GPU symbol, compute `(usage_before, usage_after)` at transition points.
7. **Memory Aliasing** — compute lifetime intervals based on symbol scope. Assign overlapping memory blocks within `MemoryPools`.
8. **Queue Assignment** — assign nodes to queues; insert cross-queue sync points (timeline semaphores).

### Output: `CompiledGraph`

```rust
pub struct CompiledGraph {
    /// Ordered list of execution steps.
    steps: Vec<ExecutionStep>,
    /// Memory aliasing plan for transient resources.
    aliasing_plan: AliasingPlan,
    /// Cross-queue synchronization points.
    sync_points: Vec<SyncPoint>,
}

pub enum ExecutionStep {
    /// Insert barriers (system-generated, no user code).
    Barriers(BarrierBatch),
    /// Execute a single pass.
    ExecutePass { pass_index: usize, queue: QueueType },
    /// Execute multiple independent passes in parallel.
    ExecuteParallel { pass_indices: Vec<usize>, queue: QueueType },
    /// Cross-queue sync point.
    Signal(SyncPoint),
    Wait(SyncPoint),
}
```

> [!IMPORTANT]
> **Open question:** Barrier batching strategy. Option A: one barrier batch per pass transition. Option B: merge adjacent barriers into larger batches. B is more GPU-efficient but more complex to implement.

---

## Phase 3: Execute

Walk the `CompiledGraph`. Each pass records into its own **primary command buffer**.

```
For each ExecutionStep:
  Barriers(batch)       → System emits vkCmdPipelineBarrier in current CB
  ExecutePass(i)        → pass[i].execute(ctx) records into a dedicated CB
  ExecuteParallel(list) → rayon: each pass records into its own CB
  Signal/Wait           → Timeline semaphore operations at submit
```

### Threading Model (Fork-Join / Work-Stealing)

- **Implementation**: `rayon` — fork-join with work-stealing.
- **Thread count**: automatic. N cores = N worker threads. No hardcoded limits.
- **Scaling**: linear with core count. 8 cores → 8 threads, 64 cores → 64 threads.
- **Nested parallelism**: supported. A pass group can fork internally via `rayon::scope`.
- **Workload balancing**: work-stealing handles imbalanced passes (tiny blit vs heavy GBuffer) naturally.

**Execution flow:**
1. Compile phase produces **parallelism groups** (sets of independent passes).
2. Execute phase maps each group to a `rayon::scope` → passes in the group execute in parallel.
3. Sequential dependencies between groups are barriers / sync points.
4. **CPU passes**: same pool, same work-stealing. No separate thread pool.

**Command pool allocation**: one `VkCommandPool` per thread per frame (thread-local). Each standalone pass grabs a CB from its thread's pool. No contention.

### Command Buffer Strategy

- **Standalone passes**: one primary CB per pass (eligible for inter-pass parallel recording).
- **Inline passes**: consecutive inline passes on the same queue are merged into a single primary CB.
- At submit: all CBs are submitted in topological order via `vkQueueSubmit` (batched).

**Inline pass merging**: the compiler fuses consecutive inline passes (same queue, sequential in DAG) into a single CB. Barriers between them are emitted as `vkCmdPipelineBarrier` within the CB. A standalone pass breaks an inline chain.

### Auto Begin/End Rendering

For `Graphics` domain passes, the system **automatically** handles rendering scopes based on declared intents. 

| Intents | Auto `vkCmdBeginRendering`? | Use Case |
|---|---|---|
| `ColorAttachment` / `DepthAttachment` | ✅ Yes | Rasterization, Mesh Shaders |
| `StorageReadWrite` (UAV) only | ❌ No | Ray Tracing, Compute-in-Graphics |
| None (Read-only) | ❌ No | Blits (sometimes), Debug |

**The pass author never calls begin/end rendering.** For Raster/Mesh passes, they just record draw calls. For RT, they just record `vkCmdTraceRaysKHR`.

### Intra-Pass Parallel Recording (Secondary Command Buffers)

For heavy passes (e.g., 12k objects in GBuffer), the pass can request **parallel recording via secondary CBs**. The primary CB handles begin/end rendering; secondaries record draw calls in parallel.

```rust
fn execute(&self, ctx: &mut PassContext) {
    // System has already called vkCmdBeginRendering on the primary CB.
    // Request parallel recording via secondaries:
    ctx.parallel_record(&self.objects, 1000, |sub_ctx, chunk| {
        for obj in chunk {
            sub_ctx.draw(obj);
        }
    });
    // → N secondary CBs recorded in parallel (rayon, thread-local pools)
    // → Primary CB calls vkCmdExecuteCommands(secondaries)
    // → System calls vkCmdEndRendering
}
```

**Key points:**
- Secondaries inherit rendering state via `VkCommandBufferInheritanceRenderingInfo` (Vulkan 1.3 — no VkRenderPass compatibility needed).
- Thread-local command pools: one `VkCommandPool` per thread per frame, zero contention.
- Chunking granularity is controlled by the pass author (e.g., 1000 objects/secondary).
- If the pass doesn't call `parallel_record()`, it records directly into the primary CB (no secondary overhead).

The pass **never** creates/destroys resources or inserts barriers. It only records draw/dispatch/copy commands through the `PassContext`.

---

## Pipeline State & Shaders

We adopt **Vulkan terminology** (`GraphicsPipeline`, `ComputePipeline`, `RayTracingPipeline`) to avoid leveling down to lower-denominator APIs.

- **Shader Language**: **Slang** is the primary target, providing high-level features while emitting performant SPIR-V/DXIL.
- **PSO Ownership**: The `Node` provides a `PipelineDescription`. The **backend** is responsible for caching and deduplication.
- **Creation**: Pipeline compilation happens during or before graph execution, potentially as a `Cpu` node.

---

## Symbol Model

Both GPU resources and CPU data are unified as **Symbols**. A symbol represents a typed value within the graph's scope tree.

```rust
pub enum SymbolType {
    Image(ImageDesc),
    Buffer(BufferDesc),
    CpuData(TypeId), // References std::any::TypeId
}

pub enum SymbolLifetime {
    /// Exists only within its declaring scope. 
    /// Candidate for memory aliasing (GPU) or scope-exit drop (CPU).
    Transient,
    /// Persists across frames. Owned by the graph.
    Persistent,
    /// Injected from outside.
    External,
}
```

### Temporal Symbols (History)

To support temporal algorithms (TAA, GI feedback), symbols can declare a **history depth**. 

- **Versioning**: The engine maintains a ring buffer of `depth + 1` versions.
- **Reference**: Nodes access versions relative to the current frame.

```rust
fn record(&mut self, builder: &mut PassBuilder) {
    // Read previous frame's result symbol
    let prev_color = builder.read_history("ColorBuffer", -1, ShaderReadOnly);
    // Write current frame's result symbol
    builder.write("ColorBuffer", ColorAttachment);
}
```

**Initialization & First Frame:**
- On the very first frame (or after a reset), history versions are effectively "empty" or "black".
- The engine can provide a `is_first_frame()` hint so the pass can skip history sampling or use a different path.

**Automatic Versioning (Double/Triple Buffering):**
Even if `history_depth` is 0, resources are internally versioned by the engine to support **Frame Overlap** (N frames in flight). This is transparent: the pass always gets the "correct" version for its frame index. Explicit `history_depth` is only for when the *content* of the resource must be preserved across frames.

### Resolution Change

The graph propagates the current render size. Nodes query it during `record()` and compute dimensions.

```rust
fn record(&mut self, builder: &mut PassBuilder) {
    let (w, h) = builder.render_size();
    builder.publish("InternalRT", ImageDesc::new(w, h, ...));
}
```

**On resize**: 
1. The system detects a resolution change.
2. The graph is rebuilt: all nodes have their `record()` method called.
3. Persistent symbols detect descriptor changes and reallocate.

`declare()` is the **unique source of truth** for all graph-managed resources. Passes do not need a separate resize hook.

### External Symbols (Imported)

Static assets (textures, meshes) are managed outside the Frame Graph but are "bound" as symbols.

- **Ownership**: The Frame Graph **never** owns or destroys external symbols.
- **Importing**: Done via `graph.bind_external<T>(name, handle)`.
- **Swapchain Integration**: The backbuffer is a special external symbol introduced via `acquire_backbuffer()`.

**The Flow:**
1. **Acquire**: `builder.acquire_backbuffer(window)` creates a symbol.
2. **Use**: Nodes render into the symbol.
3. **Present**: `ctx.present(symbol)` triggers the queue call.

```rust
// 1. External Acquire
let (swap_handle, ready_sem) = hri.acquire_next_image();

// 2. Import Backbuffer (HRI handles ready_sem internally)
let backbuffer = graph.import_backbuffer(swap_handle, "backbuffer");

// 3. Render directly into it
graph.add_pass(UI_Pass {
    // declare(): write(backbuffer, ColorAttachment)
});

// 4. Final transition to Present
graph.add_pass(Present_Pass {
    // declare(): write(backbuffer, Present)
});
```

### CPU Data Symbols (Typed Blackboard)

The Scoped Symbol Table replaces the traditional untyped blackboard.

```rust
// Game loop binds external data symbols
graph.bind_external("MainCamera", camera_controller.get());

// Pass A consumes camera, publishes internal data
builder.add_pass("Culling", |sub| {
    let cam = sub.consume::<Camera>("MainCamera");
    let visible_list = perform_culling(cam);
    sub.publish("VisibleObjects", visible_list);
    |_| {}
});

// Pass B (Graphics) consumes the culled list
builder.add_pass("GBuffer", |sub| {
    let list = sub.consume::<CullingResult>("VisibleObjects");
    move |ctx| { ctx.draw_list(list); }
});
```

**What the compiler sees:** `Camera (ext) → PrepareConstants (Cpu) → GBuffer (Graphics)`

**What this enables:**
- CPU passes that don't touch the same data run in **parallel** (thread pool)
- CPU → GPU ordering is automatic (the CPU pass finishes before the GPU pass records)
- Ownership follows Rust semantics: the creating pass owns the data, readers borrow

---

## Node Hierarchy (Composition)

Composition is achieved through nesting. A Node can contain children, effectively creating a sub-graph.

**Rules:**
- **Symbol Scoping**: Symbols published by a node are visible to its descendants.
- **Resource Aliasing**: The compiler tracks the "In-Use" range of a symbol based on its scope. If a branch node ends, all its transient GPU symbols are released for aliasing.
- **Flattening**: During compilation, the tree is linearized into a list of hardware execution steps, while preserving the semantic barriers between scopes.

---

## Multi-Queue Model

```
┌─────────────────────────────────────────────────┐
│  Graphics Queue    ┃  Compute Queue  ┃ Transfer │
│  ━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━╋━━━━━━━━━ │
│  GBuffer pass      ┃  SSAO compute   ┃          │
│  Lighting pass     ┃  Particle sim   ┃          │
│  PostFX pass       ┃                 ┃          │
│       ▲            ┃      │          ┃          │
│       └── wait ────╋──────┘          ┃          │
│   (timeline sem)   ┃                 ┃          │
└─────────────────────────────────────────────────┘
```

- Pass declares `QueueAffinity` as a **hint**, not a hard constraint.
- Compiler assigns queues based on actual GPU capabilities (`vkGetPhysicalDeviceQueueFamilyProperties`).
- **Fallback**: if no async compute queue, compute-affinity passes run on graphics queue. Zero code change in the pass.
- Cross-queue sync via **timeline semaphores** (Vulkan 1.2).
- **Queue sharing** (hybrid strategy):
  - **Buffers** cross-queue → `VK_SHARING_MODE_CONCURRENT` (no hardware compression to lose, zero overhead in practice).
  - **Images** cross-queue → `VK_SHARING_MODE_EXCLUSIVE` (preserves DCC; ownership transfers handled by the graph compiler via symbol tracking).
  - **Images** single-queue → `VK_SHARING_MODE_EXCLUSIVE` (default, no question).

---

## Memory Aliasing

Transient resources with non-overlapping lifetimes within a frame share the same logical offsets in a `MemoryPool`.

```
Pass A creates T1 (64MB)    ████░░░░░░░░░░░░
Pass B creates T2 (64MB)    ░░░░░░████░░░░░░
                             ↑ T1 and T2 share the same 64MB block
```

### Interaction with Asynchronous Submission

To support **Zero Stall** parallel execution (CPU recording Frame N+1 while GPU executes Frame N):

1.  **Multi-Framing**: The system maintains a ring buffer of `MemoryPool` objects (typically 2 or 3, matching the frame-in-flight count).
2.  **Safety**: A `MemoryPool` used in Frame N is **locked** by its `PendingSubmission`. 
3.  **Reuse**: The pool is only "reset" and made available for aliasing in a new frame once `collect_garbage()` detects that the GPU has finished using it.

This means aliasing is a **two-tier optimization**:
- **Tier 1 (Intra-frame)**: Bin-packing resources within a single pool based on the DAG.
- **Tier 2 (Inter-frame)**: Rotating/Ring-buffering pools to allow overlap without data corruption.

---

## Error Handling

Reliability is paramount. Conflict detection happens during the **Compile** phase.

- **Conflicting Declarations**: If two passes write to the same resource without an ordering dependency (DAG cycle or independent branches), a `ResourceConflictError` is raised.
- **Invalid Transitions**: Attempting to transition a resource to an incompatible state (e.g., Depth → Color) triggers an error.
- **Undeclared Access**: Debug builds of the `RenderContext` check that every resource used in `execute()` was properly declared.
- **Type Safety**: `compile()` returns `Result<CompiledGraph, GraphError>`, allowing the engine to gracefully handle or report failures without crashing.

---

## Debugging & Profiling

The Frame Graph provides observability by default.

- **GPU Timestamps**: The system can inject `vkCmdWriteTimestamp` queries before/after every pass. This provides per-pass GPU timing without manual instrumentation.
- **RenderDoc Integration**: Pass names are propagated to `vkCmdBeginDebugUtilsLabel`. Resources are named in Vulkan based on their Graph name.
- **Visualizer**: The `CompiledGraph` can be exported as a `.dot` file for visualization in Graphviz.
- **Validation Layers**: The graph's explicit synchronization logic should eliminate validation errors. If they occur, they are likely bugs in the graph compiler itself.

---

---

## Runtime / Backend Decoupling (HRI boundary)

To ensure the Frame Graph remains **API agnostic**, we enforce a strict separation between the logical graph (Runtime) and the hardware-specific implementation (HRI - Hardware Rendering Interface).

```mermaid
graph LR
    User["Engine/Pass Author"] -- "record tree" --> FG["Frame Graph (Agnostic)"]
    FG -- "compile & flatten" --> CG["Compiled Graph (Linear Steps)"]
    CG -- "execute (with Symbols)" --> RHI["RenderBackend (Vulkan/DX12)"]
    RHI -- "native calls" --> GPU["GPU"]
```

### The Boundary: Hardware Rendering Interface (HRI)

The Frame Graph doesn't know what a `VkImage` or `ID3D12Resource` is. It operates on **Logical Handles**.

1.  **Logical Commands**: The graph produces a stream of commands (`Draw`, `Dispatch`) using **Symbol IDs**.
2.  **The Registry**: Maps Symbol IDs to hardware resources (`VkImage`, `VkBuffer`).
3.  **Barrier Translation**: The Runtime says "Transition Symbol 42 to ColorAttachment". The backend translates this into a `VkImageMemoryBarrier2`.

---

## API Specification

### 1. The Pass Trait
This is the only thing the user implements.

```rust
pub trait RenderPass {
    fn name(&self) -> &str;
    fn domain(&self) -> PassDomain; // Graphics, Compute, Transfer

    /// Declare intents. No hardware access allowed.
    /// This is called whenever the graph needs rebuilding (including resizes).
    fn declare(&mut self, builder: &mut PassBuilder);

    /// Record commands. Logical access via ctx.
    fn execute(&self, ctx: &mut PassContext);
}
```

### 2. PassBuilder & FrameGraph API
Unified Symbol interaction.

```rust
pub trait PassBuilder {
    // Symbol Management
    fn publish<T>(&mut self, name: &str, data: T);
    fn consume<T>(&mut self, name: &str) -> &T;
    
    // GPU Generation
    fn declare_image(&mut self, name: &str, desc: ImageDesc) -> ImageHandle;
    fn acquire_backbuffer(&mut self, window: WindowHandle) -> ImageHandle;

    // Intents (Leaf only)
    fn read(&mut self, handle: ImageHandle, usage: ResourceUsage);
    fn write(&mut self, handle: ImageHandle, usage: ResourceUsage);

    // Tree Construction
    fn add_node(&mut self, name: &str, setup: impl FnOnce(&mut PassBuilder));
}

impl FrameGraph {
    /// Entry point for external symbol binding.
    fn bind_external<T>(&mut self, name: &str, data: T);
    
    /// Root node recording.
    fn record(&mut self, setup: impl FnOnce(&mut PassBuilder));
}
```

### 3. PassContext (Execution Phase)
The bridge to the HRI.

```rust
pub trait PassContext {
    fn set_pipeline(&mut self, pipeline: PipelineHandle);
    fn draw(&mut self, count: u32, first: u32);
    fn dispatch(&mut self, x: u32, y: u32, z: u32);
    
    /// Queue commands
    fn present(&mut self, image: ImageHandle);
}
```

### 4. The HRI Backend Trait
What i3fx must implement for each API.

```rust
pub trait HriBackend {
    fn create_texture(&mut self, desc: &ResourceDesc) -> HriTexture;
    fn create_buffer(&mut self, desc: &ResourceDesc) -> HriBuffer;
    
    /// The "Magic" function that turns logical transitions into API barriers.
    fn apply_barriers(&mut self, batch: &BarrierBatch, cmdbuf: &mut NativeCmdBuf);
    
    fn begin_rendering(&mut self, attachments: &[RenderingAttachment], cmdbuf: &mut NativeCmdBuf);
    fn end_rendering(&mut self, cmdbuf: &mut NativeCmdBuf);

    /// Submit command buffers and return a handle to track GPU completion.
    fn submit(&mut self, cmdbufs: &[NativeCmdBuf]) -> PendingSubmission;

    /// Process finished submissions and release associated resources.
    fn collect_garbage(&mut self);
}
```

---

## Asynchronous Submission & Lifetime

Submission is the boundary where logical passes become asynchronous GPU work.

1.  **Multi-Framing**: The system maintains a ring buffer of `MemoryPool` objects.
2.  **Safety**: A `MemoryPool` used in Frame N is **locked** until GPU completion. 
3.  **Reuse**: The pool is reset once `collect_garbage()` detects completion.
4.  **Symbol Tracking**: Persistent symbols (history) are managed via versioning in the root symbol table.

---


## Verification Plan

This is a design document — no code to test yet. Verification will consist of:

### 1. The Design Review
- User reviews and approves architecture before any implementation.
- Iterate on open questions until all are resolved.

### 2. NullBackend Strategy (The "Oracle")
A specialized `HriBackend` that performs no hardware work but logs every action.
- **Validation**: Ensures that barriers are logically correct (no read-after-write without sync).
- **Visualization**: Generates Mermaid or DOT diagrams of the compiled graph and memory aliasing plan.
- **Unit Testing**: Allows testing the Frame Graph Runtime in CI without a GPU/Vulkan driver.

### 3. Integration Testing
- **Simple Triangle**: Minimal Graphics pass to verify HRI command recording.
- **Async Compute**: Verify cross-queue timeline semaphore signals and waits.
- **Temporal Stress Test**: Verify history depth ring-buffer rotation and resize behavior.

---

## Areas for Refinement

1. **Hot reload**: Can passes be swapped at runtime? Supported by per-frame rebuild.
2. **Cyclic dependencies** (future): Temporal reprojection nodes.

---
