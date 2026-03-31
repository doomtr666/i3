# Frame Graph Optimizer & Sync Oracle — Technical Specification

## Overview

The **Frame Graph Optimizer** covers two orthogonal concerns:

```
User passes (declarations)
        │
        ▼
┌─────────────────────┐
│  Compiler           │  builds DAG, assigns queues, determines execution order
│  scheduler.rs       │  ← Phase 5: HEFT scheduling (currently: BFS level sort)
└─────────────────────┘
        │ flat_passes (ordered)
        ▼
┌─────────────────────┐
│  Sync Oracle        │  takes the order as given, computes all barriers
│  oracle.rs          │  ← Phase 1 (this document)
└─────────────────────┘
        │ SyncPlan
        ▼
┌─────────────────────┐
│  Backend            │  records Vulkan command buffers
│  commands.rs        │
└─────────────────────┘
```

These two concerns are **orthogonal**: the Oracle does not decide order, the Scheduler does not compute barriers. This separation is critical — the Oracle can be built and validated independently of any scheduling improvements.

Primary duties:
1. **Sync Oracle**: Generating a static `SyncPlan` for pre-calculated barriers (Phase 1).
2. **Aliasing Analyzer**: Resource lifetime analysis for memory reuse (Phase 4).
3. **Scheduler**: Finding the execution order that minimizes frame time (Phase 5).

---

## 1. Integration Point in the Compiler

The Oracle inserts itself in `compiler.rs::execute()` between existing steps:

```
// Step 3: Swapchain acquire + external resource registration  ← existing
// Step NEW: Oracle analysis → produces SyncPlan              ← new
// Step 4.5: prepare_pass loop (now reads from SyncPlan)       ← modified
```

A new method is added to `RenderBackendInternal`:

```rust
fn analyze_frame(&mut self, passes: &[FlatPass]) -> SyncPlan;
```

Called once per frame with the full ordered pass list. `prepare_pass` signature stays the same but its implementation becomes a lookup into the pre-computed `SyncPlan` instead of computing barriers on the fly.

The `SyncPlan` is stored on `VulkanBackend` for the duration of the frame:

```rust
pub(crate) sync_plan: Option<SyncPlan>, // set by analyze_frame, consumed by prepare_pass
```

---

## 2. Core Data Structures

### 2.1 `ResourceState` — Canonical GPU State

Compact, copyable. Lives on `PhysicalImage` / `PhysicalBuffer`. Represents the last **committed** state on the GPU.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceState {
    pub layout: vk::ImageLayout,   // UNDEFINED for buffers/AS
    pub access: vk::AccessFlags2,
    pub stage: vk::PipelineStageFlags2,
    pub queue_family: u32,
}
```

This is the only persistent sync state. Updated once at frame end via `SyncPlan::final_states`.

### 2.2 `ResourceFlowState` — Transient Analysis State

Used exclusively inside the Oracle during `analyze_frame`. Never stored between frames.

```rust
struct ResourceFlowState {
    current: ResourceState,           // Evolves as passes are simulated
    first_use_this_frame: bool,       // True until the resource is first accessed
    is_transient: bool,               // Drives UNDEFINED promotion (see §3.2)
    last_writer_pass: Option<usize>,  // Pass index, for Release injection (see §3.3)
    // Phase 4 hooks (tracked but unused until aliasing):
    first_use_pass: Option<usize>,
    last_use_pass: Option<usize>,
}
```

### 2.3 `PassSyncData` — Per-Pass Barrier Slots

Three distinct barrier slots per pass. The separation is mandatory for correct command buffer recording.

```rust
struct PassSyncData {
    /// Emitted BEFORE pass commands (layout transitions, WAR/WAW sync, Acquire barriers)
    pre_barriers: Vec<Barrier>,
    /// Emitted AFTER pass commands (Release barriers to transfer ownership to next queue)
    post_barriers: Vec<Barrier>,
    /// Load op decision for each render target written by this pass (images only)
    load_ops: HashMap<ResourceId, vk::AttachmentLoadOp>,
}
```

**Why `post_barriers` is separate:**
A Release barrier must be recorded *after* the producing commands on its queue, and *before* the consuming queue's Acquire. This applies equally to graphics and compute passes:

- **Graphics pass**: emit `post_barriers` after `cmd_end_rendering()`, before `cmd_end_command_buffer()`.
- **Compute pass**: emit `post_barriers` after the last `cmd_dispatch*()`, before `cmd_end_command_buffer()`.
- **Transfer pass**: emit `post_barriers` after the last `cmd_copy_*()`, before `cmd_end_command_buffer()`.

The recording logic in `record_pass` uses the same `post_barriers` field regardless of domain — the domain only determines *where* rendering/dispatch ends.

### 2.4 `SyncPlan` — Oracle Output

```rust
pub struct SyncPlan {
    /// Per-pass barrier data, indexed by position in flat_passes
    pub passes: Vec<PassSyncData>,
    /// Final ResourceState for each resource — committed to PhysicalImage/Buffer at frame end
    pub final_states: HashMap<ResourceId, ResourceState>,
}
```

Indexed by position in `flat_passes` (same index used by the compiler's `prepared_passes` vec).

### 2.5 `Barrier` — Unified Barrier Type

Acceleration Structures do not have their own barrier type in Vulkan — AS reads/writes are expressed via `VkBufferMemoryBarrier2` on the underlying buffer, with `ACCESS_ACCELERATION_STRUCTURE_READ/WRITE_KHR` flags.

```rust
pub enum Barrier {
    Image(vk::ImageMemoryBarrier2<'static>),
    Buffer(vk::BufferMemoryBarrier2<'static>), // also used for AS
}
```

---

## 3. Resource Coverage

The Oracle handles all three resource categories. The analysis logic is identical; only the state derivation differs.

| Resource type     | Vulkan barrier type       | Layout field        | Notes |
|-------------------|---------------------------|---------------------|-------|
| Image             | `VkImageMemoryBarrier2`   | Full layout tracking | Requires aspect mask, subresource range |
| Buffer            | `VkBufferMemoryBarrier2`  | Always `UNDEFINED`  | Covers `SHADER_READ/WRITE`, `TRANSFER_*`, `INDIRECT_READ` |
| Acceleration Structure | `VkBufferMemoryBarrier2` on underlying buffer | Always `UNDEFINED` | Uses `ACCESS_ACCELERATION_STRUCTURE_READ/WRITE_KHR`, stage `ACCELERATION_STRUCTURE_BUILD_KHR` or `RAY_TRACING_SHADER_KHR` |

**AS sync state**: `PhysicalAccelerationStructure` currently has no `sync` field. Add one:

```rust
pub struct PhysicalAccelerationStructure {
    pub handle: vk::AccelerationStructureKHR,
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub address: u64,
    pub sync: ResourceState,   // ← add
    pub build_info: Option<BlasCreateInfo>,
}
```

Initialized to: `access = NONE`, `stage = TOP_OF_PIPE`, `queue_family = graphics_family`.

After a `build_blas` / `build_tlas`, state becomes:
```
access = ACCELERATION_STRUCTURE_WRITE_KHR
stage  = ACCELERATION_STRUCTURE_BUILD_KHR
```

`get_as_state(usage)` maps `ACCEL_STRUCT_READ` → `(ACCELERATION_STRUCTURE_READ_KHR, RAY_TRACING_SHADER_KHR)` and `ACCEL_STRUCT_WRITE` → `(ACCELERATION_STRUCTURE_WRITE_KHR, ACCELERATION_STRUCTURE_BUILD_KHR)`.

---

## 4. Algorithmic Complexity

Let:
- **P** = number of passes in the frame
- **R** = total number of distinct resources (images + buffers + AS)
- **D** = average resource *degree* = average number of resources accessed per pass

### Oracle Simulation Loop

```
Phase 1 Init:       O(R)         — one pass over all physical resources
Phase 2 Simulation: O(P × D)     — for each pass, iterate its declared resources
Phase 3 Final:      O(R)         — write final_states from ResourceFlowMap
```

**Total: O(R + P × D)**

In practice D is small (a typical pass accesses 2–10 resources). For a 200-pass frame with 500 resources: ~2000 iterations. This is negligible compared to the GPU workload.

### Memory Allocations

The `ResourceFlowMap` is a `HashMap<ResourceId, ResourceFlowState>` pre-allocated with capacity `R`. No allocations inside the hot loop except `Vec::push` for barriers (amortized O(1)).

`SyncPlan::passes` is pre-allocated with capacity `P` before the loop:

```rust
let mut plan = SyncPlan {
    passes: Vec::with_capacity(flat_passes.len()),
    final_states: HashMap::with_capacity(resource_count),
};
```

### What to Avoid

- **No nested loops over resources per pass** — state lookup is O(1) via HashMap.
- **No sorting of barriers** — insertion order from the simulation is the correct execution order.
- **No clone of barriers** — barriers in `SyncPlan` are moved into `VulkanPreparedPass` by index reference, not copied.

---

## 5. The Analysis Algorithm

Single-threaded. Called once after swapchain acquire, before any `prepare_pass`.

### Phase 1: Initialization

Build `ResourceFlowMap` from backend canonical state:

```
for each (id, img) in backend.images.iter():
    flow_map[id] = ResourceFlowState {
        current: img.sync,
        first_use_this_frame: true,
        is_transient: img.is_transient,
        last_writer_pass: None,
        first_use_pass: None, last_use_pass: None,
    }
// same for buffers and accel_structs
```

### Phase 2: Simulation Loop

For each `(pass_index, pass)` in `flat_passes`:

#### Step A — Determine required state per resource

For images: `get_image_state(usage, is_write, bind_point)` (existing, unchanged).
For buffers: `get_buffer_state(usage, bind_point)` (existing, unchanged).
For AS: `get_as_state(usage)` (new, trivial).

`bind_point` is derived from `pass.queue`: `AsyncCompute | Transfer → COMPUTE`, else `GRAPHICS`.

#### Step B — First-Use and UNDEFINED Promotion

```
if state.first_use_this_frame:
    state.first_use_this_frame = false
    state.first_use_pass = Some(pass_index)   // Phase 4 hook
    if state.is_transient:
        current.layout = UNDEFINED   // images only; skip for buffers/AS
```

**`is_transient` semantics:**
- `true` → created via `create_transient_image` / pooled. Content discardable each frame.
- `false` → textures uploaded via staging, shadow maps, history buffers, persistent depth. Layout preserved across frames.

**Load op decision** (for `COLOR_ATTACHMENT` or `DEPTH_STENCIL` writes only):
```
if current.layout == UNDEFINED → CLEAR
else if no prior write this frame in SyncPlan → LOAD   (persistent resource)
else → LOAD   (already written this frame)
```

Simplification: for now, "prior write this frame" = `state.last_writer_pass.is_some()`.

#### Step C — Barrier Generation (same-queue)

```
needs_barrier =
    current.layout != required.layout           (images only)
    || last_access_was_write(current.access)    (WAW or RAW hazard)
    || current.access doesn't cover required.access
    || current.stage doesn't cover required.stage
```

If `needs_barrier`:
```
barrier = ImageMemoryBarrier2 / BufferMemoryBarrier2 {
    src_stage:  sanitize(current.stage, queue_type_of(current.queue_family))
    src_access: sanitize(current.access, src_stage, src_queue_type)
    dst_stage:  sanitize(required.stage, pass_queue_type)
    dst_access: sanitize(required.access, dst_stage, dst_queue_type)
    old_layout / new_layout (images only)
    src_queue_family_index: QUEUE_FAMILY_IGNORED
    dst_queue_family_index: QUEUE_FAMILY_IGNORED
}
pass_sync.pre_barriers.push(barrier)
```

Update `current` to `required`.

#### Step D — Queue Family Migration (EXCLUSIVE resources only)

```
if !resource.concurrent && current.queue_family != pass_queue_family:
    old_layout = layout BEFORE Step C updated it
    new_layout = required.layout (after Step C)

    release = barrier {
        src_qf: current.queue_family, dst_qf: pass_queue_family,
        old_layout, new_layout,
        src_stage: sanitize(pre_C_stage, src_queue_type),
        src_access: sanitize(...),
        dst_stage: EMPTY, dst_access: EMPTY,
    }

    acquire = barrier {
        src_qf: current.queue_family, dst_qf: pass_queue_family,
        old_layout, new_layout,
        src_stage: EMPTY, src_access: EMPTY,
        dst_stage: sanitize(required.stage, dst_queue_type),
        dst_access: sanitize(...),
    }

    if let Some(writer) = state.last_writer_pass:
        plan.passes[writer].post_barriers.push(Release)
    // else: first-frame use on non-graphics queue — emit Release-less Acquire
    // (valid per spec if resource was never owned by another queue this frame)

    plan.passes[pass_index].pre_barriers.insert(0, Acquire)
    // Acquire must precede regular transition barriers in pre_barriers

    current.queue_family = pass_queue_family
```

**Concurrent resources**: skip Step D. No Release/Acquire needed.

#### Step E — Update tracking

```
state.last_use_pass = Some(pass_index)   // Phase 4 hook
if is_write:
    state.last_writer_pass = Some(pass_index)
```

### Phase 3: Final States

```
for (id, state) in flow_map:
    plan.final_states[id] = state.current
```

---

## 6. Graph Outputs and Headless Graphs

### 6.1 The OUTPUT Abstraction

A frame graph produces **outputs** — resources that are the final products intended for consumption outside the graph. The DCE algorithm seeds liveness from these outputs. `Present` is simply one kind of output.

Rather than inferring roots from `SymbolLifetime` (fragile, implicit), the graph exposes an explicit `mark_output` API:

```rust
pub enum OutputKind {
    Present(WindowHandle),       // transition to PRESENT_SRC_KHR + vkQueuePresentKHR
    Readback,                    // transition to TRANSFER_SRC, CPU will map/copy after execute()
    Texture,                     // baked map (RMAO, Normal, Lightmap...) — stays on GPU
    AccumulationBuffer,          // temporal / iterative baking target
}

// On the graph builder:
graph.mark_output(image_handle, OutputKind::Present(window));
graph.mark_output(rmao_map,     OutputKind::Texture);
graph.mark_output(normal_map,   OutputKind::Texture);
graph.mark_output(ao_buffer,    OutputKind::Readback);
```

**What this replaces:**
- The implicit `ResourceUsage::PRESENT` detection in `infer_domain()`
- The `swapchain_requests` special-case in `process_externals_recursive`
- The DCE root inference from `SymbolLifetime::External` / `TemporalHistory`

Everything becomes: *"trace liveness backward from declared outputs."*

### 6.2 OutputKind Effects

Each kind drives specific backend behavior after the last writing pass:

| OutputKind | Final barrier | Backend action |
|------------|--------------|----------------|
| `Present(w)` | `COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR` | `vkQueuePresentKHR` |
| `Readback` | `* → TRANSFER_SRC_OPTIMAL` | none (caller calls `map_buffer` / staging copy) |
| `Texture` | none (stays in last written layout) | none |
| `AccumulationBuffer` | none (preserved for next `execute()`) | none |

The Oracle generates the final barrier as part of Phase 3 (Final States), using the `OutputKind` to determine the target layout. `Readback` and `Texture` outputs have `is_transient = false` — layout is preserved across `execute()` calls.

### 6.3 Headless Graph — No Code Divergence

A headless bake graph simply has no `Present` output. The rest of the pipeline — compiler, Oracle, submit — is identical:

```
Presentation graph:    mark_output(backbuffer, Present(window))  → DCE root
Headless bake graph:   mark_output(rmao,  Texture)               → DCE root
                       mark_output(normals, Texture)             → DCE root
                       mark_output(raw_data, Readback)           → DCE root
```

No swapchain acquire, no binary semaphores, no `inactive_images` path. The compiler detects "no Present output" and skips the swapchain path entirely.

### 6.4 Multi-Iteration Baking

Baking runs `execute()` N times accumulating results. The Oracle handles this correctly via `is_transient`:

| OutputKind | `is_transient` | Oracle first-use | Baking semantics |
|------------|---------------|-----------------|-----------------|
| `Present` | `true` (swapchain) | UNDEFINED, CLEAR | Fresh frame each time ✓ |
| `Readback` | `false` | Layout preserved | Accumulation across iterations ✓ |
| `Texture` | `false` | Layout preserved | Persistent baked result ✓ |
| `AccumulationBuffer` | `false` | Layout preserved | Explicit cross-iteration state ✓ |
| Internal `Transient` | `true` | UNDEFINED, CLEAR | Scratch resets each iteration ✓ |

### 6.5 Readback Pattern

```rust
// Declare outputs
let rmao    = graph.create_image("RMAO",   rmao_desc,   SymbolLifetime::Persistent);
let normals = graph.create_image("Normals", normal_desc, SymbolLifetime::Persistent);
graph.mark_output(rmao,    OutputKind::Texture);
graph.mark_output(normals, OutputKind::Texture);

// Bake loop
for _ in 0..iterations {
    graph.execute(&mut backend)?;
}

// Readback (caller adds a final Readback pass or reads via staging)
backend.wait_for_timeline(timeline_value, u64::MAX);
```

The Oracle automatically generates the correct final barrier for each output based on its `OutputKind` — no manual `vkCmdPipelineBarrier` needed.

---

## 7. Commit Phase

Applied in `VulkanBackend::end_frame()`, after `submit()` returns:

```rust
if let Some(plan) = backend.sync_plan.take() {
    for (resource_id, final_state) in plan.final_states {
        if let Some(img) = backend.images.get_mut(resource_id) {
            img.sync = final_state;
        } else if let Some(buf) = backend.buffers.get_mut(resource_id) {
            buf.sync = final_state;
        } else if let Some(astr) = backend.accel_structs.get_mut(resource_id) {
            astr.sync = final_state;
        }
    }
}
```

`img.sync` / `buf.sync` / `astr.sync` are **read-only during frame execution**. This eliminates the "state updated mid-frame" race conditions and makes the sync path fully deterministic.

---

## 7. Multi-Queue Synchronization

See §5 Step D for the Release/Acquire injection algorithm.

**Vulkan spec constraints:**
- Release and Acquire barriers must use **identical** `oldLayout`, `newLayout`, `srcQueueFamilyIndex`, `dstQueueFamilyIndex`.
- Release: `dstStageMask = 0`, `dstAccessMask = 0`.
- Acquire: `srcStageMask = 0`, `srcAccessMask = 0`.
- Both must be bracketed by a timeline semaphore Signal/Wait (handled by `ExecutionStep::Signal/Wait` in the compiler — no change needed here).

---

## 8. Diagnostic & Plan Dumper — Phase 2

### 8.1 Plan Dumper

Triggered by env var `I3_DUMP_SYNC=1`. Reads directly from `SyncPlan` — no side effects.

```
SYNC PLAN — Frame 42
═══════════════════════════════════════════════════════════════════
PASS [0] GBuffer              queue=Graphics
  PRE:
    [img:SceneDepth   ] UNDEFINED        -> DEPTH_ATTACHMENT      load_op=CLEAR
    [img:GBufferAlbedo] SHADER_READ_ONLY -> COLOR_ATTACHMENT      load_op=LOAD
  POST: (none)

PASS [1] ShadowResolve        queue=AsyncCompute
  PRE:
    ACQUIRE [img:SceneDepth] Graphics(0)→Compute(2)  DEPTH_ATTACHMENT -> SHADER_READ_ONLY
    [img:SceneDepth   ] (covered by Acquire, no additional barrier)
  POST: (none)

--- Retroactive injections ---
PASS [0] GBuffer
  POST:
    RELEASE [img:SceneDepth] Graphics(0)→Compute(2)  DEPTH_ATTACHMENT -> SHADER_READ_ONLY

═══════════════════════════════════════════════════════════════════
FINAL STATES:
  [img:SceneDepth   ] SHADER_READ_ONLY  stage=COMPUTE_SHADER  qf=2
  [img:GBufferAlbedo] COLOR_ATTACHMENT  stage=COLOR_OUTPUT    qf=0
  [buf:InstanceData ] access=SHADER_READ stage=VERTEX_SHADER  qf=0
```

### 8.2 Structured Tracing

`tracing::debug!` per Oracle decision (gated behind `target: "i3_oracle"`):

```
DEBUG i3_oracle: [img:SceneDepth] first_use, transient → UNDEFINED, load_op=CLEAR
DEBUG i3_oracle: [img:SceneDepth] Graphics→Compute migration: Release@pass[0] / Acquire@pass[1]
DEBUG i3_oracle: [buf:InstanceData] WAW hazard → barrier emitted (SHADER_WRITE→SHADER_READ)
```

---

## 9. Code to Delete After Phase 1

The following code becomes dead once the Oracle is in place and verified. Delete in one cleanup commit after validation layers report zero errors.

**`crates/i3_vulkan_backend/src/sync.rs`**
- `get_image_barrier()` — stateful wrapper (replaced by Oracle)
- `get_buffer_barrier()` — stateful wrapper (replaced by Oracle)
- The `SyncContext` struct (unused placeholder, superseded by `ResourceFlowState` in `oracle.rs`)

**`crates/i3_vulkan_backend/src/backend.rs`**
- `VulkanBackend::get_image_barrier()` — wrapper calling deleted function
- `VulkanBackend::get_buffer_barrier()` — wrapper calling deleted function

**`crates/i3_vulkan_backend/src/commands.rs`**
- Entire barrier-generation section in `prepare_pass()` (lines ~790–885) — replaced by SyncPlan lookup
- `img.last_write_frame` field usage in `prepare_pass` — replaced by `load_ops` from SyncPlan
- Inline `img.sync.*` mutations in `copy_buffer`, `clear_buffer`, `mark_image_as_presented` — replaced by commit phase
- `image_barrier_scratch` / `buffer_barrier_scratch` vecs on `VulkanBackend` — no longer needed

**`crates/i3_vulkan_backend/src/resource_arena.rs`**
- `PhysicalImage::last_write_frame` field — replaced by `load_ops` in `PassSyncData`

**`crates/i3_vulkan_backend/src/resources.rs`**
- Inline `img.sync.*` mutation at end of `upload_image_data()` — replaced by commit phase

**`crates/i3_vulkan_backend/src/submission.rs`**
- Inline `img.sync.*` mutations on swapchain image acquire/recreate — replaced by Oracle init phase reading `img.sync` as canonical state (which is already correct since swapchain images are transient)

---

## 10. Implementation Roadmap

### Phase 1: SyncOracle

1. Add `is_transient: bool` to `PhysicalImage` / `PhysicalBuffer`. Add `sync: ResourceState` to `PhysicalAccelerationStructure`.
2. Define `Barrier`, `PassSyncData`, `SyncPlan` in `sync.rs`.
3. Implement `SyncOracle::analyze(flat_passes, backend) -> SyncPlan` in new `oracle.rs`.
4. Add `get_as_state(usage)` to `sync.rs`.
5. Add `analyze_frame` to `RenderBackendInternal` trait + `VulkanBackend` impl.
6. Call `analyze_frame` in `compiler.rs::execute()` between steps 3 and 4.5.
7. Refactor `prepare_pass` to read `pre_barriers`, `post_barriers`, `load_ops` from `SyncPlan` by index.
8. Emit `post_barriers` in `record_pass` after end of rendering/dispatch, before `end_command_buffer`.
9. Add commit phase in `end_frame`.
10. Validate with Vulkan Validation Layers → zero SYNC-HAZARD errors.
11. Delete dead code (see §9).

### Phase 2: Plan Dumper

1. `dump_sync_plan(plan, passes, resource_names)` in `oracle.rs`.
2. Wire to `I3_DUMP_SYNC` env var in `analyze_frame`.

### Phase 3: DCE

Pre-pass in `oracle.rs` before simulation loop. Depends on Phase 1 only.

### Phase 4: Aliasing

Requires sub-spec. `first_use_pass` / `last_use_pass` hooks in `ResourceFlowState` are already tracked from Phase 1.

---

## 11. Pass Scheduler — Phase 5

### 11.1 Problem Statement

Given a DAG of passes with fixed queue assignments (raster → Graphics, compute → Graphics or AsyncCompute), find the execution order (a valid topological sort) that **minimizes total frame time (makespan)**.

This is a DAG scheduling problem on k heterogeneous machines. It is NP-hard in general. With k≤3 queues and a sparse DAG, the **HEFT algorithm** gives near-optimal results in O(P log P) and is the industry standard for distributed/GPU task scheduling.

**Current state**: `compiler.rs` uses BFS (Kahn's algorithm) with FIFO tie-breaking. This produces a valid order but is not optimal. Passes at the same dependency level are scheduled arbitrarily — there is no attempt to maximize queue overlap.

### 11.2 Separation of Concerns

The Scheduler lives in the Compiler, before Oracle analysis:

```
compiler.rs::compile():
  1. flatten_recursive()       → flat_passes (unordered)
  2. assign_queues()           → queue per pass
  3. build_dependency_dag()    → adj (edge list)
  4. [NEW] schedule_heft()     → flat_passes reordered   ← Phase 5
  5. detect_cross_queue_transfers()
  ... Oracle analysis, execution ...
```

The Oracle receives `flat_passes` already in optimal order and is unaffected by scheduling changes.

### 11.3 Why Multiple Valid Orders Exist

A topological sort is any ordering that respects all dependency edges (A→B means A before B). For a DAG with P nodes and E edges, there can be exponentially many valid topological orderings. Example:

```
       GBuffer ──────────────────→ Lighting → Present
          │                             ↑
          └──→ ShadowMap                │
                                        │
       TLAS → AO_Compute → AO_Resolve ──┘
```

Both of these are valid:
```
Order A (BFS level):  GBuffer, TLAS, ShadowMap, AO_Compute, AO_Resolve, Lighting, Present
Order B (HEFT):       GBuffer, TLAS, ShadowMap, AO_Compute, AO_Resolve, Lighting, Present
                           ↑ same here, but TLAS starts at level 0 in both
```

The difference appears with async: in order A, TLAS might be delayed until after a Signal/Wait boundary. In order B, TLAS is scheduled as early as possible on the Compute queue, overlapping with GBuffer on Graphics.

### 11.4 The HEFT Algorithm

**Phase 1 — Rank computation** (backward pass from leaves):

```
rank(v) = cost(v) + max over successors u of (sync_cost(v→u) + rank(u))
```

Where:
- `cost(v)` = estimated GPU execution time of pass v
- `sync_cost(v→u)` = 0 if same queue, 1 if cross-queue (a Signal/Wait serializes)
- Base case (leaves): `rank(v) = cost(v)`

Without profiling data: `cost(v) = 1` for all passes. This is sufficient to identify the structural critical path.
With GPU timestamps: use measured frame N-1 values for frame N estimation. Smoothed with EMA to avoid instability.

**Phase 2 — Greedy scheduling** (process passes in decreasing rank order):

```
sort passes by rank descending
for each pass v in priority order:
    for each queue q compatible with pass domain:
        ready_time(q) = max(
            last_finish_time(q),
            max over predecessors p on different queue of (finish(p) + sync_cost)
        )
        EFT(v, q) = ready_time(q) + cost(v)
    assign v to queue q with minimum EFT
    update finish_time[v], last_finish_time[q]
```

For passes with fixed queue (raster → Graphics only), skip the queue selection step.

**Complexity**: O(P log P) for the sort + O(P × k) for assignment = **O(P log P)** total. Same asymptotic as current BFS sort.

### 11.5 What HEFT Optimizes

```
Current BFS (level 0 = roots, level N = deepest):
  Graphics: GBuffer(L0) → ShadowMap(L1) → ···  Signal ···  Lighting(L3) → Present(L4)
  Compute:                                  ↑ waits here → TLAS(L1) → AO(L2) → Resolve(L3)
                         [compute idle while graphics fills L0→L1]

HEFT (rank = critical path weight):
  rank(TLAS)=4, rank(GBuffer)=4, rank(AO)=3, rank(ShadowMap)=2 ...
  Graphics: GBuffer(rank4) → ShadowMap(rank2) → Lighting → Present
  Compute:  TLAS(rank4) ──→ AO_Compute(rank3) → AO_Resolve(rank2)
            ↑ starts same time as GBuffer, maximum overlap
```

Compute queue idle time: reduced from "all of level 0+1 on graphics" to zero.

### 11.6 Without Profiling: Structural Weights

When no timing data is available, `cost(v) = 1` gives the **longest-path** priority. This guarantees:
- Passes on the critical path are scheduled first on their queue
- Independent async passes start as early as possible
- Signal/Wait points are placed to minimize idle time

This is correct and already significantly better than BFS FIFO. Profiling data makes it better but is not required for correctness.

### 11.7 Adaptive Profiling (Future)

The Compiler can maintain a `PassTimingCache: HashMap<pass_name, f32>` (exponential moving average of GPU timestamps). Each frame:

```
frame N-1: measure GPU timestamps → update PassTimingCache
frame N:   HEFT uses PassTimingCache for cost(v)
```

This amortizes measurement overhead (one timestamp pair per pass = cheap) and converges to accurate scheduling within a few frames.

### 11.8 Complexity Summary

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Current BFS sort | O(P + E) | FIFO, no optimization |
| HEFT rank computation | O(P + E) | backward DFS |
| HEFT greedy assignment | O(P × k) | k ≤ 3 queues |
| **HEFT total** | **O(P log P + E)** | dominates at sort step |
| Oracle simulation | O(R + P × D) | R=resources, D=avg degree |
| **Full pipeline** | **O(P log P + R)** | P, R typically < 500 |

For a 200-pass frame: ~1600 comparisons for HEFT + ~2000 for Oracle. Sub-millisecond on CPU.
