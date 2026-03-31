# Frame Graph Optimizer & Sync Oracle — Technical Specification

## Overview

The **Frame Graph Optimizer** is the "brain" of the synchronization system. It sits between the **Compiler** (which determines the topological order) and the **Backend** (which records GPU commands). 

Its primary duties are:
1. **Sync Oracle**: Generating a static `SyncPlan` for pre-calculated barriers.
2. **Aliasing Analyzer**: Determining resource "vitality" (lifetimes) to enable memory reuse across transient resources.

---

## 1. The Core Data Structures

### 1.1 `ResourceState` (Stateless)
A compact, copyable representation of a resource's synchronization status.
```rust
struct ResourceState {
    layout: vk::ImageLayout,
    access: vk::AccessFlags2,
    stage: vk::PipelineStageFlags2,
    queue_family: u32,
}
```

### 1.2 ResourceFlowState (Transient)
Used during analysis to track the "current" state of a resource as we simulate the frame.
```rust
struct ResourceFlowState {
    current: ResourceState,
    last_write_frame: u64,
    has_been_used_this_frame: bool,
    is_required: bool, // Used for DCE
}
```

### 1.3 `SyncPlan` (The Output)
A static map indexed by `PassId`.
```rust
struct SyncPlan {
    // Barriers to be executed BEFORE the pass commands
    acquire_barriers: HashMap<PassId, Vec<vk::ImageMemoryBarrier2>>,
    // Barriers to be executed AFTER the pass commands (typically Release)
    release_barriers: HashMap<PassId, Vec<vk::ImageMemoryBarrier2>>,
    // Main transitions (Layout changes, Read/Write sync)
    transitions: HashMap<PassId, Vec<vk::ImageMemoryBarrier2>>,
    // Final state of each resource to be committed back to the backend
    final_states: HashMap<ResourceId, ResourceState>,
}
```

---

## 2. The Analysis Algorithm (The Oracle)

The analysis is a single-threaded "dry run" of the frame's execution order.

### Phase 1: Initialization
- Initialize `ResourceFlowMap` from the Backend's **Canonical State** (the state left after the previous frame).
- Mark all resources as `has_been_used_this_frame = false`.

### Phase 2: Simulation Loop
For each `PassDescriptor` in the order provided by the Compiler:
1. **Identify Required State**: Determine the target `layout`, `access`, and `stage` based on the pass's `ResourceUsage`.
2. **First Use Detection**: 
   - If `!state.has_been_used_this_frame`:
     - Set `old_layout = vk::ImageLayout::UNDEFINED`.
     - Transition to target state.
     - If usage includes `WRITE`, set pass `load_op = CLEAR`.
     - `state.has_been_used_this_frame = true`.
3. **Queue Migration**:
   - If `current_queue_family != pass_queue_family`:
     - Generate a **Release Barrier** (src: `current_queue`, dst: `pass_queue`).
     - Generate an **Acquire Barrier** (src: `current_queue`, dst: `pass_queue`).
     - Inject Release into the *previous* pass that owned the resource.
     - Inject Acquire into the *current* pass.
4. **Transition Sync**:
   - If `current_state` is incompatible with `required_state` (e.g. Layout mismatch, or Write-after-Read hazard):
     - Generate a `vk::ImageMemoryBarrier2`.
     - Update `current_state` to `required_state`.

---

## 3. Multi-Queue Synchronization

Vulkan ownership transfers are handled explicitly for images in `EXCLUSIVE` sharing mode.

### 3.1 Release/Acquire Mechanics
- **Release**: Must specify `srcQueueFamilyIndex` and `dstQueueFamilyIndex`. `dstStageMask` and `dstAccessMask` must be `0` (v1) or empty (v2).
- **Acquire**: Must mirrored indexes. `srcStageMask` and `srcAccessMask` must be `0`/empty.
- **Constraint**: Both barriers must use the **same** `oldLayout` and `newLayout`.

### 3.2 Detection in the Oracle
The Oracle tracks the **Last Owner** pass for every resource. When a new pass on a different queue requests the resource:
1. It looks up the `LastOwnerPassId`.
2. It appends the **Release** barrier to the `SyncPlan` for `LastOwnerPassId`.
3. It appends the **Acquire** barrier to the `SyncPlan` for the current pass.

---

## 4. Resource Aliasing & Vitality Analysis

Since the Oracle performs a linear simulation of the frame, it is the ideal place to compute resource **Lifetimes**.

### 4.1 Lifetime Intervals
For each resource $R$, the Oracle tracks:
- **First-Use Pass $P_{start}$**: The first pass that reads or writes to $R$.
- **Last-Use Pass $P_{end}$**: The final pass (in execution order) that accesses $R$.
- **Vitality Interval**: $[P_{start}, P_{end}]$.

### 4.2 Aliasing Logic
Two transient resources $A$ and $B$ can **alias** (share memory) if:
1. Their vitality intervals are **disjoint**: $A_{end} < B_{start}$ or vice versa.
2. Their physical requirements (Size, Alignment, Usage Flags) are compatible.
3. They are both marked as `Transient`.

### 4.3 Implementation: Memory Heap Reuse
The Optimizer produces an **Aliasing Map**:
- `Resource A -> (HeapId, Offset 0)`
- `Resource B -> (HeapId, Offset 0)` (if A is dead when B starts).

### 4.4 Memory Management Strategy: Role of VMA
It is important to distinguish between **Physical Allocation** and **Logical Aliasing**:
- **Static/External Resources**: Use VMA directly for individual allocations. This ensures long-lived resources (textures, meshes) are efficiently sub-allocated and managed.
- **Transient Frame Resources**: The Oracle calculates offsets within **Large Backing Heaps** (e.g. 512MB blocks) allocated via VMA. 
- **Benefit**: This strategy prevents hitting the `maxMemoryAllocationCount` limit of `VkPhysicalDeviceLimits` by grouping hundreds of transient resources into a handful of large device memory objects.
- **Heap Rotation**: For $N$ frames in flight, we rotate through $N$ sets of these heaps to ensure that Frame $N+1$ does not overwrite memory still being read by Frame $N$ on the GPU.

---

## 5. Dead Pass & Dead Resource Culling (DCE)

The Optimizer can intelligently prune the execution plan by identifying "Dead" passes that contribute nothing to the final output.

### 5.1 The Culling Algorithm
1. **Seed "Root" passes**: Any pass with side effects (calls to `Present`, writes to `External` or `History` resources) is marked as **Live**.
2. **Propagate Liveness**:
   - For every **Live** pass, trace its input resources (Reads).
   - Mark the passes that **Write** to these resources as **Live**.
3. **Repeat** until no more passes can be promoted to Live.
4. **Culling**:
   - Any pass not marked as **Live** is removed from the `SyncPlan`.
   - Any transient resource that is only written by a **Dead** pass is never allocated.

### 5.2 Benefits
- **Runtime Branching**: If a shader-driven feature (e.g. SSAO) is toggled off, the engine automatically skips the entire sub-graph of AO passes without manual `if (ssao_enabled)` checks in the backend.
- **Dynamic Resource Recycling**: Dead resources do not consume memory in the Aliasing Heaps.

---

## 6. Optimization Stones (Lifting the Stones)

### 5.1 Barrier Batching
Instead of calling `vkCmdPipelineBarrier2` for every single resource, the Backend's execution phase loops through all pre-calculated barriers for the pass and emits **one single call** with multiple image/buffer barrier pointers.

### 5.2 Concurrent Read Overlap
If multiple consecutive passes only perform `READ` operations (e.g. GBuffer resolve followed by SSAO sampling), the Oracle detects that the required layout is consistent and **emits zero barriers** between them.

### 5.3 `UNDEFINED` Promotion
By promoting the first use of the frame to `UNDEFINED`, we explicitly tell the GPU it doesn't need to perform any data-preserving tile loads during the transition, which is a major win for mobile or tile-based GPUs and helpful for memory compression on desktop.

---

## 6. Implementation Roadmap in `i3_vulkan_backend`

1. **`analyze_sync` implementation**: Move logic from `commands.rs` to a new `SyncOracle` struct in `sync.rs`.
2. **Stateless `prepare_pass`**: Redesign to accept its slice of the `SyncPlan`.
3. **Commit Phase**: Update `PhysicalImage/Buffer` state in `VulkanBackend` only at the very end of `execute()`, using the `final_states` from the `SyncPlan`.

---

## 7. Diagnostic & Safety

### 7.1 Plan Dumper
A debug tool to print the `SyncPlan` as a table:
```text
PASS: ShadowMap
  [Image: DepthBuffer] | UNDEFINED -> DEPTH_ATTACHMENT | load_op: CLEAR
PASS: GBuffer
  [Image: DepthBuffer] | DEPTH_ATTACHMENT -> DEPTH_READ_ONLY | queue: Graphics
...
```

### 7.2 Visual Debugging (DOT Visualization)
To "lift the stones" during development, the Optimizer can export binary DAG states:
- **`graph_raw.dot`**: The initial dependency DAG before any synchronization or aliasing logic is applied.
- **`graph_optimized.dot`**: The final execution plan, annotated with:
    - Barrier edges (red for cross-queue, blue for intra-queue).
    - Aliasing groups (nodes of the same color share the same memory heap).
    - Queue assignments.

### 7.3 Structured Tracing (Sync Log)
A dedicated `tracing::span` for the Optimizer will log every atomic decision:
- `DEBUG` [Pass: X] Promoting `Image: Y` to `UNDEFINED` (first frame use).
- `INFO` [Resource: Z] Reclaiming memory for `Resource: W` (aliasing).
- `WARN` [Pass: A -> B] Injecting Release/Acquire pair for `Queue: Graphics -> Compute`.

### 7.4 Hazard Detection
During simulation, the Oracle can detect "Invalid Layout" or "Missing Release" errors before they ever reach the GPU.
