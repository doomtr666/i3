# SDF Sparse Voxel Octree (`i3_sdf`)

> **Status (2026-06)** — This document describes the **current** implementation in
> the `i3_sdf` crate. It replaces the original multi-level *clipmap* brickmap design
> (kept in git history). The clipmap was a fixed 10-level toroidal grid; the SVO is an
> adaptive octree that concentrates detail where it is needed and renders a 2 km view.

## 1. Purpose

Render a Signed Distance Field scene (a CSG tree of analytic primitives) by
**baking** the field into a sparse voxel octree of small bricks and **sphere-tracing**
that octree on the GPU, instead of evaluating every primitive analytically at every
ray step. The analytic path stays competitive for a handful of primitives; the SVO is
the scalable path for complex scenes and (later) baked / streamed content.

Everything — CPU tree, GPU buffers, compute/graphics passes, and the Slang shaders —
lives in one crate (`crates/i3_sdf`). The demo (`examples/sdf`) only builds the scene,
drives the camera, and wires the passes into the render graph.

```
CPU  SdfScene (BVH of primitives)         SvoTree (adaptive octree, this crate)
        │                                     │  per-frame update(): split / merge
        │  pack_scene / pack_bvh              │  emits GpuBrickJob list + node snapshot
        ▼                                     ▼
GPU  prims[]  bvh[]   node_pool[]   jobs[]   (CpuToGpu, ring-buffered ×3)
        │                                     │
        │  SvoBakePass (compute, 9³/brick)    │
        ▼                                     ▼
     geom_atlas[]  (GpuOnly, 8 bytes/voxel: dist + normal + material)
        │
        ▼  SvoRenderPass (fullscreen fragment)
     sphere-trace octree → tangent-plane SDF → G-buffer (albedo / normal / depth / …)
```

## 2. The CPU octree — `svo.rs`

### 2.1 Node and tree

A node is a cube AABB with one of three states:

```rust
enum SvoState { Free, Leaf, Split }

struct SvoNode {
    aabb, depth, octant, parent,
    state,
    brick_slot,      // u32::MAX = no brick (empty / interior)
    children_start,  // u32::MAX = leaf; else first of 8 consecutive children
}
```

`SvoTree` owns a flat `Vec<SvoNode>` plus free lists:
- `free_groups` — recycled 8-node child groups (so `nodes` doesn't grow forever).
- `brick_free` — recycled atlas brick slots.
- `next_brick` — monotone slot allocator (used when `brick_free` is empty).
- `pending_jobs: Vec<GpuBrickJob>` — bricks to bake this frame (drained by the setup pass).

The root is one cube (the demo uses **2048 m**, centred, `max_depth = 16`). Only the
root must be cubic — `voxel_size = side / BRICK_SIZE` is applied uniformly on all axes.

### 2.2 Per-frame update — `update(cam, vp, scene, lod_threshold, …, split_budget, merge_budget)`

Two phases, each a single stack traversal, both budget-limited:

**Phase 1 — merges.** For each `Split` node, compute `diag / dist` (projected size).
Merge it (collapse its subtree back to a leaf) if it is **out of frustum** or its
projected size dropped below `lod_threshold * 0.5` (hysteresis). Merging frees the
subtree's node groups and brick slots back to the free lists.

**Phase 2 — splits.** For each `Leaf`, split it (allocate 8 children) when **all** hold:
- `depth < max_depth`
- the node is **in frustum**
- `near_surface(node)` — conservative "might contain a surface" (see 2.3)
- `diag / dist > lod_threshold` — bounded screen-space LOD (depth ∝ log distance,
  so it can never explode)
- `side > EMPTY_CAP` **OR** `crosses_surface(node)` — the empty-space cap (see 2.4)

Candidates are sorted by `diag / dist` (closest/largest first) and truncated to the
budget, so the most visually important refinement happens first when the budget is tight.

### 2.3 Two surface tests (the heart of the no-holes / no-crawl trade-off)

The SDF is **1-Lipschitz** (exact primitive distances combined with `min`/`max` CSG),
which makes two cheap, *correct* tests possible:

- **`near_surface(aabb)`** — `|sdf(center)| ≤ half_diagonal`. By Lipschitz, if any
  point of the box is on the surface then this is true, so it has **no false
  negatives**: it never culls a box that contains a surface → **no holes**. Used to
  gate **subdivision** (descend toward every surface) and **baking** (`bake_or_cull`).
  Its false *positives* (the empty shell within half-a-diagonal of a surface) are the
  cost, paid by a generous atlas.

- **`crosses_surface(aabb)`** — samples a 3³ grid and requires a **strict** sign change
  (a point with `sdf > +ε` *and* one with `sdf < -ε`, `ε = 1 mm`). True only when a
  surface actually passes *through* the box. Used to decide whether to keep refining
  **below `EMPTY_CAP`**. The strict `ε` is essential: the ground plane sits at `y = 0`,
  which coincides with octree partition planes, so empty boxes just above the floor have
  a face exactly at `sdf = 0`; a non-strict test would count that as "crossing" and
  refine a fine empty shell (see 2.4).

### 2.4 `EMPTY_CAP` — keep empty space coarse

A sphere-tracer skips a cell with one `boxExit` jump, but only if the cell has **no
brick** (it is culled) or is **large**. If the conservative `near_surface` shell is
refined to fine empty cells, the ray crawls through them one cell per step and exhausts
its step budget → the finest LODs vanish. So:

> Below `EMPTY_CAP` (0.5 m), **only nodes a surface actually crosses keep refining**.
> Empty-but-near nodes stop at `EMPTY_CAP`. Surfaces refine all the way to the LOD limit.

`near_surface` still drives the *descent* (so thin features — the torus tube, a CSG
niche — are reached), and `crosses_surface` takes over the *fine* refinement decision.

### 2.5 Baking decision — `bake_or_cull(node)`

When a node becomes a leaf (after split / merge / `invalidate`), bake it iff
`near_surface` (conservative → no holes). Otherwise free its brick slot. Empty leaves
that are genuinely far from any surface get **no brick** and are skipped by the tracer.

### 2.6 Brick jobs — `emit_bake_job(node)`

Allocates a brick slot (reusing `brick_free`, else `next_brick`) and pushes a
`GpuBrickJob` describing where to bake. `voxel_size = side / 8`; the brick's byte offset
in the atlas is `slot * BRICK_BYTES` (`BRICK_BYTES = 729 * 8 = 5832`). A `retain` removes
any stale job for the same slot (a slot can be freed and reused in one frame) so the GPU
never double-writes a slot.

### 2.7 Edits & animation — `invalidate(region, scene)`

After the scene changes (dig/fill, the orbiting gem), every leaf overlapping `region`
is re-evaluated by `bake_or_cull` — re-baked if still near a surface, freed if it became
empty.

## 3. GPU data — `gpu_scene.rs`, `gpu_buffers.rs`

### 3.1 Structures (mirror the Slang structs exactly)

```
GpuSvoNode (32 B)   aabb_min, first_child(u32::MAX=leaf), aabb_max, brick_offset(u32::MAX=none)
GpuBrickJob (32 B)  brick_world_min, voxel_size, atlas_offset, _pad0, _pad1, _pad2
GpuPrimitive(80 B)  inv-rotation rows + translation, inv_scale, data0..2, type, subtract, material
GpuBvhNode (32 B)   aabb_min, left(u32::MAX=leaf), aabb_max, right_or_prim
```

> **Layout gotcha.** Mirror Slang structs with **scalar** trailing pads, never a
> `uint3`/`float3`: a vec3 has 16-byte base alignment in std430, so a trailing `uint3`
> pad would inflate `GpuBrickJob` to **48 B** on the GPU while the `#[repr(C)]` CPU struct
> stays **32 B** → every `jobs[i>0]` reads shifted fields → uniform corruption.

### 3.2 Brick & atlas layout — 8 bytes/voxel, gradient-augmented

A brick is `(BRICK_SIZE+1)³ = 9³ = 729` voxels — the `+1` is a **one-voxel positive
overlap** per axis so the sampler reads a same-LOD neighbour's first voxel without
clamping (seamless same-LOD bricks). Each voxel is **8 bytes** (`BRICK_BYTES = 729*8 =
5832`), byte offset `slot * BRICK_BYTES + vox*8`, little-endian:

| bytes | field | encoding |
|---|---|---|
| 0–1 | signed distance | **snorm16** over `±BAND·voxel_size` (`BAND = 4`) |
| 2–3 | octahedral normal X | snorm16 |
| 4–5 | octahedral normal Y | snorm16 |
| 6   | material id | u8 |
| 7   | flags | u8 (edited / veg-hint / reserved) |

Distance decode: `dist = snorm16 * BAND * voxel_size`. The bake and render share `BAND`.
We deliberately **avoid `f16`** for distance — `f32tof16`/`f16tof32` on the Slang→SPIRV
path produced garbage; snorm16-over-a-band is a reliable integer round-trip and gives
~`voxel_size/8192` precision near the surface (finer than f16 there). Cells straddling
the surface are within `±√3·voxel_size ≪ band`, so the reconstruction below is exact for
them; far cells clamp to `±band` (a valid "far" value for sphere-marching).

**Why store a normal per voxel (gradient-augmented bricks).** Plain trilinear blending of
distance *values* is only C⁰ and O(h²). Instead each voxel stores its analytic normal
`n_i = ∇d` (free at bake time, via central differences on the CSG field), defining a
**tangent plane** `d_i(p) = d_i + n_i·(p − corner_i)`. Trilinearly blending the eight
corner planes is exact for planar surfaces, Hermite-smooth (O(h³)) for curved ones, and
the blended field stays 1-Lipschitz (`|Σ wᵢ nᵢ| ≤ 1`) so it is safe to sphere-trace. The
blended normal `Σ wᵢ nᵢ` is reused directly as the shading normal — no finite differences
on the quantised field. Net effect: far smoother surfaces/normals at a given brick
density (or fewer bricks at a given error).

### 3.3 Ring-buffered CPU→GPU buffers (`RING = 3`)

`node_pool`, `jobs`, `prims`, `bvh` are `CpuToGpu` and **triple-buffered**. The renderer
keeps 3 frames in flight and `map_buffer` returns a single allocation, so writing frame
N must not land on a buffer the GPU is still reading for frame N-1/N-2. The frame fence
guarantees frame N-3 is done, so `buf[frame % 3]` is always free. The setup pass advances
the ring slot once per frame and imports that slot under a stable name; the bake/render
passes resolve by name and get the same slot.

The **atlas** (`geom_atlas`) is a single `GpuOnly` buffer — it is persistent (bricks
accumulate across frames) and made consistent by single-queue ordering, so it must
**not** be ring-buffered.

## 4. The three GPU passes — `passes/`

1. **`SvoSetupPass`** (pre-G-buffer, compute domain, no dispatch). Drains the tree's
   pending jobs and node snapshot, packs the SDF scene (`prims`, `bvh`), and uploads all
   of them to the active ring slot via `map_buffer`. First frame: clears the atlas.
   It **declares the CpuToGpu buffers it map-writes as `SHADER_WRITE`** — even though the
   write is a host map — so the frame graph sees a real write→read dependency and
   **serialises** setup → bake → render. Without it the graph (which records passes in
   parallel on rayon threads) would run setup and bake concurrently and race the
   `job_count` side-channel atomic → truncated dispatch → corruption.

2. **`SvoBakePass`** (pre-G-buffer, compute). Dispatches `job_count` workgroups of
   `[9,9,9] = 729` threads (`svo_bake.slang`). Each **thread bakes one voxel directly** (no
   LDS): it evaluates the CSG distance + nearest material, the analytic normal via central
   differences, and `Store2`s the 8-byte voxel into `geom_atlas` at its byte offset.

3. **`SvoRenderPass`** (G-buffer, fullscreen fragment, `svo_render.slang`). Sphere-traces
   the octree per pixel and writes the deferred G-buffer (albedo, octahedral normal,
   rough/metal, emissive, HiZ, depth). Reads `node_pool`, `geom_atlas`, and `prims` (the
   last only for the analytic debug modes).

The factory `create_svo_passes()` returns `(compute_passes, render_pass)` so the demo
pushes the compute passes into `extra_pre_gbuffer_passes` and the render pass into
`extra_gbuffer_passes`.

## 5. The ray-marcher — `svo_render.slang`

`sampleSvo(p)` descends from the root, choosing the octant by `p > mid` per axis, until
it reaches a leaf. It returns one of:
- **surface leaf** (has a brick): the **tangent-plane reconstruction** — load the 8 corner
  voxels (distance + stored normal), form each corner's plane `d_i + n_i·(p − corner_i)`,
  blend with trilinear weights for the distance, and blend the corner normals `Σ wᵢ nᵢ`
  for the shading normal (returned directly, no finite differences). Plus `voxel_size` and
  the dominant-corner material;
- **empty leaf** (no brick) or **outside the tree**: the cell bounds, so the tracer can
  skip exactly to the cell exit.

The trace loop (start `t = near plane = 0.1`, ≤ 512 steps):
- not inside the tree → stop (ray left the scene);
- empty cell → `t += boxExit(p, rd, cell)` (skip the whole cell, no crawl);
- surface, `sdf < voxel_size·0.08` → hit (refined to the zero-crossing along the
  gradient to remove the threshold bulge);
- else sphere-march `t += sdf·0.8`, **clamped to the leaf exit** so it never steps past a
  surface that begins in a neighbouring finer leaf.

> **Hard constraint:** the shader's `MAX_SVO_DEPTH` (currently **18**) bounds the descent
> loop and **must be ≥ the tree's `max_depth`** (16). If it is smaller, nodes deeper than
> the cap are unreachable and the finest-LOD cubes stop rendering (holes that appear as
> you approach). This bit us hard — see `doc` memory.

## 6. Constants

| Constant | Value | Meaning |
|---|---|---|
| `BRICK_SIZE` | 8 | voxels per brick axis (9³ with overlap) |
| `VOXEL_BYTES` / `BRICK_BYTES` | 8 / 5832 | bytes per voxel / per brick |
| `BAND` | 4.0 | distance snorm16 half-range, in voxels (bake/render must match) |
| `EMPTY_CAP` | 0.5 m | below this, only surface-crossing nodes keep refining |
| `MAX_SVO_DEPTH` | 16 (CPU) / 18 (shader) | shader ≥ CPU, always |
| `MAX_SVO_NODES` | 262 144 | node pool cap |
| `MAX_SVO_BRICKS` | 49 152 | atlas cap (≈ 286 MB, single atlas) |
| `RING` | 3 | CpuToGpu ring depth = frames in flight |
| `BASE_SIZE` | 4.0 m | *(currently unused — reserved)* |

## 7. Debug tooling (`debug_ui.rs`, shader `debug_flags`)

Kept in tree (low-overhead): the assertion and slot tracking are `#[cfg(debug_assertions)]`
only; the shader debug paths run only when their flag bit is set.

- **CPU stats** (per-frame): node/brick occupancy %, splits/merges/bakes/culls,
  `split wanted vs budget`, and ⚠ flags for node/brick cap exhaustion; per-depth histogram.
- **Render modes** (mutually-exclusive triangulation tools):
  - bit 0 SVO depth colours · bit 1 node AABBs · bit 2 error heat · bit 3 step-count heat
  - **bit 4 — ground truth**: sphere-trace the analytic CSG, ignoring tree *and* atlas.
  - **bit 5 — traversal + analytic**: use the octree traversal but sample analytic SDF at
    the leaf (isolates traversal from brick content).
  - **bit 6 — brick error**: at the brick hit, `|analytic SDF|` as heat (red = wrong brick).
- **Freeze** toggles (tree / gem) to separate static vs dynamic issues.
- **Invariant assertion** (`debug_check_invariants`): slot aliasing, tree-structure
  consistency, and stale-slot detection (a leaf whose slot was last baked for a different
  position) — panics with exact indices at the moment of corruption.

These three render modes were what turned "it's broken" into a binary search across the
data path (scene → traversal → brick content → tracer) and found every bug.

## 8. Known limitations / next steps

- **Gradient-augmented bricks — done.** Each voxel now stores its analytic normal
  alongside the distance (§3.2); the leaf reconstruction is the tangent-plane blend (§5).
  Two non-obvious bugs were paid for on the way (both worth remembering): `f16` distance
  intrinsics produced garbage on this Slang→SPIRV path (→ snorm16-over-a-band), and a
  trailing `uint3` pad in `GpuBrickJob` inflated its std430 stride to 48 B vs the 32 B CPU
  struct (→ scalar pads).
- **Material**: proximity-weighted trilinear blend of the 8 corner materials
  (`exp(-max(sd,0)/vs)`), ported from the clipmap — done (`sampleMaterial`).
- **Error-driven refinement — done** (`error_metric::reconstruction_residual`). A surface
  leaf below `EMPTY_CAP` keeps splitting only while its residual (true SDF probed one voxel
  along the surface tangents, where the tangent-plane reconstruction predicts 0) exceeds
  `RESID_DIAG_FRAC` of the node diagonal, scaled by the `sdf_weight` ("Curve detail")
  slider. Flat surfaces (residual 0) stop at `EMPTY_CAP`; curved ones refine where they
  deviate; the screen-space gate supplies distance falloff. Crucially the residual is
  **bounded by ~`voxel_size` on sharp edges** (1-Lipschitz field), so — unlike the old
  curvature term — it does not run away on box corners. `sdf_weight = 0` falls back to pure
  screen-space LOD.
- **LOD seams (T-junctions)** between neighbour nodes of different depth are not blended;
  exact for the flat floor (linear ⇒ tangent-exact), mild on curves now that each brick
  fits better. The error metric above is the proper fix.
- **Empty-shell cost / grazing crawl**: the conservative bake spends bricks on a thin
  shell around surfaces, and the tracer sphere-marches baked cells clamped to the leaf exit
  → at grazing angles it can crawl and exhaust its 512-step budget. A per-brick "empty"
  flag (skip via `boxExit` without holes) or a coarser empty cap would help.
- **Perf**: per-pixel traversal dominates; the SVO only beats analytic brute force on
  complex scenes. `near_surface`/`crosses_surface` (BVH + samples per candidate) could be
  cached in the node if the CPU update becomes a bottleneck on large scenes.
