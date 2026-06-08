use i3_math::{nalgebra::Point3, AABB};
use i3_voxel::SdfScene;

use crate::{
    BRICK_BYTES, BRICK_SIZE, EMPTY_CAP, MAX_SVO_BRICKS, MAX_SVO_DEPTH, MAX_SVO_NODES,
    error_metric::reconstruction_residual,
    gpu_scene::{GpuBrickJob, GpuSvoNode},
};

/// Surface-refinement target: a crossing leaf below `EMPTY_CAP` keeps splitting
/// while its tangent-plane reconstruction residual exceeds this fraction of the
/// node diagonal (scaled by `sdf_weight`). ~0.01 ⇒ refine until the brick is
/// non-planar by less than ~0.14 voxel. Tuned live via the "Curve detail" slider.
const RESID_DIAG_FRAC: f32 = 0.01;

/// True if a surface may cross `aabb` — conservative, with NO false negatives.
/// The SDF is 1-Lipschitz (exact primitives + min/max CSG), so if any point in
/// the box is on the surface then `|center_sdf| ≤ half_diagonal`. Testing that
/// bound therefore never culls a box that actually contains a surface (no holes).
/// The cost is some empty bricks in the ≤half-diagonal shell around surfaces —
/// absorbed by a generous atlas rather than risking holes with sparse sampling.
fn near_surface(scene: &SdfScene, aabb: &AABB) -> bool {
    let nodes = scene.get_nodes(aabb);
    if nodes.is_empty() { return false; }
    let d = SdfScene::sample(&nodes, &aabb.center()).value.abs();
    d <= aabb.diagonal_length() * 0.5
}

/// True if a surface actually crosses `aabb` — used to decide whether a LEAF gets
/// a baked brick (vs. being left empty so the ray-marcher skips it). Samples a 3³
/// grid and reports a sign change. Denser than centre+corners so thin features
/// (torus tube, CSG niche) aren't missed; on a leaf (already refined near the
/// surface by the conservative split test) this is reliable. Empty shell nodes
/// (all one sign) are culled → no wasted brick, no grazing-angle crawl.
fn crosses_surface(scene: &SdfScene, aabb: &AABB) -> bool {
    let nodes = scene.get_nodes(aabb);
    if nodes.is_empty() { return false; }
    // STRICT sign change: a point clearly inside AND a point clearly outside.
    // The `eps` excludes samples that merely sit *on* the surface — crucial because
    // the ground plane at y=0 coincides with octree partition planes, so empty
    // nodes just above the floor have a face exactly at sdf=0. Without eps those
    // empty nodes count as "crossing" and refine into a fine empty shell → the
    // ray-marcher crawls through it and exhausts its budget → floor holes.
    const EPS: f32 = 1e-3;
    // 5³ grid (was 3³): denser sampling catches surfaces that graze a leaf without a
    // clear sign change on a coarse grid — the leaves near_surface borderline-misses
    // with the d/max(|∇d|,1) estimate. Cheap: computed once per node (cached).
    const N: usize = 5;
    let step = 1.0 / (N - 1) as f32;
    let min = aabb.min;
    let ext = aabb.max - aabb.min;
    let mut has_pos = false;
    let mut has_neg = false;
    for k in 0..N {
        for j in 0..N {
            for i in 0..N {
                let p = Point3::new(
                    min.x + ext.x * (i as f32 * step),
                    min.y + ext.y * (j as f32 * step),
                    min.z + ext.z * (k as f32 * step),
                );
                let v = SdfScene::sample(&nodes, &p).value;
                has_pos |= v > EPS;
                has_neg |= v < -EPS;
                if has_pos && has_neg { return true; }
            }
        }
    }
    false
}

// ─── SvoState ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum SvoState {
    #[default]
    Free,
    Leaf,
    Split,
}

// ─── SvoNode ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SvoNode {
    pub aabb:           AABB,
    pub state:          SvoState,
    pub brick_slot:     u32,          // u32::MAX = no brick
    pub children_start: u32,          // u32::MAX = leaf; first of 8 consecutive children
    pub parent:         u32,          // u32::MAX = root
    pub depth:          u8,
    pub octant:         u8,
    // Cached surface metrics for this node's AABB (a function of the static scene, so
    // computed ONCE at creation / on edit — never per frame). Lets the per-frame update
    // be screen-space only, no SDF/noise evaluation. Invalidated via `compute_metrics`.
    pub m_valid:   bool,   // false ⇒ metrics not yet computed (root, or after an edit)
    pub m_near:    bool,   // near_surface: box may contain a surface (conservative)
    pub m_crosses: bool,   // crosses_surface: a surface actually crosses (strict sign)
    pub m_resid:   f32,    // reconstruction residual (0 if !m_crosses)
}

impl Default for SvoNode {
    fn default() -> Self {
        Self {
            aabb:           AABB::new(Point3::origin(), Point3::origin()),
            state:          SvoState::Free,
            brick_slot:     u32::MAX,
            children_start: u32::MAX,
            parent:         u32::MAX,
            depth:          0,
            octant:         0,
            m_valid:        false,
            m_near:         false,
            m_crosses:      false,
            m_resid:        0.0,
        }
    }
}

impl SvoNode {
    pub fn to_gpu(&self) -> GpuSvoNode {
        let brick_offset = if self.brick_slot != u32::MAX {
            self.brick_slot * BRICK_BYTES as u32
        } else {
            u32::MAX
        };
        GpuSvoNode {
            aabb_min:     [self.aabb.min.x, self.aabb.min.y, self.aabb.min.z],
            first_child:  if self.state == SvoState::Split { self.children_start } else { u32::MAX },
            aabb_max:     [self.aabb.max.x, self.aabb.max.y, self.aabb.max.z],
            brick_offset,
        }
    }
}

// ─── SvoTree ──────────────────────────────────────────────────────────────────

/// Per-frame + cumulative diagnostics for the debug UI.
#[derive(Default, Clone)]
pub struct SvoStats {
    pub nodes_live:   u32,
    pub nodes_cap:    u32,
    pub bricks_used:  u32,
    pub bricks_cap:   u32,
    /// This frame's mutation counts.
    pub splits:       u32,
    pub merges:       u32,
    pub bakes:        u32,        // bake jobs emitted (split + invalidate + merge)
    pub culls:        u32,        // empty nodes that skipped/freed a brick
    /// Starvation flags — the smoking guns for "won't converge".
    pub split_wanted: u32,        // leaves that wanted to split this frame
    pub split_budget: u32,        // budget cap applied
    pub node_cap_hit: bool,       // ran out of node-pool slots
    pub brick_cap_hit:bool,       // ran out of atlas brick slots
    /// Live nodes / bricked nodes per depth.
    pub per_depth:         [u32; 16],
    pub per_depth_bricked: [u32; 16],
}

pub struct SvoTree {
    pub nodes:        Vec<SvoNode>,
    free_groups:      Vec<u32>,           // free 8-slot groups (starting index)
    brick_free:       Vec<u32>,           // free atlas brick slots
    next_brick:       u32,                // monotone allocator
    pub pending_jobs: Vec<GpuBrickJob>,   // bake jobs for this frame (cleared after upload)
    pub root_aabb:    AABB,
    pub max_depth:    u32,
    /// Mutation counters for the frame in progress; snapshotted by `stats()`.
    frame:            SvoStats,
    /// Debug: world-min last baked into each atlas slot. Used to assert that a
    /// node's slot actually holds *its* brick (not a stale previous occupant's).
    #[cfg(debug_assertions)]
    slot_baked_min:   Vec<[f32; 3]>,
    /// Debug: frame counter to throttle the per-frame invariant check.
    #[cfg(debug_assertions)]
    dbg_frame:        u32,
}

impl SvoTree {
    /// `root_aabb` should be roughly cubic for correct voxel size computation.
    pub fn new(root_aabb: AABB, max_depth: u32) -> Self {
        let root = SvoNode {
            aabb:           root_aabb,
            state:          SvoState::Leaf,
            depth:          0,
            octant:         0,
            ..SvoNode::default()   // metrics computed lazily on the first update (needs the scene)
        };
        let mut tree = Self {
            nodes:        vec![root],
            free_groups:  Vec::new(),
            brick_free:   Vec::new(),
            next_brick:   0,
            pending_jobs: Vec::new(),
            root_aabb,
            max_depth: max_depth.min(MAX_SVO_DEPTH),
            frame:        SvoStats::default(),
            #[cfg(debug_assertions)]
            slot_baked_min: vec![[f32::NAN; 3]; MAX_SVO_BRICKS as usize],
            #[cfg(debug_assertions)]
            dbg_frame:    0,
        };
        tree.emit_bake_job(0);
        tree
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Snapshot of tree diagnostics: occupancy, per-depth distribution, and this
    /// frame's mutation/starvation counters.
    pub fn stats(&self) -> SvoStats {
        let mut s = self.frame.clone();
        s.nodes_cap   = MAX_SVO_NODES;
        s.bricks_cap  = MAX_SVO_BRICKS;
        s.bricks_used = self.next_brick.saturating_sub(self.brick_free.len() as u32);
        s.nodes_live  = 0;
        s.per_depth         = [0; 16];
        s.per_depth_bricked = [0; 16];
        for n in &self.nodes {
            if n.state == SvoState::Free { continue; }
            s.nodes_live += 1;
            let d = (n.depth as usize).min(15);
            s.per_depth[d] += 1;
            if n.brick_slot != u32::MAX { s.per_depth_bricked[d] += 1; }
        }
        s
    }

    /// Re-bake (or cull, if now empty) all leaves overlapping `region` after a
    /// scene edit. A leaf that lost its geometry (e.g. after a dig) is freed.
    pub fn invalidate(&mut self, region: &AABB, scene: &SdfScene) {
        let mut stack = vec![0u32];
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx as usize];
            if node.state == SvoState::Free { continue; }
            if !node.aabb.intersects(region) { continue; }

            match node.state {
                SvoState::Leaf => {
                    // The edit changed the field here → cached metrics are stale.
                    self.compute_metrics(idx, scene);
                    self.bake_or_cull(idx, scene);
                }
                SvoState::Split => {
                    // Recompute this internal node's metrics too (its merge decision uses
                    // them), then recurse to the leaves.
                    self.compute_metrics(idx, scene);
                    let cs = self.nodes[idx as usize].children_start;
                    for i in 0..8u32 { stack.push(cs + i); }
                }
                SvoState::Free => {}
            }
        }
    }

    /// Called by SetupPass: drain pending bake jobs and snapshot GPU node data.
    pub fn collect_frame_upload(&mut self) -> (Vec<GpuBrickJob>, Vec<GpuSvoNode>) {
        // Invariant check is O(nodes); run it periodically (debug builds only) so the
        // safety net stays — corruption is caught within ~½ s — without paying the cost
        // every frame. Release builds skip it entirely.
        #[cfg(debug_assertions)]
        {
            self.dbg_frame = self.dbg_frame.wrapping_add(1);
            if self.dbg_frame % 30 == 0 {
                self.debug_check_invariants();
            }
        }

        let jobs = std::mem::take(&mut self.pending_jobs);
        let gpu_nodes: Vec<GpuSvoNode> = self.nodes.iter().map(|n| n.to_gpu()).collect();
        (jobs, gpu_nodes)
    }

    /// Detect atlas-slot aliasing: two live nodes referencing the same brick
    /// slot means one of them was never baked → ghost. Panics with the offending
    /// node indices so the corruption can be traced to its source.
    #[cfg(debug_assertions)]
    fn debug_check_invariants(&self) {
        use std::collections::HashMap;
        let mut owner: HashMap<u32, u32> = HashMap::new();
        for (idx, n) in self.nodes.iter().enumerate() {
            if n.state == SvoState::Free { continue; }

            // (1) Atlas-slot aliasing — two live nodes sharing a brick slot.
            if n.brick_slot != u32::MAX {
                if let Some(&prev) = owner.get(&n.brick_slot) {
                    panic!(
                        "SVO slot aliasing: nodes {prev} and {idx} both own brick_slot {} \
                         (states {:?} / {:?})",
                        n.brick_slot, self.nodes[prev as usize].state, n.state,
                    );
                }
                owner.insert(n.brick_slot, idx as u32);
            }

            // (1b) Stale-slot: a leaf with a brick must hold ITS OWN brick.
            // slot_baked_min[S] records the world-min last baked into slot S.
            // If a node owns S but S was last baked for a different position, the
            // node was never (re)baked → it renders a previous occupant's brick
            // ("clean block, wrong place"). Skip leaves with a pending job this
            // frame (they'll be baked before the GPU reads them).
            if n.state == SvoState::Leaf && n.brick_slot != u32::MAX {
                let off = n.brick_slot * BRICK_BYTES as u32;
                let pending = self.pending_jobs.iter().any(|j| j.atlas_offset == off);
                if !pending {
                    let baked = self.slot_baked_min[n.brick_slot as usize];
                    let mn = [n.aabb.min.x, n.aabb.min.y, n.aabb.min.z];
                    let d = (baked[0]-mn[0]).abs() + (baked[1]-mn[1]).abs() + (baked[2]-mn[2]).abs();
                    assert!(d < 1e-3,
                        "SVO STALE SLOT: leaf {idx} owns slot {} baked for {:?} but node is at {:?}",
                        n.brick_slot, baked, mn);
                }
            }

            // (2) Tree-structure: a Split node's 8 children must be live and
            // point back to it. A dangling/aliased children_start means the GPU
            // descends into garbage → ghost geometry.
            if n.state == SvoState::Split {
                let cs = n.children_start;
                assert!(cs != u32::MAX && (cs as usize + 8) <= self.nodes.len(),
                    "SVO node {idx}: Split but children_start {cs} out of range");
                for i in 0..8u32 {
                    let c = &self.nodes[(cs + i) as usize];
                    assert!(c.state != SvoState::Free,
                        "SVO node {idx}: child {} (slot {}) is Free", cs + i, cs + i);
                    assert!(c.parent == idx as u32,
                        "SVO node {idx}: child {} has parent {} (expected {idx})",
                        cs + i, c.parent);
                }
            }
        }
    }

    // ── Tree update ───────────────────────────────────────────────────────────

    /// Per-frame LOD update. Surface-aware: only leaves whose box is crossed by a
    /// surface are subdivided, so atlas bricks are spent on geometry, not empty sky.
    pub fn update(
        &mut self,
        cam:           Point3<f32>,
        vp:            &nalgebra::Matrix4<f32>,
        scene:         &SdfScene,
        lod_threshold: f32,
        sdf_weight:    f32,  // 0 ⇒ pure screen-space; >0 ⇒ residual-gated surface LOD
        split_budget:  u32,
        merge_budget:  u32,
    ) {
        self.frame = SvoStats { split_budget, ..Default::default() };

        // Root metrics are computed lazily here (the constructor has no scene). Every
        // other node gets its metrics at creation (do_split) or edit (invalidate), so
        // the per-frame phases below read the cache only — no SDF/noise evaluation.
        if self.nodes[0].state != SvoState::Free && !self.nodes[0].m_valid {
            self.compute_metrics(0, scene);
        }

        // ── Phase 1: collect + execute merges ─────────────────────────────────
        let mut merge_cands: Vec<(u32, f32)> = Vec::new();
        {
            let mut stack = vec![0u32];
            while let Some(idx) = stack.pop() {
                let node = &self.nodes[idx as usize];
                match node.state {
                    SvoState::Free => {}
                    SvoState::Leaf => {}
                    SvoState::Split => {
                        let diag = node.aabb.diagonal_length();
                        let dist = (cam - node.aabb.clamp(&cam)).norm().max(0.01);
                        // Merge (collapse children → one brick) when: out of view; the
                        // projected size dropped below half the split threshold (screen-
                        // space hysteresis); OR this node's own brick would reconstruct
                        // the surface well enough — a flat surface that earlier got
                        // over-refined. The `*0.5` is split/merge hysteresis to avoid
                        // oscillation. Short-circuits so the SDF probes run only when the
                        // cheap screen tests didn't already decide to keep it.
                        let merge = !node.aabb.is_in_frustum(vp)
                            || diag / dist < lod_threshold * 0.5
                            || (sdf_weight > 0.0
                                && node.m_crosses
                                && node.m_resid / diag * sdf_weight <= RESID_DIAG_FRAC * 0.5);
                        if merge {
                            merge_cands.push((idx, diag / dist));
                        } else {
                            let cs = node.children_start;
                            for i in 0..8u32 { stack.push(cs + i); }
                        }
                    }
                }
            }
        }
        merge_cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, _) in merge_cands.into_iter().take(merge_budget as usize) {
            self.do_merge(idx, scene);
        }

        // ── Phase 2: fresh traversal + execute splits ─────────────────────────
        let mut split_cands: Vec<(u32, f32)> = Vec::new();
        {
            let mut stack = vec![0u32];
            while let Some(idx) = stack.pop() {
                let node = &self.nodes[idx as usize];
                match node.state {
                    SvoState::Free => {}
                    SvoState::Leaf => {
                        // Cheap screen-space gate first, then the expensive SDF/BVH
                        // near-surface test — only subdivide geometry, not empty sky.
                        if u32::from(node.depth) < self.max_depth
                            && node.aabb.is_in_frustum(vp)
                            && node.m_near
                        {
                            // Bounded screen-space LOD (depth ∝ log dist). BUT only
                            // refine *empty* space (conservative shell, no actual
                            // crossing) down to EMPTY_CAP — past that, refine only
                            // nodes a surface really crosses. This keeps empty cells
                            // coarse so the ray-marcher skips them in a few big jumps
                            // instead of crawling cell-by-cell through a deeply-refined
                            // empty shell (which exhausted the step budget when close,
                            // making the finest LODs vanish).
                            let side = node.aabb.max.x - node.aabb.min.x;
                            let diag = node.aabb.diagonal_length();
                            let dist = (cam - node.aabb.clamp(&cam)).norm().max(0.01);
                            if diag / dist > lod_threshold {
                                let want = if node.m_crosses {
                                    // Surface present: refine while the brick can't
                                    // reconstruct it well — residual as a fraction of the
                                    // node diagonal (scale-invariant), scaled by the
                                    // "Curve detail" slider. Flat surfaces (residual≈0)
                                    // STOP here at any size → the floor stays coarse; curved
                                    // ones keep going. The screen-space gate above supplies
                                    // distance falloff. sdf_weight=0 ⇒ always refine
                                    // (old pure-screen-space behaviour).
                                    sdf_weight <= 0.0
                                        || node.m_resid / diag * sdf_weight > RESID_DIAG_FRAC
                                } else {
                                    // Empty shell near a surface: collapse to EMPTY_CAP so
                                    // the tracer skips it in one boxExit jump.
                                    side > EMPTY_CAP
                                };
                                if want {
                                    split_cands.push((idx, diag / dist));
                                }
                            }
                        }
                    }
                    SvoState::Split => {
                        let cs = node.children_start;
                        for i in 0..8u32 { stack.push(cs + i); }
                    }
                }
            }
        }
        split_cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        self.frame.split_wanted = split_cands.len() as u32;
        for (idx, _) in split_cands.into_iter().take(split_budget as usize) {
            self.do_split(idx, scene);
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn emit_bake_job(&mut self, node_idx: u32) {
        let slot = {
            let node = &mut self.nodes[node_idx as usize];
            if node.brick_slot == u32::MAX {
                match self.brick_free.pop().or_else(|| {
                    if self.next_brick < MAX_SVO_BRICKS {
                        let s = self.next_brick;
                        self.next_brick += 1;
                        Some(s)
                    } else {
                        None
                    }
                }) {
                    Some(s) => { node.brick_slot = s; s }
                    None => { self.frame.brick_cap_hit = true; return; }
                }
            } else {
                node.brick_slot
            }
        };
        self.frame.bakes += 1;
        let node = &self.nodes[node_idx as usize];
        let vs          = (node.aabb.max.x - node.aabb.min.x) / BRICK_SIZE as f32;
        // Byte offset of this brick in the geometry atlas (8 bytes/voxel).
        let atlas_offset = slot * BRICK_BYTES as u32;

        // A slot may get a stale job (from invalidate) and then be freed + reused
        // in the same frame.  Keep only the latest job per slot so the GPU never
        // executes two conflicting writes to the same atlas region.
        self.pending_jobs.retain(|j| j.atlas_offset != atlas_offset);

        let world_min = [node.aabb.min.x, node.aabb.min.y, node.aabb.min.z];
        #[cfg(debug_assertions)]
        { self.slot_baked_min[slot as usize] = world_min; }

        self.pending_jobs.push(GpuBrickJob {
            brick_world_min: world_min,
            voxel_size:      vs,
            atlas_offset,
            _pad:            [0; 3],
        });
    }

    /// Compute and cache the surface metrics for a node's AABB. A function of the
    /// (static) scene only, so this is the ONLY place the noise/SDF graph is evaluated
    /// for refinement — once per node at creation / on edit, never per frame.
    fn compute_metrics(&mut self, node_idx: u32, scene: &SdfScene) {
        let aabb = self.nodes[node_idx as usize].aabb;
        let near    = near_surface(scene, &aabb);
        let crosses = crosses_surface(scene, &aabb);
        let resid   = if crosses { reconstruction_residual(scene, &aabb) } else { 0.0 };
        let n = &mut self.nodes[node_idx as usize];
        n.m_near = near; n.m_crosses = crosses; n.m_resid = resid; n.m_valid = true;
    }

    /// Emit a bake job for the node only if its box is crossed by a surface;
    /// otherwise leave it empty (free any brick it held). This is the empty-space
    /// cull that keeps the atlas reserved for actual geometry.
    fn bake_or_cull(&mut self, node_idx: u32, scene: &SdfScene) {
        let _ = scene;
        // Bake every node that MIGHT contain a surface (cached metrics). near_surface is
        // conservative for a TRUE distance field; crosses_surface (strict sign change) is
        // exact even when the terrain's d/|∇d| magnitude overestimates distance where it
        // is steep (|∇d|→0, vertical/overhang) and would otherwise false-cull a crossing
        // leaf. Their union has no false negatives → no holes. Else culled (boxExit-skip).
        let n = &self.nodes[node_idx as usize];
        if n.m_near || n.m_crosses {
            self.emit_bake_job(node_idx);
        } else {
            self.frame.culls += 1;
            let slot = self.nodes[node_idx as usize].brick_slot;
            if slot != u32::MAX {
                let off = slot * BRICK_BYTES as u32;
                self.pending_jobs.retain(|j| j.atlas_offset != off);
                self.brick_free.push(slot);
                self.nodes[node_idx as usize].brick_slot = u32::MAX;
            }
        }
    }

    fn do_split(&mut self, node_idx: u32, scene: &SdfScene) {
        if self.nodes[node_idx as usize].state != SvoState::Leaf { return; }

        let children_start = if let Some(s) = self.free_groups.pop() {
            s
        } else {
            let s = self.nodes.len() as u32;
            if s + 8 > MAX_SVO_NODES { self.frame.node_cap_hit = true; return; }
            for _ in 0..8 { self.nodes.push(SvoNode::default()); }
            s
        };
        self.frame.splits += 1;

        let (mid, min, max, depth) = {
            let n = &self.nodes[node_idx as usize];
            (n.aabb.center(), n.aabb.min, n.aabb.max, n.depth)
        };

        let child_aabbs = octant_aabbs(min, mid, max);
        for i in 0..8u32 {
            let ci = (children_start + i) as usize;
            self.nodes[ci] = SvoNode {
                aabb:           child_aabbs[i as usize],
                state:          SvoState::Leaf,
                parent:         node_idx,
                depth:          depth + 1,
                octant:         i as u8,
                ..SvoNode::default()
            };
            self.compute_metrics(children_start + i, scene); // once, here (not per frame)
            self.bake_or_cull(children_start + i, scene);
        }

        self.nodes[node_idx as usize].state          = SvoState::Split;
        self.nodes[node_idx as usize].children_start = children_start;
    }

    fn do_merge(&mut self, node_idx: u32, scene: &SdfScene) {
        if self.nodes[node_idx as usize].state != SvoState::Split { return; }
        let cs = self.nodes[node_idx as usize].children_start;
        if cs == u32::MAX { return; }
        self.frame.merges += 1;

        // Recursively free the entire subtree — simple loop frees only direct
        // children, leaking any grand-children brick slots and node slots.
        for i in 0..8u32 {
            self.free_subtree(cs + i);
        }
        self.free_groups.push(cs);

        let node = &mut self.nodes[node_idx as usize];
        node.state          = SvoState::Leaf;
        node.children_start = u32::MAX;

        self.bake_or_cull(node_idx, scene);
    }

    fn free_subtree(&mut self, root_idx: u32) {
        let mut stack = vec![root_idx];
        while let Some(idx) = stack.pop() {
            let (state, cs, slot) = {
                let n = &self.nodes[idx as usize];
                (n.state, n.children_start, n.brick_slot)
            };
            match state {
                SvoState::Free => continue,
                SvoState::Leaf => {
                    if slot != u32::MAX {
                        // Remove any pending bake job for this slot immediately.
                        // Without this, a stale invalidate job emitted before
                        // update() would survive the frame and bake ghost data
                        // into a slot that might later be reused by a different node.
                        let off = slot * BRICK_BYTES as u32;
                        self.pending_jobs.retain(|j| j.atlas_offset != off);
                        self.brick_free.push(slot);
                    }
                    self.nodes[idx as usize] = SvoNode::default();
                }
                SvoState::Split => {
                    if slot != u32::MAX {
                        let off = slot * BRICK_BYTES as u32;
                        self.pending_jobs.retain(|j| j.atlas_offset != off);
                        self.brick_free.push(slot);
                    }
                    self.nodes[idx as usize] = SvoNode::default();
                    if cs != u32::MAX {
                        for i in 0..8u32 { stack.push(cs + i); }
                        self.free_groups.push(cs);
                    }
                }
            }
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn octant_aabbs(
    min: Point3<f32>,
    mid: Point3<f32>,
    max: Point3<f32>,
) -> [AABB; 8] {
    [
        AABB::new(Point3::new(min.x, min.y, min.z), mid),
        AABB::new(Point3::new(mid.x, min.y, min.z), Point3::new(max.x, mid.y, mid.z)),
        AABB::new(Point3::new(min.x, mid.y, min.z), Point3::new(mid.x, max.y, mid.z)),
        AABB::new(Point3::new(mid.x, mid.y, min.z), Point3::new(max.x, max.y, mid.z)),
        AABB::new(Point3::new(min.x, min.y, mid.z), Point3::new(mid.x, mid.y, max.z)),
        AABB::new(Point3::new(mid.x, min.y, mid.z), Point3::new(max.x, mid.y, max.z)),
        AABB::new(Point3::new(min.x, mid.y, mid.z), Point3::new(mid.x, max.y, max.z)),
        AABB::new(mid, max),
    ]
}
