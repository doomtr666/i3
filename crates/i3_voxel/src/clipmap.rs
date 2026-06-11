//! Toroidal brick clipmap — the SDF terrain's space structure (replaces the SVO).
//!
//! N cascade levels, each a `GRID³` toroidal grid of brick slots centred on the camera.
//! Level `l` has voxel size `LEVEL_VOXEL_SIZES[l]` (each ~2× the previous) so the finest
//! detail rings the camera and coarser shells reach out to the horizon. A world point's
//! leaf is an **O(1) lookup** (level = first window containing it, then one toroidal
//! `page_table` read) — no pointer-chasing descent, which was the SVO renderer's hotspot.
//!
//! Sparsity: only bricks a surface passes through get a slot + bake job; the rest are
//! `PAGE_EMPTY` and the ray-marcher `boxExit`s them. As the camera moves, only the cells
//! whose world-brick changed (the entering border slab) are re-evaluated — cost ∝ motion,
//! not volume. Edits (dig/build) re-bake the touched bricks via `invalidate_sphere`.
//!
//! The geometry/weight atlases, the GPU bake, the brick format and the sphere-march/
//! texturing are all reused unchanged from the SVO path — only the *which brick lives
//! where* structure changed.

use i3_math::{nalgebra::Point3, AABB};
use rayon::prelude::*;

use crate::{gpu_scene::GpuBrickJob, BRICK_BYTES, BRICK_SIZE, SdfScene};

// ─── Cascade configuration ────────────────────────────────────────────────────

pub const NUM_LEVELS: usize = 8;
/// Bricks per axis in each toroidal level. Window coverage = GRID·BRICK_SIZE·voxel_size.
pub const CLIPMAP_GRID: i32 = 32;
pub const GRID_VOL: usize = (CLIPMAP_GRID * CLIPMAP_GRID * CLIPMAP_GRID) as usize;
/// Slot capacity per level. A surface is ~2-D, so a `GRID³` window holds ≈ GRID² surface
/// bricks (thicker on overhangs/folds); this is the high-water budget per level. Total
/// `NUM_LEVELS·MAX_BRICKS_PER_LEVEL` deliberately equals the atlas capacity (MAX_SVO_BRICKS).
pub const MAX_BRICKS_PER_LEVEL: u32 = 6144;   // 8 × 6144 = 49152 = MAX_SVO_BRICKS
pub const PAGE_EMPTY: u32 = u32::MAX;

/// Voxel size (m) per level — each doubles. L0 0.05 m → window 12.8 m; L7 6.40 m → 1638 m.
pub const LEVEL_VOXEL_SIZES: [f32; NUM_LEVELS] =
    [0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20, 6.40];

const SENTINEL: [i32; 3] = [i32::MIN, i32::MIN, i32::MIN];

/// Per-level params uploaded to the shader (`clip_levels` buffer). 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClipLevelGpu {
    pub world_origin: [f32; 3], // window corner = origin_brick · (BRICK_SIZE·voxel_size)
    pub voxel_size:   f32,
}

// ─── Stats (debug UI) ─────────────────────────────────────────────────────────

#[derive(Default, Clone)]
pub struct ClipmapStats {
    pub bricks_used:    [u32; NUM_LEVELS],
    pub bricks_cap:     u32,
    pub bakes:          u32,   // bake jobs emitted this frame
    pub cells_changed:  u32,   // toroidal cells re-evaluated this frame
    pub level_overflow: bool,  // a level ran out of slots (holes)
}

// ─── State ────────────────────────────────────────────────────────────────────

struct LevelMeta {
    voxel_size:   f32,
    origin_brick: [i32; 3],   // world-brick coord of the window corner
    free:         Vec<u32>,   // recycled slot indices
    next_slot:    u32,        // fresh-slot high-water
}

pub struct ClipmapState {
    levels:       Vec<LevelMeta>,
    /// `[NUM_LEVELS · GRID_VOL]` — slot index (or `PAGE_EMPTY`) per toroidal cell.
    page_table:   Vec<u32>,
    /// `[NUM_LEVELS · GRID_VOL]` — the world-brick coord currently occupying each cell.
    /// Lets `update_camera` re-evaluate ONLY cells whose world brick changed.
    occupant:     Vec<[i32; 3]>,
    pending_jobs: Vec<GpuBrickJob>,
    first:        bool,
    frame:        ClipmapStats,
}

impl Default for ClipmapState {
    fn default() -> Self { Self::new() }
}

#[inline]
fn posmod(a: i32, m: i32) -> i32 { ((a % m) + m) % m }

/// True if a surface plausibly passes through the brick at world-brick coord `wb`.
/// Conservative (no false negatives → no holes): the band test `|center| ≤ half-diag` is
/// exact for a true distance field, and the 8-corner sign change catches surfaces the
/// centre sample misses where the (non-Lipschitz) terrain field OVERESTIMATES distance on
/// steep/folded ground (∇d→0) — the SVO's old culled-crossing-brick hole cause.
fn brick_surface(nodes: &[&crate::SdfNode], wb: [i32; 3], bws: f32, half_diag: f32) -> bool {
    let mn = [wb[0] as f32 * bws, wb[1] as f32 * bws, wb[2] as f32 * bws];
    let center = Point3::new(mn[0] + 0.5 * bws, mn[1] + 0.5 * bws, mn[2] + 0.5 * bws);
    let cs = SdfScene::sample(nodes, &center).value;
    if cs.abs() <= half_diag { return true; }
    const EPS: f32 = 1e-3;
    let (mut has_pos, mut has_neg) = (cs > EPS, cs < -EPS);
    for c in 0..8u32 {
        let p = Point3::new(
            mn[0] + if c & 1 != 0 { bws } else { 0.0 },
            mn[1] + if c & 2 != 0 { bws } else { 0.0 },
            mn[2] + if c & 4 != 0 { bws } else { 0.0 },
        );
        let v = SdfScene::sample(nodes, &p).value;
        // Surface CROSSES the cell → sign change among corners/centre.
        has_pos |= v > EPS;
        has_neg |= v < -EPS;
        if has_pos && has_neg { return true; }
        // Surface only TOUCHES/grazes a face → a corner sits on it but no sign change
        // (the cell is one-sided). The conservative centre band misses these because the
        // terrain SDF overestimates distance. Bake them too, else a ray exactly in that
        // face plane (axis-aligned view) skips into the empty cell → 1-pixel line holes.
        if v.abs() <= bws * 0.5 { return true; }
    }
    false
}

impl ClipmapState {
    pub fn new() -> Self {
        let levels = (0..NUM_LEVELS)
            .map(|l| LevelMeta {
                voxel_size:   LEVEL_VOXEL_SIZES[l],
                origin_brick: SENTINEL,
                free:         Vec::new(),
                next_slot:    0,
            })
            .collect();
        Self {
            levels,
            page_table:   vec![PAGE_EMPTY; NUM_LEVELS * GRID_VOL],
            occupant:     vec![SENTINEL;   NUM_LEVELS * GRID_VOL],
            pending_jobs: Vec::new(),
            first:        true,
            frame:        ClipmapStats::default(),
        }
    }

    // ── GPU upload accessors ────────────────────────────────────────────────────

    /// Flat `[NUM_LEVELS·GRID_VOL]` page table for the device buffer.
    pub fn page_table(&self) -> &[u32] { &self.page_table }

    /// Per-level origin + voxel size for the shader's `clipmapLookup`.
    pub fn levels_gpu(&self) -> [ClipLevelGpu; NUM_LEVELS] {
        std::array::from_fn(|l| {
            let m = &self.levels[l];
            let bws = BRICK_SIZE as f32 * m.voxel_size;
            ClipLevelGpu {
                world_origin: [
                    m.origin_brick[0] as f32 * bws,
                    m.origin_brick[1] as f32 * bws,
                    m.origin_brick[2] as f32 * bws,
                ],
                voxel_size: m.voxel_size,
            }
        })
    }

    /// Drain this frame's bake jobs (call once per frame from the setup pass).
    pub fn collect_frame_upload(&mut self) -> Vec<GpuBrickJob> {
        std::mem::take(&mut self.pending_jobs)
    }

    pub fn stats(&self) -> ClipmapStats {
        let mut s = self.frame.clone();
        s.bricks_cap = MAX_BRICKS_PER_LEVEL;
        for l in 0..NUM_LEVELS {
            s.bricks_used[l] = self.levels[l].next_slot - self.levels[l].free.len() as u32;
        }
        s
    }

    // ── Per-frame camera update ──────────────────────────────────────────────────

    /// Re-centre every level on the camera and re-bake the cells whose world brick
    /// changed (the entering border slab; the whole window on the first frame / a giant
    /// jump). Each changed cell: free any old slot, then surface-test the new world brick
    /// and (if a surface passes through it) allocate a slot + emit a bake job.
    pub fn update_camera(&mut self, cam: Point3<f32>, scene: &SdfScene) {
        self.frame = ClipmapStats::default();

        // SDF nodes once, covering the coarsest window — surface-testing per cell then
        // costs one `SdfScene::sample` (no per-cell BVH query).
        let reach = CLIPMAP_GRID as f32 * BRICK_SIZE as f32 * LEVEL_VOXEL_SIZES[NUM_LEVELS - 1];
        let big = AABB::new(
            Point3::new(cam.x - reach, cam.y - reach, cam.z - reach),
            Point3::new(cam.x + reach, cam.y + reach, cam.z + reach),
        );
        let nodes = scene.get_nodes(&big);
        let g = CLIPMAP_GRID;

        for lev in 0..NUM_LEVELS {
            let vs = self.levels[lev].voxel_size;
            let bws = BRICK_SIZE as f32 * vs;

            // Window corner brick, snapped so the window is centred on the camera.
            let ob = [
                (cam.x / bws).round() as i32 - g / 2,
                (cam.y / bws).round() as i32 - g / 2,
                (cam.z / bws).round() as i32 - g / 2,
            ];
            if !self.first && ob == self.levels[lev].origin_brick {
                continue; // this level didn't move → no cell changed
            }
            self.levels[lev].origin_brick = ob;

            let base = lev * GRID_VOL;

            // Cells whose world brick changed (toroidal cell c → world brick wb).
            let changed: Vec<(usize, [i32; 3])> = (0..GRID_VOL)
                .filter_map(|flat| {
                    let f = flat as i32;
                    let c = [f % g, (f / g) % g, f / (g * g)];
                    let wb = [
                        ob[0] + posmod(c[0] - ob[0], g),
                        ob[1] + posmod(c[1] - ob[1], g),
                        ob[2] + posmod(c[2] - ob[2], g),
                    ];
                    if self.occupant[base + flat] == wb { None } else { Some((flat, wb)) }
                })
                .collect();

            // Surface test in parallel (the only expensive step on a big move / first frame).
            let half_diag = bws * 0.866_025_4; // √3/2 · side
            let surf: Vec<bool> = changed
                .par_iter()
                .map(|(_, wb)| brick_surface(&nodes, *wb, bws, half_diag))
                .collect();

            self.frame.cells_changed += changed.len() as u32;

            // Commit: free vacated slots, then allocate + bake the surface bricks.
            for (i, (flat, wb)) in changed.iter().enumerate() {
                let idx = base + *flat;
                if self.page_table[idx] != PAGE_EMPTY {
                    self.levels[lev].free.push(self.page_table[idx]);
                    self.page_table[idx] = PAGE_EMPTY;
                }
                self.occupant[idx] = *wb;
                if surf[i] {
                    if let Some(slot) = self.alloc(lev) {
                        self.page_table[idx] = slot;
                        self.emit_job(lev, slot, *wb, vs, bws);
                    } else {
                        self.frame.level_overflow = true;
                    }
                }
            }
        }
        self.first = false;
    }

    // ── Edits ────────────────────────────────────────────────────────────────────

    /// Re-bake every brick overlapping `center±radius` after an SDF edit. A brick that lost
    /// its geometry (fully dug out) is freed; one that gained a carved surface is re-baked.
    pub fn invalidate_sphere(&mut self, center: [f32; 3], radius: f32, scene: &SdfScene) {
        let big = AABB::new(
            Point3::new(center[0] - radius - 1.0, center[1] - radius - 1.0, center[2] - radius - 1.0),
            Point3::new(center[0] + radius + 1.0, center[1] + radius + 1.0, center[2] + radius + 1.0),
        );
        let nodes = scene.get_nodes(&big);
        let g = CLIPMAP_GRID;

        for lev in 0..NUM_LEVELS {
            let vs = self.levels[lev].voxel_size;
            let bws = BRICK_SIZE as f32 * vs;
            let ob = self.levels[lev].origin_brick;
            if ob == SENTINEL { continue; }
            let half_diag = bws * 0.866_025_4;
            let base = lev * GRID_VOL;

            let wbmin = [
                ((center[0] - radius) / bws).floor() as i32,
                ((center[1] - radius) / bws).floor() as i32,
                ((center[2] - radius) / bws).floor() as i32,
            ];
            let wbmax = [
                ((center[0] + radius) / bws).floor() as i32,
                ((center[1] + radius) / bws).floor() as i32,
                ((center[2] + radius) / bws).floor() as i32,
            ];

            for wz in wbmin[2]..=wbmax[2] {
                if wz < ob[2] || wz >= ob[2] + g { continue; }
                for wy in wbmin[1]..=wbmax[1] {
                    if wy < ob[1] || wy >= ob[1] + g { continue; }
                    for wx in wbmin[0]..=wbmax[0] {
                        if wx < ob[0] || wx >= ob[0] + g { continue; }
                        let c = [posmod(wx, g), posmod(wy, g), posmod(wz, g)];
                        let idx = base + (c[0] + g * (c[1] + g * c[2])) as usize;
                        if self.page_table[idx] != PAGE_EMPTY {
                            self.levels[lev].free.push(self.page_table[idx]);
                            self.page_table[idx] = PAGE_EMPTY;
                        }
                        let wb = [wx, wy, wz];
                        self.occupant[idx] = wb;
                        if brick_surface(&nodes, wb, bws, half_diag) {
                            if let Some(slot) = self.alloc(lev) {
                                self.page_table[idx] = slot;
                                self.emit_job(lev, slot, wb, vs, bws);
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Private ──────────────────────────────────────────────────────────────────

    fn alloc(&mut self, lev: usize) -> Option<u32> {
        let m = &mut self.levels[lev];
        if let Some(s) = m.free.pop() { return Some(s); }
        if m.next_slot < MAX_BRICKS_PER_LEVEL {
            let s = m.next_slot;
            m.next_slot += 1;
            Some(s)
        } else {
            None
        }
    }

    fn emit_job(&mut self, lev: usize, slot: u32, wb: [i32; 3], vs: f32, bws: f32) {
        let atlas_offset = (lev as u32 * MAX_BRICKS_PER_LEVEL + slot) * BRICK_BYTES as u32;
        // A slot freed + reused in the same frame must keep only its latest job so the GPU
        // never executes two conflicting writes to the same atlas region.
        self.pending_jobs.retain(|j| j.atlas_offset != atlas_offset);
        self.pending_jobs.push(GpuBrickJob {
            brick_world_min: [wb[0] as f32 * bws, wb[1] as f32 * bws, wb[2] as f32 * bws],
            voxel_size:      vs,
            atlas_offset,
            _pad:            [0; 3],
        });
        self.frame.bakes += 1;
    }
}
