use half::f16;
use i3_math::AABB;
use i3_voxel::SdfScene;
use nalgebra::Point3;
use rayon::prelude::*;

use crate::gpu_scene::GpuBrickJob;
use crate::{BakeState, MAX_BRICKS_PER_LEVEL};

pub const BRICK_SIZE: u32 = 8;
/// Atlas voxels per brick = (BRICK_SIZE+1)³ — includes 1-voxel positive overlap on each axis
/// so the trilinear sampler can fetch cross-brick corner data without clamping.
pub const BRICK_VOXELS: usize = ((BRICK_SIZE + 1) * (BRICK_SIZE + 1) * (BRICK_SIZE + 1)) as usize; // 9³ = 729
/// DWORDs per brick in the u8-packed atlas: ceil(BRICK_VOXELS / 4) = 183
pub const BRICK_DWORDS: usize = (BRICK_VOXELS + 3) / 4;

// ─── BrickmapBaker ────────────────────────────────────────────────────────────

pub struct BrickmapBaker {
    pub grid_dims:        [u32; 3],
    pub world_origin:     [f32; 3],
    pub voxel_size:       f32,
    /// Atlas capacity; new brick allocation is skipped when reached.
    /// Re-baking existing bricks (page_table[flat] != u16::MAX) is always allowed.
    pub max_atlas_bricks: u32,
}

impl BrickmapBaker {
    /// World-space size of one brick (= BRICK_SIZE × voxel_size).
    pub fn brick_world_size(&self) -> f32 {
        BRICK_SIZE as f32 * self.voxel_size
    }

    /// Half-diagonal of a brick in world space — used for SDF normalization.
    pub fn brick_half_diag(&self) -> f32 {
        self.brick_world_size() * (3.0_f32).sqrt() * 0.5
    }

    /// AABB of brick `(bx, by, bz)` in world space.
    fn brick_aabb(&self, bx: u32, by: u32, bz: u32) -> AABB {
        let bws = self.brick_world_size();
        let min = Point3::new(
            self.world_origin[0] + bx as f32 * bws,
            self.world_origin[1] + by as f32 * bws,
            self.world_origin[2] + bz as f32 * bws,
        );
        AABB::new(min, Point3::new(min.x + bws, min.y + bws, min.z + bws))
    }

    /// Flat index into page_table for brick `(bx, by, bz)`.
    pub fn page_idx(&self, bx: u32, by: u32, bz: u32) -> usize {
        let [gx, gy, _gz] = self.grid_dims;
        (bx + gx * (by + gy * bz)) as usize
    }

    // ── Bake a single brick — returns None if the brick is empty ──────────────

    fn bake_one(
        &self,
        bx: u32,
        by: u32,
        bz: u32,
        sdf: &SdfScene,
        half_diag: f32,
    ) -> Option<(Vec<u16>, Vec<u8>)> {
        let brick_aabb = self.brick_aabb(bx, by, bz);
        // Expand query by 2 voxels so boundary primitives aren't missed
        let query_aabb = brick_aabb.expand(self.voxel_size * 2.0);
        let nodes = sdf.get_nodes(&query_aabb);

        if nodes.is_empty() {
            return None;
        }

        // Cull bricks whose centre is far from any surface
        let center = brick_aabb.center();
        let center_sdf = SdfScene::value(&nodes, &center);
        if center_sdf.abs() > half_diag * 2.0 {
            return None;
        }

        // Evaluate SDF at 9³ positions (BRICK_SIZE+1 per axis), including a
        // 1-voxel overlap on the positive side of each axis.  The overlap voxel
        // at index BRICK_SIZE along each axis corresponds to the neighbouring
        // brick's first voxel, giving the trilinear sampler valid cross-brick
        // corner data instead of clamped boundary values.
        let atlas_dim = BRICK_SIZE + 1;
        let mut sdf_samples = Vec::with_capacity(BRICK_VOXELS);
        let mat_samples = vec![0u8; BRICK_VOXELS];

        for sz in 0..atlas_dim {
            for sy in 0..atlas_dim {
                for sx in 0..atlas_dim {
                    let p = Point3::new(
                        brick_aabb.min.x + (sx as f32 + 0.5) * self.voxel_size,
                        brick_aabb.min.y + (sy as f32 + 0.5) * self.voxel_size,
                        brick_aabb.min.z + (sz as f32 + 0.5) * self.voxel_size,
                    );
                    let d = SdfScene::value(&nodes, &p);
                    let norm = (d / half_diag).clamp(-1.0, 1.0);
                    sdf_samples.push(f16::from_f32(norm).to_bits());
                }
            }
        }

        Some((sdf_samples, mat_samples))
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// CPU-side preparation for GPU baking (toroidal clipmap).
    ///
    /// `origin_brick` is the world-brick coordinate of the grid corner for this level.
    /// Cell (cx,cy,cz) represents world-brick (ob + offset) where
    /// offset = (cell - ob%grid + grid) % grid, so brick_world_min = (ob + offset) * bws.
    pub fn prepare_batch_gpu(
        &self,
        sdf: &SdfScene,
        data: &mut BrickmapData,
        state: &mut BakeState,
        budget: usize,
        lev: usize,
        origin_brick: [i32; 3],
    ) -> Vec<GpuBrickJob> {
        let half_diag = self.brick_half_diag();
        let bws = self.brick_world_size();
        let [gx, gy, gz] = self.grid_dims.map(|v| v as i32);
        let [ob_x, ob_y, ob_z] = origin_brick;
        let ob_x_mod = ((ob_x % gx) + gx) % gx;
        let ob_y_mod = ((ob_y % gy) + gy) % gy;
        let ob_z_mod = ((ob_z % gz) + gz) % gz;

        let count = budget.min(state.pending.len());
        let batch: Vec<[u32; 3]> = state.pending.drain(..count).collect();
        state.last_baked = 0;

        let mut jobs = Vec::with_capacity(batch.len());

        for [cx, cy, cz] in batch {
            // Toroidal: compute offset from origin, then world-space min of this cell
            let off_x = ((cx as i32 - ob_x_mod + gx) % gx) as f32;
            let off_y = ((cy as i32 - ob_y_mod + gy) % gy) as f32;
            let off_z = ((cz as i32 - ob_z_mod + gz) % gz) as f32;
            let world_min = [
                (ob_x as f32 + off_x) * bws,
                (ob_y as f32 + off_y) * bws,
                (ob_z as f32 + off_z) * bws,
            ];

            let brick_aabb = AABB::new(
                Point3::new(world_min[0], world_min[1], world_min[2]),
                Point3::new(world_min[0] + bws, world_min[1] + bws, world_min[2] + bws),
            );
            let query_aabb = brick_aabb.expand(self.voxel_size * 2.0);
            let nodes      = sdf.get_nodes(&query_aabb);
            let flat       = self.page_idx(cx, cy, cz);

            if nodes.is_empty() {
                // Brick is empty — free its atlas slot if it had one
                let old = data.page_table[flat];
                if old != u16::MAX { data.free_slot(old); }
                data.page_table[flat] = u16::MAX;
                continue;
            }

            let center_sdf = SdfScene::value(&nodes, &brick_aabb.center());
            if center_sdf.abs() > half_diag * 2.0 {
                let old = data.page_table[flat];
                if old != u16::MAX { data.free_slot(old); }
                data.page_table[flat] = u16::MAX;
                continue;
            }

            // Atlas slot allocation: reuse existing or alloc new (with recycling)
            let brick_idx = if data.page_table[flat] == u16::MAX {
                match data.alloc_slot(self.max_atlas_bricks) {
                    Some(idx) => {
                        data.page_table[flat] = idx;
                        idx as usize
                    }
                    None => continue, // atlas full
                }
            } else {
                data.page_table[flat] as usize
            };

            let atlas_offset = (lev * MAX_BRICKS_PER_LEVEL + brick_idx) * BRICK_DWORDS * 4;
            jobs.push(GpuBrickJob {
                brick_world_min: world_min,
                voxel_size:      self.voxel_size,
                atlas_offset:    atlas_offset as u32,
                half_diag,
                _pad:            [0; 2],
            });
            state.last_baked += 1;
        }

        state.pending_count = state.pending.len();
        jobs
    }

    /// Bake every brick in the grid in parallel. Blocks until complete.
    pub fn bake_all(&self, sdf: &SdfScene) -> BrickmapData {
        let [gx, gy, gz] = self.grid_dims;
        let total = (gx * gy * gz) as usize;
        let half_diag = self.brick_half_diag();

        let results: Vec<Option<(Vec<u16>, Vec<u8>)>> = (0..total)
            .into_par_iter()
            .map(|flat| {
                let bx = (flat as u32) % gx;
                let by = ((flat as u32) / gx) % gy;
                let bz = (flat as u32) / (gx * gy);
                self.bake_one(bx, by, bz, sdf, half_diag)
            })
            .collect();

        let mut page_table = vec![u16::MAX; total];
        let mut sdf_atlas: Vec<u16> = Vec::new();
        let mut mat_atlas: Vec<u8> = Vec::new();
        let mut brick_count = 0u32;

        for (flat, result) in results.into_iter().enumerate() {
            if let Some((sdf_data, mat_data)) = result {
                page_table[flat] = brick_count as u16;
                sdf_atlas.extend_from_slice(&sdf_data);
                mat_atlas.extend_from_slice(&mat_data);
                brick_count += 1;
            }
        }

        BrickmapData {
            page_table,
            sdf_atlas,
            mat_atlas,
            brick_count,
            free_list: Vec::new(),
            grid_dims: self.grid_dims,
            voxel_size: self.voxel_size,
            world_origin: self.world_origin,
            half_diag,
        }
    }

    /// Bake up to `budget` bricks from `state.pending`, updating `data` in-place.
    pub fn bake_batch(&self, sdf: &SdfScene, data: &mut BrickmapData, state: &mut BakeState, budget: usize) {
        let half_diag = self.brick_half_diag();
        let count = budget.min(state.pending.len());
        let batch: Vec<[u32; 3]> = state.pending.drain(..count).collect();
        state.last_baked = 0;

        for [bx, by, bz] in batch {
            let flat = self.page_idx(bx, by, bz);
            match self.bake_one(bx, by, bz, sdf, half_diag) {
                Some((sdf_data, mat_data)) => {
                    let brick_idx = if data.page_table[flat] == u16::MAX {
                        // New allocation — respect atlas capacity
                        if data.brick_count >= self.max_atlas_bricks {
                            continue;
                        }
                        let idx = data.brick_count as u16;
                        data.page_table[flat] = idx;
                        data.brick_count += 1;
                        data.sdf_atlas.extend(std::iter::repeat(0u16).take(BRICK_VOXELS));
                        data.mat_atlas.extend(std::iter::repeat(0u8).take(BRICK_VOXELS));
                        idx as usize
                    } else {
                        // Re-bake of existing brick — always allowed
                        data.page_table[flat] as usize
                    };
                    let base = brick_idx * BRICK_VOXELS;
                    data.sdf_atlas[base..base + BRICK_VOXELS].copy_from_slice(&sdf_data);
                    data.mat_atlas[base..base + BRICK_VOXELS].copy_from_slice(&mat_data);
                    state.last_baked += 1;
                }
                None => {
                    // Brick became empty (e.g. after an edit) — mark as unallocated
                    data.page_table[flat] = u16::MAX;
                }
            }
        }

        state.pending_count = state.pending.len();
    }
}

// ─── BrickmapData ─────────────────────────────────────────────────────────────

pub struct BrickmapData {
    /// Flat brick index (0xFFFF = empty). Index = bx + gx*(by + gy*bz).
    /// With toroidal addressing, cell (cx,cy,cz) stores data for the world-brick
    /// whose index % grid_dims == (cx,cy,cz) and that is within the current view.
    pub page_table:  Vec<u16>,
    /// f16 SDF samples (CPU bake path only). GPU bake writes to the GPU atlas directly.
    pub sdf_atlas:   Vec<u16>,
    pub mat_atlas:   Vec<u8>,
    pub brick_count: u32,
    /// Recycled atlas slots freed when bricks leave the clipmap view.
    pub free_list:   Vec<u16>,
    pub grid_dims:   [u32; 3],
    pub voxel_size:  f32,
    pub world_origin: [f32; 3],
    pub half_diag:   f32,
}

impl BrickmapData {
    pub fn empty(grid_dims: [u32; 3], voxel_size: f32, world_origin: [f32; 3]) -> Self {
        let [gx, gy, gz] = grid_dims;
        let bws = BRICK_SIZE as f32 * voxel_size;
        let half_diag = bws * (3.0_f32).sqrt() * 0.5;
        Self {
            page_table:  vec![u16::MAX; (gx * gy * gz) as usize],
            sdf_atlas:   Vec::new(),
            mat_atlas:   Vec::new(),
            brick_count: 0,
            free_list:   Vec::new(),
            grid_dims,
            voxel_size,
            world_origin,
            half_diag,
        }
    }

    /// Allocate an atlas slot, recycling freed slots first.
    pub fn alloc_slot(&mut self, max: u32) -> Option<u16> {
        if let Some(slot) = self.free_list.pop() {
            return Some(slot);
        }
        if self.brick_count < max {
            let slot = self.brick_count as u16;
            self.brick_count += 1;
            Some(slot)
        } else {
            None
        }
    }

    /// Return an atlas slot to the free list.
    pub fn free_slot(&mut self, slot: u16) {
        self.free_list.push(slot);
    }

    /// Flat page-table index for brick `(bx, by, bz)`.
    pub fn page_idx(&self, bx: u32, by: u32, bz: u32) -> usize {
        let [gx, gy, _] = self.grid_dims;
        (bx + gx * (by + gy * bz)) as usize
    }

    /// World-space size of one brick.
    pub fn brick_world_size(&self) -> f32 {
        BRICK_SIZE as f32 * self.voxel_size
    }
}
