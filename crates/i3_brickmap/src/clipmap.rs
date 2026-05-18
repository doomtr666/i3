use i3_voxel::SdfScene;

use crate::gpu_scene::{GpuBrickJob, GpuPrimitive, pack_scene};
use crate::{BRICK_SIZE, BakeState, BrickmapBaker, BrickmapData};

pub const NUM_LEVELS: usize = 10;
pub const CLIPMAP_GRID: [u32; 3] = [32, 32, 32];
/// Voxel sizes (m) — each doubles the previous.
/// Coverage per level = GRID * BRICK_SIZE * voxel_size = 256 * vs.
/// L0  0.0125 →  3.2 m   (finer near detail)
/// L1  0.025  →  6.4 m
/// L2  0.05   → 12.8 m
/// L3  0.10   → 25.6 m
/// L4  0.20   → 51.2 m
/// L5  0.40   → 102.4 m
/// L6  0.80   → 204.8 m
/// L7  1.60   → 409.6 m
/// L8  4.00   → 1024  m
/// L9  8.00   → 2048  m  (≥ 2 km)
pub const LEVEL_VOXEL_SIZES: [f32; NUM_LEVELS] =
    [0.0125, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 4.00, 8.00];
pub const MAX_BRICKS_PER_LEVEL: usize = 8192;
pub const GRID_VOL: usize = (CLIPMAP_GRID[0] * CLIPMAP_GRID[1] * CLIPMAP_GRID[2]) as usize;

pub struct ClipmapLevel {
    pub data: BrickmapData,
    pub bake_state: BakeState,
    /// World-brick coordinates of the grid corner (integer, can be negative).
    /// Cell (cx,cy,cz) represents world-brick (ob + offset) where
    /// offset = (cell - ob%grid + grid) % grid ∈ [0, grid).
    pub origin_brick: [i32; 3],
}

pub struct BrickmapClipmapState {
    pub levels: Vec<ClipmapLevel>,
}

impl BrickmapClipmapState {
    pub fn new(camera_pos: [f32; 3]) -> Self {
        let mut levels = Vec::with_capacity(NUM_LEVELS);
        for lev in 0..NUM_LEVELS {
            let voxel_size = LEVEL_VOXEL_SIZES[lev];
            let origin = snap_origin(camera_pos, voxel_size, CLIPMAP_GRID);
            let origin_brick = world_to_origin_brick(origin, voxel_size);
            let data = BrickmapData::empty(CLIPMAP_GRID, voxel_size, origin);
            let mut bake_state = BakeState::new();
            bake_state.enqueue_all(CLIPMAP_GRID);
            levels.push(ClipmapLevel {
                data,
                bake_state,
                origin_brick,
            });
        }
        Self { levels }
    }

    /// Toroidal camera update.
    ///
    /// When the camera moves, the clipmap origin shifts by whole bricks.
    /// Only bricks that "enter" the new view (the boundary slab) are invalidated;
    /// all other bricks remain stable in world space — no full re-bake.
    pub fn update_camera(&mut self, camera_pos: [f32; 3]) {
        for lev in 0..NUM_LEVELS {
            let voxel_size = LEVEL_VOXEL_SIZES[lev];
            let bws = BRICK_SIZE as f32 * voxel_size;
            let new_origin = snap_origin(camera_pos, voxel_size, CLIPMAP_GRID);
            let old_origin = self.levels[lev].data.world_origin;

            let dx = ((new_origin[0] - old_origin[0]) / bws).round() as i32;
            let dy = ((new_origin[1] - old_origin[1]) / bws).round() as i32;
            let dz = ((new_origin[2] - old_origin[2]) / bws).round() as i32;
            if dx == 0 && dy == 0 && dz == 0 {
                continue;
            }

            let [gx, gy, gz] = CLIPMAP_GRID.map(|v| v as i32);
            let [gx_u, gy_u, gz_u] = CLIPMAP_GRID;

            let old_ob = self.levels[lev].origin_brick;
            let new_ob = [old_ob[0] + dx, old_ob[1] + dy, old_ob[2] + dz];

            self.levels[lev].data.world_origin = new_origin;
            self.levels[lev].origin_brick = new_ob;

            // Giant jump: wipe everything and re-enqueue all cells
            if dx.abs() >= gx || dy.abs() >= gy || dz.abs() >= gz {
                for v in &mut self.levels[lev].data.page_table {
                    *v = u16::MAX;
                }
                self.levels[lev].data.free_list.clear();
                self.levels[lev].data.brick_count = 0;
                self.levels[lev].bake_state = BakeState::new();
                self.levels[lev].bake_state.enqueue_all(CLIPMAP_GRID);
                continue;
            }

            // Toroidal incremental update:
            // A cell (cx,cy,cz) is invalidated when its world-brick LEAVES the new view.
            // For dx > 0: offsets 0..dx leave (old leading edge wraps to new trailing edge).
            // For dx < 0: offsets (gx+dx)..gx leave.
            let ob_x_mod = ((old_ob[0] % gx) + gx) % gx;
            let ob_y_mod = ((old_ob[1] % gy) + gy) % gy;
            let ob_z_mod = ((old_ob[2] % gz) + gz) % gz;

            let level = &mut self.levels[lev];
            for cz in 0..gz_u {
                for cy in 0..gy_u {
                    for cx in 0..gx_u {
                        let off_x = (cx as i32 - ob_x_mod + gx) % gx;
                        let off_y = (cy as i32 - ob_y_mod + gy) % gy;
                        let off_z = (cz as i32 - ob_z_mod + gz) % gz;

                        let leave = (dx > 0 && off_x < dx)
                            || (dx < 0 && off_x >= gx + dx)
                            || (dy > 0 && off_y < dy)
                            || (dy < 0 && off_y >= gy + dy)
                            || (dz > 0 && off_z < dz)
                            || (dz < 0 && off_z >= gz + dz);

                        if leave {
                            let flat = (cx + gx_u * (cy + gy_u * cz)) as usize;
                            let old = level.data.page_table[flat];
                            if old != u16::MAX {
                                level.data.free_slot(old);
                            }
                            level.data.page_table[flat] = u16::MAX;
                            level.bake_state.enqueue(cx, cy, cz);
                        }
                    }
                }
            }
        }
    }

    /// Invalidate all bricks within `radius` metres of `center` (after an SDF edit).
    pub fn invalidate_sphere(&mut self, center: [f32; 3], radius: f32) {
        for lev in 0..NUM_LEVELS {
            let voxel_size = LEVEL_VOXEL_SIZES[lev];
            let bws = BRICK_SIZE as f32 * voxel_size;
            let r = radius + voxel_size * 4.0;
            let ob = self.levels[lev].origin_brick;
            let [gx, gy, gz] = CLIPMAP_GRID.map(|v| v as i32);

            // World-brick range of the sphere, clamped to the current view
            let bwx0 = ((center[0] - r) / bws).floor() as i32;
            let bwy0 = ((center[1] - r) / bws).floor() as i32;
            let bwz0 = ((center[2] - r) / bws).floor() as i32;
            let bwx1 = ((center[0] + r) / bws).ceil() as i32;
            let bwy1 = ((center[1] + r) / bws).ceil() as i32;
            let bwz1 = ((center[2] + r) / bws).ceil() as i32;

            let bwx0 = bwx0.max(ob[0]);
            let bwy0 = bwy0.max(ob[1]);
            let bwz0 = bwz0.max(ob[2]);
            let bwx1 = bwx1.min(ob[0] + gx);
            let bwy1 = bwy1.min(ob[1] + gy);
            let bwz1 = bwz1.min(ob[2] + gz);

            for bwz in bwz0..bwz1 {
                for bwy in bwy0..bwy1 {
                    for bwx in bwx0..bwx1 {
                        let cx = ((bwx % gx) + gx) % gx;
                        let cy = ((bwy % gy) + gy) % gy;
                        let cz = ((bwz % gz) + gz) % gz;
                        let flat = (cx + gx * (cy + gy * cz)) as usize;
                        let old = self.levels[lev].data.page_table[flat];
                        if old != u16::MAX {
                            self.levels[lev].data.free_slot(old);
                        }
                        self.levels[lev].data.page_table[flat] = u16::MAX;
                        self.levels[lev]
                            .bake_state
                            .enqueue_priority(cx as u32, cy as u32, cz as u32);
                    }
                }
            }
        }
    }

    /// CPU prep for GPU baking: BVH query + centre-SDF cull + atlas allocation.
    /// Returns (jobs, primitives) for the compute dispatch.
    pub fn prepare_gpu_bake(
        &mut self,
        scene: &SdfScene,
        budget: usize,
    ) -> (Vec<GpuBrickJob>, Vec<GpuPrimitive>) {
        let mut all_jobs = Vec::new();

        for lev in 0..NUM_LEVELS {
            if self.levels[lev].bake_state.pending.is_empty() {
                continue;
            }

            let voxel_size = LEVEL_VOXEL_SIZES[lev];
            let world_origin = self.levels[lev].data.world_origin;
            let origin_brick = self.levels[lev].origin_brick;
            let baker = BrickmapBaker {
                grid_dims: CLIPMAP_GRID,
                world_origin,
                voxel_size,
                max_atlas_bricks: MAX_BRICKS_PER_LEVEL as u32,
            };
            let level = &mut self.levels[lev];
            let jobs = baker.prepare_batch_gpu(
                scene,
                &mut level.data,
                &mut level.bake_state,
                budget,
                lev,
                origin_brick,
            );
            all_jobs.extend(jobs);
        }

        let prims = pack_scene(scene);
        (all_jobs, prims)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn snap_origin(camera: [f32; 3], voxel_size: f32, grid_dims: [u32; 3]) -> [f32; 3] {
    let bws = BRICK_SIZE as f32 * voxel_size;
    let half = [
        grid_dims[0] as f32 * bws * 0.5,
        grid_dims[1] as f32 * bws * 0.5,
        grid_dims[2] as f32 * bws * 0.5,
    ];
    [
        (camera[0] / bws).round() * bws - half[0],
        (camera[1] / bws).round() * bws - half[1],
        (camera[2] / bws).round() * bws - half[2],
    ]
}

fn world_to_origin_brick(world_origin: [f32; 3], voxel_size: f32) -> [i32; 3] {
    let bws = BRICK_SIZE as f32 * voxel_size;
    [
        (world_origin[0] / bws).round() as i32,
        (world_origin[1] / bws).round() as i32,
        (world_origin[2] / bws).round() as i32,
    ]
}
