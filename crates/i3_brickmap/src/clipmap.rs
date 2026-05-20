use crate::BRICK_SIZE;

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

// ─── GPU-uploadable structs ────────────────────────────────────────────────────

/// Per-level camera state uploaded each frame to the GPU dirty pass.
/// Layout: 64 bytes, 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LevelState {
    pub old_origin: [f32; 3],
    pub voxel_size: f32,
    pub new_origin: [f32; 3],
    /// 1 = the camera jumped farther than one full grid — wipe the entire level.
    pub giant_jump: u32,
    pub old_ob:     [i32; 3],
    pub _pad0:      u32,
    pub new_ob:     [i32; 3],
    pub _pad1:      u32,
}

/// Sphere region that was edited this frame — GPU dirty pass marks every brick
/// that overlaps the sphere as needing re-bake.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InvalidationSphere {
    pub center: [f32; 3],
    pub radius: f32,
}

// ─── CPU state ────────────────────────────────────────────────────────────────

pub struct ClipmapLevel {
    pub voxel_size:  f32,
    pub world_origin: [f32; 3],
    pub origin_brick: [i32; 3],
    prev_origin: [f32; 3],
    prev_ob:     [i32; 3],
    giant_jump:  bool,
}

pub struct BrickmapClipmapState {
    pub levels: Vec<ClipmapLevel>,
    pub invalidation_spheres: Vec<InvalidationSphere>,
}

impl BrickmapClipmapState {
    pub fn new(camera_pos: [f32; 3]) -> Self {
        let levels = (0..NUM_LEVELS).map(|lev| {
            let voxel_size = LEVEL_VOXEL_SIZES[lev];
            let world_origin = snap_origin(camera_pos, voxel_size, CLIPMAP_GRID);
            let origin_brick = world_to_origin_brick(world_origin, voxel_size);
            ClipmapLevel {
                voxel_size,
                world_origin,
                origin_brick,
                prev_origin: world_origin,
                prev_ob:     origin_brick,
                giant_jump:  true,   // first frame: bake everything
            }
        }).collect();
        Self { levels, invalidation_spheres: Vec::new() }
    }

    /// Update per-level origins when the camera moves.  Call once per frame before
    /// `collect_frame_upload`.  Only tracks the logical shift — the GPU dirty pass
    /// does the actual brick invalidation.
    pub fn update_camera(&mut self, camera_pos: [f32; 3]) {
        let [gx, gy, gz] = CLIPMAP_GRID.map(|v| v as i32);
        for lev in &mut self.levels {
            let bws = BRICK_SIZE as f32 * lev.voxel_size;
            let new_origin = snap_origin(camera_pos, lev.voxel_size, CLIPMAP_GRID);
            let new_ob = world_to_origin_brick(new_origin, lev.voxel_size);

            let dx = new_ob[0] - lev.origin_brick[0];
            let dy = new_ob[1] - lev.origin_brick[1];
            let dz = new_ob[2] - lev.origin_brick[2];

            if dx == 0 && dy == 0 && dz == 0 {
                continue;
            }

            lev.prev_origin = lev.world_origin;
            lev.prev_ob     = lev.origin_brick;
            lev.world_origin = new_origin;
            lev.origin_brick = new_ob;
            lev.giant_jump  = dx.abs() >= gx || dy.abs() >= gy || dz.abs() >= gz;

            let _ = bws; // used implicitly above
        }
    }

    /// Accumulate a sphere-shaped invalidation region (e.g. after an SDF edit).
    /// The GPU dirty pass reads these each frame and marks all overlapping bricks.
    pub fn invalidate_sphere(&mut self, center: [f32; 3], radius: f32) {
        self.invalidation_spheres.push(InvalidationSphere { center, radius });
    }

    /// Called once per frame by the compute bake pass to collect GPU upload data.
    /// Returns the current `LevelState` array and drains the invalidation sphere list.
    /// Commits new origins as the "old" origins for the next frame.
    pub fn collect_frame_upload(&mut self) -> (Vec<LevelState>, Vec<InvalidationSphere>) {
        let level_states: Vec<LevelState> = self.levels.iter_mut().map(|lev| {
            let ls = LevelState {
                old_origin: lev.prev_origin,
                voxel_size: lev.voxel_size,
                new_origin: lev.world_origin,
                giant_jump: lev.giant_jump as u32,
                old_ob:     lev.prev_ob,
                _pad0:      0,
                new_ob:     lev.origin_brick,
                _pad1:      0,
            };
            lev.prev_origin = lev.world_origin;
            lev.prev_ob     = lev.origin_brick;
            lev.giant_jump  = false;
            ls
        }).collect();
        let spheres = std::mem::take(&mut self.invalidation_spheres);
        (level_states, spheres)
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
