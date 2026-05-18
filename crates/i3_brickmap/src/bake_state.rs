use std::collections::VecDeque;

/// Tracks bricks pending bake for incremental / per-frame baking.
pub struct BakeState {
    /// Bricks queued for bake: (bx, by, bz) in brick-grid coordinates.
    pub pending: VecDeque<[u32; 3]>,
    /// Number of bricks baked in the last `bake_batch` call.
    pub last_baked: usize,
    /// Number of dirty bricks remaining.
    pub pending_count: usize,
}

impl BakeState {
    pub fn new() -> Self {
        Self { pending: VecDeque::new(), last_baked: 0, pending_count: 0 }
    }

    pub fn enqueue(&mut self, bx: u32, by: u32, bz: u32) {
        self.pending.push_back([bx, by, bz]);
        self.pending_count += 1;
    }

    /// Enqueue at the front of the queue (higher priority than pending bricks).
    pub fn enqueue_priority(&mut self, bx: u32, by: u32, bz: u32) {
        self.pending.push_front([bx, by, bz]);
        self.pending_count += 1;
    }

    /// Fill queue with every brick in the grid (for full initial bake).
    pub fn enqueue_all(&mut self, grid_dims: [u32; 3]) {
        let [gx, gy, gz] = grid_dims;
        self.pending.reserve((gx * gy * gz) as usize);
        for bz in 0..gz {
            for by in 0..gy {
                for bx in 0..gx {
                    self.pending.push_back([bx, by, bz]);
                }
            }
        }
        self.pending_count = self.pending.len();
    }
}

impl Default for BakeState {
    fn default() -> Self {
        Self::new()
    }
}
