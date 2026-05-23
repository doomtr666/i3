use crate::sdf::SdfScene;
use crate::voxel::{VoxelVertex, VOXEL_BLOCK_WIDTH, VOXEL_DIST};
use i3_math::AABB;
use nalgebra::{Point3, Matrix4};
use std::sync::Arc;
use rayon::prelude::*;

pub const PENDING_MESH_ID: u32 = u32::MAX - 1;
pub const PENDING_OBJECT_ID: u64 = u64::MAX - 1;

#[derive(Clone, Debug)]
pub struct GenerationTask {
    pub root_index: usize,
    pub depth: u32,
    pub path_bits: u32,
    pub center: Point3<f32>,
    pub half_size: f32,
    pub score: f32,
}

/// Implemented by the application to relay GPU mesh add/remove calls from the octree.
pub trait VoxelSceneSink {
    fn add_mesh(
        &mut self,
        vertices: &[VoxelVertex],
        indices: &[u32],
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
    ) -> (u32, u64);

    fn remove_mesh(&mut self, mesh_id: u32, object_id: u64);
}

// ─── NodeState ────────────────────────────────────────────────────────────────

enum NodeState {
    /// Not yet generated. Will produce a Leaf on the next budget slot.
    Ungenerated,
    /// Active leaf with a GPU mesh.
    /// mesh_id/object_id == u32::MAX/u64::MAX → empty block (no GPU upload).
    Leaf { mesh_id: u32, object_id: u64 },
    /// Coarse mesh stays visible while 8 children finish generating.
    /// Transitions to Split once all children are no longer Ungenerated.
    Splitting { mesh_id: u32, object_id: u64, children: Box<[OctreeNode; 8]> },
    /// All children are active; this node has no mesh of its own.
    Split { children: Box<[OctreeNode; 8]> },
    /// Children stay visible while the coarse mesh regenerates.
    /// Once the coarse mesh is ready, children are removed and we become Leaf.
    Merging { children: Box<[OctreeNode; 8]>, coarse: Option<(u32, u64)> },
}

// ─── OctreeNode ───────────────────────────────────────────────────────────────

struct OctreeNode {
    center: Point3<f32>,
    half_size: f32,
    depth: u32,
    state: NodeState,
}

impl OctreeNode {
    fn new(center: Point3<f32>, half_size: f32, depth: u32) -> Self {
        OctreeNode { center, half_size, depth, state: NodeState::Ungenerated }
    }

    fn distance_to_camera(&self, camera: &Point3<f32>) -> f32 {
        let h = self.half_size;
        let c = self.center;
        let min = Point3::new(c.x - h, c.y - h, c.z - h);
        let max = Point3::new(c.x + h, c.y + h, c.z + h);
        let aabb = AABB::new(min, max);
        let clamped = aabb.clamp(camera);
        (camera - clamped).norm()
    }

    fn compute_score(&self, camera: &Point3<f32>, vp: &Matrix4<f32>) -> f32 {
        let dist = self.distance_to_camera(camera);
        let h = self.half_size;
        let c = self.center;
        let aabb = AABB::new(
            Point3::new(c.x - h, c.y - h, c.z - h),
            Point3::new(c.x + h, c.y + h, c.z + h),
        );
        if aabb.is_in_frustum(vp) {
            dist
        } else {
            dist + 10000.0 // heavily penalize outside frustum
        }
    }

    fn needs_split(&self, camera: &Point3<f32>, max_depth: u32, split_factor: f32) -> bool {
        self.depth < max_depth
            && self.distance_to_camera(camera) < self.half_size * split_factor
    }

    fn should_merge(&self, camera: &Point3<f32>, split_factor: f32, merge_hysteresis: f32) -> bool {
        self.distance_to_camera(camera) > self.half_size * split_factor * merge_hysteresis
    }

    fn child_nodes(&self) -> [OctreeNode; 8] {
        let h = self.half_size * 0.5;
        let c = self.center;
        [
            OctreeNode::new(Point3::new(c.x - h, c.y - h, c.z - h), h, self.depth + 1),
            OctreeNode::new(Point3::new(c.x + h, c.y - h, c.z - h), h, self.depth + 1),
            OctreeNode::new(Point3::new(c.x - h, c.y + h, c.z - h), h, self.depth + 1),
            OctreeNode::new(Point3::new(c.x + h, c.y + h, c.z - h), h, self.depth + 1),
            OctreeNode::new(Point3::new(c.x - h, c.y - h, c.z + h), h, self.depth + 1),
            OctreeNode::new(Point3::new(c.x + h, c.y - h, c.z + h), h, self.depth + 1),
            OctreeNode::new(Point3::new(c.x - h, c.y + h, c.z + h), h, self.depth + 1),
            OctreeNode::new(Point3::new(c.x + h, c.y + h, c.z + h), h, self.depth + 1),
        ]
    }



    fn remove_leaf<S: VoxelSceneSink>(mesh_id: u32, object_id: u64, sink: &mut S) {
        if mesh_id != u32::MAX && mesh_id != PENDING_MESH_ID {
            sink.remove_mesh(mesh_id, object_id);
        }
    }

    fn remove_all<S: VoxelSceneSink>(&mut self, sink: &mut S) {
        let old = std::mem::replace(&mut self.state, NodeState::Ungenerated);
        match old {
            NodeState::Leaf { mesh_id, object_id } => {
                Self::remove_leaf(mesh_id, object_id, sink);
            }
            NodeState::Splitting { mesh_id, object_id, mut children } => {
                Self::remove_leaf(mesh_id, object_id, sink);
                for child in children.iter_mut() {
                    child.remove_all(sink);
                }
            }
            NodeState::Split { mut children } => {
                for child in children.iter_mut() {
                    child.remove_all(sink);
                }
            }
            NodeState::Merging { mut children, coarse } => {
                if let Some((mesh_id, object_id)) = coarse {
                    Self::remove_leaf(mesh_id, object_id, sink);
                }
                for child in children.iter_mut() {
                    child.remove_all(sink);
                }
            }
            NodeState::Ungenerated => {}
        }
    }

    /// Returns true when the subtree has finished generating (no Ungenerated node).
    fn is_generated(&self) -> bool {
        match self.state {
            NodeState::Ungenerated => false,
            NodeState::Leaf { mesh_id, .. } => mesh_id != PENDING_MESH_ID,
            _ => true,
        }
    }

    fn gather<S: VoxelSceneSink>(
        &mut self,
        camera: &Point3<f32>,
        vp: &Matrix4<f32>,
        sdf: &SdfScene,
        sink: &mut S,
        max_depth: u32,
        split_factor: f32,
        merge_hysteresis: f32,
        can_split: bool,
        root_index: usize,
        depth: u32,
        path_bits: u32,
        tasks: &mut Vec<GenerationTask>,
    ) {
        let old = std::mem::replace(&mut self.state, NodeState::Ungenerated);

        match old {
            // ── Generate at current LOD ──────────────────────────────────────
            NodeState::Ungenerated => {
                let score = self.compute_score(camera, vp);
                tasks.push(GenerationTask {
                    root_index, depth, path_bits,
                    center: self.center, half_size: self.half_size, score,
                });
                self.state = NodeState::Leaf { mesh_id: PENDING_MESH_ID, object_id: PENDING_OBJECT_ID };
            }

            // ── Decide whether to start splitting ───────────────────────────
            NodeState::Leaf { mesh_id, object_id } => {
                if mesh_id == PENDING_MESH_ID {
                    // still pending, just re-queue
                    let score = self.compute_score(camera, vp);
                    tasks.push(GenerationTask {
                        root_index, depth, path_bits,
                        center: self.center, half_size: self.half_size, score,
                    });
                    self.state = NodeState::Leaf { mesh_id, object_id };
                } else if mesh_id != u32::MAX && can_split && self.needs_split(camera, max_depth, split_factor) {
                    // Keep coarse mesh visible; children fill in over coming frames.
                    self.state = NodeState::Splitting {
                        mesh_id,
                        object_id,
                        children: Box::new(self.child_nodes()),
                    };
                    // Start filling children immediately, but block their own splits
                    // (can_split=false) to avoid cascading multi-resolution overlaps.
                    if let NodeState::Splitting { children, .. } = &mut self.state {
                        for (i, child) in children.iter_mut().enumerate() {
                            child.gather(camera, vp, sdf, sink, max_depth, split_factor,
                                merge_hysteresis, false, root_index, depth + 1, path_bits | ((i as u32) << (depth * 3)), tasks);
                        }
                    }
                } else {
                    self.state = NodeState::Leaf { mesh_id, object_id };
                }
            }

            // ── Wait for all children, then drop coarse mesh ─────────────────
            NodeState::Splitting { mesh_id, object_id, mut children } => {
                if self.should_merge(camera, split_factor, merge_hysteresis) {
                    // Abort split: clean up children and revert to Leaf
                    for child in children.iter_mut() {
                        child.remove_all(sink);
                    }
                    self.state = NodeState::Leaf { mesh_id, object_id };
                } else {
                    // Children may only generate their Leaf, not sub-split (can_split=false).
                    for (i, child) in children.iter_mut().enumerate() {
                        child.gather(camera, vp, sdf, sink, max_depth, split_factor,
                            merge_hysteresis, false, root_index, depth + 1, path_bits | ((i as u32) << (depth * 3)), tasks);
                    }

                    if children.iter().all(|c| c.is_generated()) {
                        // All children ready — drop the coarse mesh.
                        Self::remove_leaf(mesh_id, object_id, sink);
                        self.state = NodeState::Split { children };
                    } else {
                        self.state = NodeState::Splitting { mesh_id, object_id, children };
                    }
                }
            }

            // ── Update children; decide whether to start merging ─────────────
            NodeState::Split { mut children } => {
                if self.should_merge(camera, split_factor, merge_hysteresis) {
                    let score = self.compute_score(camera, vp);
                    tasks.push(GenerationTask {
                        root_index, depth, path_bits,
                        center: self.center, half_size: self.half_size, score,
                    });
                    self.state = NodeState::Merging { children, coarse: Some((PENDING_MESH_ID, PENDING_OBJECT_ID)) };
                    self.try_finish_merge(sink);
                } else {
                    for (i, child) in children.iter_mut().enumerate() {
                        child.gather(camera, vp, sdf, sink, max_depth, split_factor,
                            merge_hysteresis, true, root_index, depth + 1, path_bits | ((i as u32) << (depth * 3)), tasks);
                    }
                    self.state = NodeState::Split { children };
                }
            }

            // ── Finish merging once coarse mesh is available ─────────────────
            NodeState::Merging { mut children, coarse } => {
                let coarse_is_empty = matches!(coarse, Some((m, _)) if m == u32::MAX);
                if !coarse_is_empty && can_split && self.needs_split(camera, max_depth, split_factor) {
                    // Abort merge: keep children, drop coarse if generated, revert to Split
                    if let Some((mesh_id, object_id)) = coarse {
                        Self::remove_leaf(mesh_id, object_id, sink);
                    }
                    // Re-gather children because we aborted the merge
                    for (i, child) in children.iter_mut().enumerate() {
                        child.gather(camera, vp, sdf, sink, max_depth, split_factor,
                            merge_hysteresis, true, root_index, depth + 1, path_bits | ((i as u32) << (depth * 3)), tasks);
                    }
                    self.state = NodeState::Split { children };
                } else {
                    if let Some((mesh_id, _object_id)) = coarse {
                        if mesh_id == PENDING_MESH_ID {
                            let score = self.compute_score(camera, vp);
                            tasks.push(GenerationTask {
                                root_index, depth, path_bits,
                                center: self.center, half_size: self.half_size, score,
                            });
                        }
                    }
                    self.state = NodeState::Merging { children, coarse };
                    self.try_finish_merge(sink);
                }
            }
        }
    }

    /// If coarse mesh is ready, removes all children and collapses to Leaf.
    fn try_finish_merge<S: VoxelSceneSink>(&mut self, sink: &mut S) {
        if let NodeState::Merging { coarse: Some((mesh_id, _object_id)), .. } = &self.state {
            if *mesh_id != PENDING_MESH_ID {
                let old = std::mem::replace(&mut self.state, NodeState::Ungenerated);
                if let NodeState::Merging { mut children, coarse: Some((m, o)) } = old {
                    for child in children.iter_mut() {
                        child.remove_all(sink);
                    }
                    self.state = NodeState::Leaf { mesh_id: m, object_id: o };
                }
            }
        }
    }
}

// ─── VoxelOctree ──────────────────────────────────────────────────────────────

pub struct VoxelOctree {
    roots: Vec<OctreeNode>,
    sdf: Arc<SdfScene>,
    pub max_depth: u32,
    pub split_factor: f32,
    pub merge_hysteresis: f32,
}

impl VoxelOctree {
    /// Creates a rectangular grid of `grid[0] × grid[1] × grid[2]` root nodes.
    ///
    /// Using a flat grid (e.g. `[16, 2, 16]`) avoids wasting root nodes on
    /// empty Y columns above and below the terrain.
    ///
    /// At `max_depth` the voxel resolution matches `VOXEL_DIST` (0.05 m).
    /// Each shallower level doubles the voxel size while keeping the voxel count.
    pub fn new(
        sdf: Arc<SdfScene>,
        world_origin: Point3<f32>,
        grid: [u32; 3],
        max_depth: u32,
        split_factor: f32,
        merge_hysteresis: f32,
    ) -> Self {
        let root_half_size =
            VOXEL_BLOCK_WIDTH as f32 * VOXEL_DIST * (1u32 << max_depth) as f32 * 0.5;
        let block_size = root_half_size * 2.0;
        let [gx_count, gy_count, gz_count] = grid;

        let cap = (gx_count * gy_count * gz_count) as usize;
        let mut roots = Vec::with_capacity(cap);
        for gz in 0..gz_count {
            for gy in 0..gy_count {
                for gx in 0..gx_count {
                    let center = Point3::new(
                        world_origin.x + (gx as f32 + 0.5) * block_size,
                        world_origin.y + (gy as f32 + 0.5) * block_size,
                        world_origin.z + (gz as f32 + 0.5) * block_size,
                    );
                    roots.push(OctreeNode::new(center, root_half_size, 0));
                }
            }
        }
        VoxelOctree { roots, sdf, max_depth, split_factor, merge_hysteresis }
    }

    /// Per-frame update. Generates at most `budget` new meshes in parallel.
    pub fn update<S: VoxelSceneSink>(
        &mut self,
        camera: Point3<f32>,
        vp: &Matrix4<f32>,
        sink: &mut S,
        budget: usize,
    ) {
        let sdf = Arc::clone(&self.sdf);
        let max_depth = self.max_depth;
        let split_factor = self.split_factor;
        let merge_hysteresis = self.merge_hysteresis;

        // 1. Gather pass
        let mut tasks = Vec::new();
        for (i, root) in self.roots.iter_mut().enumerate() {
            root.gather(&camera, vp, &sdf, sink, max_depth, split_factor, merge_hysteresis, true, i, 0, 0, &mut tasks);
        }

        if tasks.is_empty() {
            return;
        }

        // 2. Sort and cull tasks
        tasks.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        tasks.truncate(budget);

        // 3. Parallel generation
        let results: Vec<(GenerationTask, Vec<VoxelVertex>, Vec<u32>)> = tasks
            .into_par_iter()
            .map(|task| {
                let mut verts = Vec::new();
                let mut inds = Vec::new();
                let voxel_dist = task.half_size * 2.0 / VOXEL_BLOCK_WIDTH as f32;
                let c = task.center;
                let h = task.half_size;
                let node_aabb = AABB::new(
                    Point3::new(c.x - h, c.y - h, c.z - h),
                    Point3::new(c.x + h, c.y + h, c.z + h),
                );
                crate::voxel::generate_mesh_from_sdf(
                    &sdf,
                    node_aabb.expand(2.0 * voxel_dist),
                    voxel_dist,
                    |x, y, z| {
                        Point3::new(
                            c.x - h + x as f32 * voxel_dist,
                            c.y - h + y as f32 * voxel_dist,
                            c.z - h + z as f32 * voxel_dist,
                        )
                    },
                    &mut verts,
                    &mut inds,
                );
                (task, verts, inds)
            })
            .collect();

        // 4. Upload and inject
        for (task, verts, inds) in results {
            let (mesh_id, object_id) = if verts.is_empty() || inds.is_empty() {
                (u32::MAX, u64::MAX)
            } else {
                let h = task.half_size;
                let c = task.center;
                sink.add_mesh(&verts, &inds, [c.x - h, c.y - h, c.z - h], [c.x + h, c.y + h, c.z + h])
            };

            // Traverse to the node
            let mut current = &mut self.roots[task.root_index];
            for d in 0..task.depth {
                let child_idx = (task.path_bits >> (d * 3)) & 0b111;
                current = match &mut current.state {
                    NodeState::Splitting { children, .. }
                    | NodeState::Split { children }
                    | NodeState::Merging { children, .. } => {
                        &mut children[child_idx as usize]
                    }
                    _ => unreachable!("Tree mutated during sync update pass?"),
                };
            }

            // Apply mesh
            match &mut current.state {
                NodeState::Leaf { mesh_id: m, object_id: o } if *m == PENDING_MESH_ID => {
                    *m = mesh_id;
                    *o = object_id;
                }
                NodeState::Merging { coarse: Some((m, o)), .. } if *m == PENDING_MESH_ID => {
                    *m = mesh_id;
                    *o = object_id;
                    current.try_finish_merge(sink);
                }
                _ => {
                    // Node state changed or was aborted (should not happen within same frame)
                    if mesh_id != u32::MAX {
                        sink.remove_mesh(mesh_id, object_id);
                    }
                }
            }
        }
    }

    /// Returns the AABB of every active leaf node (for debug draw).
    pub fn iter_node_aabbs(&self) -> Vec<AABB> {
        let mut out = Vec::new();
        for root in &self.roots {
            collect_aabbs(root, &mut out);
        }
        out
    }
}

fn collect_aabbs(node: &OctreeNode, out: &mut Vec<AABB>) {
    match &node.state {
        NodeState::Leaf { mesh_id, .. } | NodeState::Splitting { mesh_id, .. }
            if *mesh_id != u32::MAX && *mesh_id != PENDING_MESH_ID =>
        {
            let h = node.half_size;
            let c = node.center;
            out.push(AABB::new(
                Point3::new(c.x - h, c.y - h, c.z - h),
                Point3::new(c.x + h, c.y + h, c.z + h),
            ));
        }
        NodeState::Splitting { children, .. }
        | NodeState::Split { children }
        | NodeState::Merging { children, .. } => {
            for child in children.iter() {
                collect_aabbs(child, out);
            }
        }
        _ => {}
    }
}
