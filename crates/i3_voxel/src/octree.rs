use crate::sdf::SdfScene;
use crate::voxel::{generate_mesh_from_sdf, VoxelVertex, VOXEL_BLOCK_WIDTH, VOXEL_DIST};
use i3_math::AABB;
use nalgebra::Point3;
use std::sync::Arc;

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

    fn needs_split(&self, camera: &Point3<f32>, max_depth: u32, split_factor: f32) -> bool {
        self.depth < max_depth
            && (self.center - camera).norm() < self.half_size * split_factor
    }

    fn should_merge(&self, camera: &Point3<f32>, split_factor: f32, merge_hysteresis: f32) -> bool {
        (self.center - camera).norm() > self.half_size * split_factor * merge_hysteresis
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

    fn generate_mesh(
        &self,
        sdf: &SdfScene,
        packed_vertices: &mut Vec<VoxelVertex>,
        packed_indices: &mut Vec<u32>,
    ) {
        let voxel_dist = self.half_size * 2.0 / VOXEL_BLOCK_WIDTH as f32;
        let center = self.center;
        let half_size = self.half_size;
        let node_aabb = AABB::new(
            Point3::new(center.x - half_size, center.y - half_size, center.z - half_size),
            Point3::new(center.x + half_size, center.y + half_size, center.z + half_size),
        );
        generate_mesh_from_sdf(
            sdf,
            node_aabb.expand(2.0 * voxel_dist),
            move |x, y, z| {
                Point3::new(
                    center.x - half_size + x as f32 * voxel_dist,
                    center.y - half_size + y as f32 * voxel_dist,
                    center.z - half_size + z as f32 * voxel_dist,
                )
            },
            packed_vertices,
            packed_indices,
        );
    }

    /// Upload a generated mesh to the sink. Returns (u32::MAX, u64::MAX) if empty.
    fn upload_mesh<S: VoxelSceneSink>(
        &self,
        sdf: &SdfScene,
        sink: &mut S,
    ) -> (u32, u64) {
        let mut verts = Vec::new();
        let mut inds = Vec::new();
        self.generate_mesh(sdf, &mut verts, &mut inds);
        if verts.is_empty() || inds.is_empty() {
            return (u32::MAX, u64::MAX);
        }
        let h = self.half_size;
        let c = self.center;
        sink.add_mesh(&verts, &inds, [c.x - h, c.y - h, c.z - h], [c.x + h, c.y + h, c.z + h])
    }

    fn remove_leaf<S: VoxelSceneSink>(mesh_id: u32, object_id: u64, sink: &mut S) {
        if mesh_id != u32::MAX {
            sink.remove_mesh(mesh_id, object_id);
        }
    }

    fn remove_all<S: VoxelSceneSink>(&mut self, sink: &mut S) {
        let old = std::mem::replace(&mut self.state, NodeState::Ungenerated);
        match old {
            NodeState::Leaf { mesh_id, object_id }
            | NodeState::Splitting { mesh_id, object_id, .. } => {
                Self::remove_leaf(mesh_id, object_id, sink);
            }
            NodeState::Split { mut children } | NodeState::Merging { mut children, .. } => {
                for child in children.iter_mut() {
                    child.remove_all(sink);
                }
            }
            NodeState::Ungenerated => {}
        }
    }

    /// Returns true when the subtree has finished generating (no Ungenerated node).
    fn is_generated(&self) -> bool {
        !matches!(self.state, NodeState::Ungenerated)
    }

    fn update<S: VoxelSceneSink>(
        &mut self,
        camera: &Point3<f32>,
        sdf: &SdfScene,
        sink: &mut S,
        max_depth: u32,
        split_factor: f32,
        merge_hysteresis: f32,
        budget: &mut usize,
        // false when called from a Splitting parent: children may only generate
        // their own Leaf mesh, not subdivide further. This prevents cascading
        // Splitting states that show 3+ resolutions of the same region at once.
        can_split: bool,
    ) {
        let old = std::mem::replace(&mut self.state, NodeState::Ungenerated);

        match old {
            // ── Generate at current LOD ──────────────────────────────────────
            NodeState::Ungenerated => {
                if *budget == 0 {
                    return; // stay Ungenerated
                }
                let (mesh_id, object_id) = self.upload_mesh(sdf, sink);
                *budget -= 1;
                self.state = NodeState::Leaf { mesh_id, object_id };
            }

            // ── Decide whether to start splitting ───────────────────────────
            NodeState::Leaf { mesh_id, object_id } => {
                if can_split && self.needs_split(camera, max_depth, split_factor) {
                    // Keep coarse mesh visible; children fill in over coming frames.
                    self.state = NodeState::Splitting {
                        mesh_id,
                        object_id,
                        children: Box::new(self.child_nodes()),
                    };
                    // Start filling children immediately, but block their own splits
                    // (can_split=false) to avoid cascading multi-resolution overlaps.
                    if let NodeState::Splitting { children, .. } = &mut self.state {
                        for child in children.iter_mut() {
                            if *budget == 0 { break; }
                            child.update(camera, sdf, sink, max_depth, split_factor,
                                merge_hysteresis, budget, false);
                        }
                    }
                } else {
                    self.state = NodeState::Leaf { mesh_id, object_id };
                }
            }

            // ── Wait for all children, then drop coarse mesh ─────────────────
            NodeState::Splitting { mesh_id, object_id, mut children } => {
                // Children may only generate their Leaf, not sub-split (can_split=false).
                for child in children.iter_mut() {
                    if *budget == 0 { break; }
                    child.update(camera, sdf, sink, max_depth, split_factor,
                        merge_hysteresis, budget, false);
                }

                if children.iter().all(|c| c.is_generated()) {
                    // All children ready — drop the coarse mesh.
                    Self::remove_leaf(mesh_id, object_id, sink);
                    self.state = NodeState::Split { children };
                } else {
                    self.state = NodeState::Splitting { mesh_id, object_id, children };
                }
            }

            // ── Update children; decide whether to start merging ─────────────
            NodeState::Split { mut children } => {
                if self.should_merge(camera, split_factor, merge_hysteresis) {
                    // self.state is Ungenerated here, so upload_mesh borrows cleanly.
                    let coarse = if *budget > 0 {
                        *budget -= 1;
                        Some(self.upload_mesh(sdf, sink))
                    } else {
                        None
                    };
                    self.state = NodeState::Merging { children, coarse };
                    self.try_finish_merge(sink);
                } else {
                    for child in children.iter_mut() {
                        if *budget == 0 { break; }
                        child.update(camera, sdf, sink, max_depth, split_factor,
                            merge_hysteresis, budget, true);
                    }
                    self.state = NodeState::Split { children };
                }
            }

            // ── Finish merging once coarse mesh is available ─────────────────
            NodeState::Merging { children, coarse: None } => {
                // self.state is Ungenerated here, so upload_mesh borrows cleanly.
                let coarse = if *budget > 0 {
                    *budget -= 1;
                    Some(self.upload_mesh(sdf, sink))
                } else {
                    None
                };
                self.state = NodeState::Merging { children, coarse };
                self.try_finish_merge(sink);
            }

            NodeState::Merging { children, coarse: Some(pair) } => {
                self.state = NodeState::Merging { children, coarse: Some(pair) };
                self.try_finish_merge(sink);
            }
        }
    }

    /// If coarse mesh is ready, removes all children and collapses to Leaf.
    fn try_finish_merge<S: VoxelSceneSink>(&mut self, sink: &mut S) {
        if let NodeState::Merging { coarse: Some(_), .. } = &self.state {
            let old = std::mem::replace(&mut self.state, NodeState::Ungenerated);
            if let NodeState::Merging { mut children, coarse: Some((mesh_id, object_id)) } = old {
                for child in children.iter_mut() {
                    child.remove_all(sink);
                }
                self.state = NodeState::Leaf { mesh_id, object_id };
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

    /// Per-frame update. Generates at most `budget` new meshes.
    pub fn update<S: VoxelSceneSink>(
        &mut self,
        camera: Point3<f32>,
        sink: &mut S,
        budget: usize,
    ) {
        let sdf = Arc::clone(&self.sdf);
        let max_depth = self.max_depth;
        let split_factor = self.split_factor;
        let merge_hysteresis = self.merge_hysteresis;
        let mut remaining = budget;

        for root in &mut self.roots {
            if remaining == 0 { break; }
            root.update(&camera, &sdf, sink, max_depth, split_factor, merge_hysteresis, &mut remaining, true);
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
            if *mesh_id != u32::MAX =>
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
