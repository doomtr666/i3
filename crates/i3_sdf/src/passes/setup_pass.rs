use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i3_gfx::prelude::*;

use crate::gpu_scene::{GpuBrickJob, GpuSvoNode, pack_bvh, pack_scene};
use crate::gpu_buffers::RING;
use crate::MAX_PRIMS;
use super::SvoShared;

// ─── SvoSetupPass ─────────────────────────────────────────────────────────────
// Uploads per-frame CPU data (node pool, bake jobs, SDF scene) to GPU.
// First frame: clears sdf/mat atlases to "fully outside" sentinel.

pub struct SvoSetupPass {
    shared:          Arc<SvoShared>,
    first_frame:     AtomicBool,
    node_pool_virt:  BufferHandle,
    geom_atlas_virt: BufferHandle,
    jobs_virt:       BufferHandle,
    prims_virt:      BufferHandle,
    bvh_virt:        BufferHandle,
    vm_ops_virt:     BufferHandle,
}

impl SvoSetupPass {
    pub(crate) fn new(shared: Arc<SvoShared>, _inv: BufferHandle) -> Self {
        Self {
            shared,
            first_frame:     AtomicBool::new(true),
            node_pool_virt:  BufferHandle::INVALID,
            geom_atlas_virt: BufferHandle::INVALID,
            jobs_virt:       BufferHandle::INVALID,
            prims_virt:      BufferHandle::INVALID,
            bvh_virt:        BufferHandle::INVALID,
            vm_ops_virt:     BufferHandle::INVALID,
        }
    }
}

impl RenderPass for SvoSetupPass {
    fn name(&self) -> &str { "SvoSetupPass" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let gb = &self.shared.gpu_buffers;

        // Advance the per-frame ring slot. SvoSetupPass::declare is the first SVO
        // pass to run each frame, so this establishes the slot that BakePass and
        // RenderPass will resolve by name.
        let ring = self.shared.cur_ring.load(Ordering::Relaxed);
        let next = (ring + 1) % RING;
        self.shared.cur_ring.store(next, Ordering::Relaxed);

        self.node_pool_virt  = builder.import_buffer("SvoNodePool",  gb.node_pool[next]);
        self.geom_atlas_virt = builder.import_buffer("SvoGeomAtlas", gb.geom_atlas_buf);
        self.jobs_virt       = builder.import_buffer("SvoBakeJobs",  gb.jobs[next]);
        self.prims_virt      = builder.import_buffer("SvoPrims",     gb.prims[next]);
        self.bvh_virt        = builder.import_buffer("SvoBvh",       gb.bvh[next]);
        self.vm_ops_virt     = builder.import_buffer("SvoVmOps",     gb.vm_ops[next]);

        // Declare the CpuToGpu buffers this pass map-writes as WRITES. The actual
        // write is a host map, invisible to the frame graph — but declaring it as a
        // write creates a real write→read dependency with the bake pass (jobs/prims)
        // and render pass (node_pool). Without this the graph sees only reads, finds
        // no dependency, and records setup + bake IN PARALLEL (ExecuteParallel /
        // rayon) — racing the `job_count` side-channel atomic so the bake dispatches
        // a stale count and some slots are never baked (stale "wrong-place" bricks).
        builder.write_buffer(self.node_pool_virt, ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.jobs_virt,      ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.prims_virt,     ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.bvh_virt,       ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.vm_ops_virt,    ResourceUsage::SHADER_WRITE);

        if self.first_frame.load(Ordering::Relaxed) {
            builder.write_buffer(self.geom_atlas_virt, ResourceUsage::TRANSFER_WRITE);
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        let (jobs, gpu_nodes): (Vec<GpuBrickJob>, Vec<GpuSvoNode>) = {
            let mut tree = self.shared.svo_tree.write().unwrap();
            tree.collect_frame_upload()
        };

        let scene    = self.shared.sdf_scene.read().unwrap();
        let prims    = pack_scene(&scene);
        let bvh      = pack_bvh(&scene);
        let has_terrain = scene.nodes().iter().any(|n| {
            matches!(n.primitive(), i3_voxel::SdfPrimitive::VolumeTerrain { .. })
        });
        drop(scene);

        self.shared.terrain_on.store(has_terrain as u32, Ordering::Release);
        self.shared.prim_count.store(prims.len() as u32, Ordering::Release);
        self.shared.bvh_root.store(if bvh.is_empty() { u32::MAX } else { 0 }, Ordering::Release);
        self.shared.job_count.store(jobs.len() as u32, Ordering::Release);

        upload(ctx, self.node_pool_virt, &gpu_nodes, gpu_nodes.len());
        { let ops = &self.shared.vm_ops; if !ops.is_empty() { upload(ctx, self.vm_ops_virt, ops, ops.len()); } }
        if !jobs.is_empty()  { upload(ctx, self.jobs_virt,  &jobs,  jobs.len()); }
        if !prims.is_empty() { upload(ctx, self.prims_virt, &prims, prims.len().min(MAX_PRIMS as usize)); }
        if !bvh.is_empty()   { upload(ctx, self.bvh_virt,   &bvh,   bvh.len().min(65536)); }

        // Clear to a "far outside" sentinel: each 8-byte voxel's distance is the
        // low f16 of every u32; 0x7BFF is f16 ≈ 65504, so any unbaked/in-flight slot
        // reads as far-from-surface (no spurious surface) rather than distance 0.
        if self.first_frame.swap(false, Ordering::AcqRel) {
            ctx.clear_buffer(self.geom_atlas_virt, 0x7BFF_7BFFu32);
        }

        tracing::debug!(
            "SvoSetupPass: {} nodes  {} jobs  {} prims",
            gpu_nodes.len(), jobs.len(), prims.len(),
        );
    }
}

fn upload<T>(ctx: &mut dyn PassContext, handle: BufferHandle, data: &[T], count: usize) {
    let ptr = ctx.map_buffer(handle);
    if !ptr.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut T, count.min(data.len()));
        }
        ctx.unmap_buffer(handle);
    }
}
