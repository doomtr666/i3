use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use i3_gfx::prelude::*;

use crate::clipmap::{ClipLevelGpu, GRID_VOL, NUM_LEVELS};
use crate::gpu_buffers::RING;
use crate::gpu_scene::{pack_bvh, pack_scene};
use crate::MAX_PRIMS;
use super::ClipmapShared;

// ─── ClipmapSetupPass ─────────────────────────────────────────────────────────────
// Uploads per-frame CPU data (clipmap page table + level params, bake jobs, SDF scene)
// to GPU. First frame: clears geom/weight atlases to the "fully outside" sentinel.

pub struct ClipmapSetupPass {
    shared:            Arc<ClipmapShared>,
    first_frame:       AtomicBool,
    page_table_virt:   BufferHandle,
    page_table_dev:    BufferHandle,   // device-local copy, read by render
    clip_levels_virt:  BufferHandle,
    geom_atlas_virt:   BufferHandle,
    weight_atlas_virt: BufferHandle,   // pre-mix layer weights (baked, read by render)
    jobs_virt:         BufferHandle,
    prims_virt:        BufferHandle,
    bvh_virt:          BufferHandle,
    vm_ops_virt:       BufferHandle,
}

impl ClipmapSetupPass {
    pub(crate) fn new(shared: Arc<ClipmapShared>, _inv: BufferHandle) -> Self {
        Self {
            shared,
            first_frame:       AtomicBool::new(true),
            page_table_virt:   BufferHandle::INVALID,
            page_table_dev:    BufferHandle::INVALID,
            clip_levels_virt:  BufferHandle::INVALID,
            geom_atlas_virt:   BufferHandle::INVALID,
            weight_atlas_virt: BufferHandle::INVALID,
            jobs_virt:         BufferHandle::INVALID,
            prims_virt:        BufferHandle::INVALID,
            bvh_virt:          BufferHandle::INVALID,
            vm_ops_virt:       BufferHandle::INVALID,
        }
    }
}

impl RenderPass for ClipmapSetupPass {
    fn name(&self) -> &str { "ClipmapSetupPass" }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let gb = &self.shared.gpu_buffers;

        // Advance the per-frame ring slot. ClipmapSetupPass::declare is the first SVO
        // pass to run each frame, so this establishes the slot that BakePass and
        // RenderPass will resolve by name.
        let ring = self.shared.cur_ring.load(Ordering::Relaxed);
        let next = (ring + 1) % RING;
        self.shared.cur_ring.store(next, Ordering::Relaxed);

        self.page_table_virt = builder.import_buffer("ClipPageTable", gb.page_table[next]);
        // Device-local page table the render reads (frame-graph-managed: pooling +
        // in-flight buffering + the copy→read barrier handled by the graph). The host-
        // visible staging above is copied into it each frame, so the marcher reads fast
        // VRAM instead of host memory over PCIe at every cell crossing.
        self.page_table_dev  = builder.declare_buffer_output("ClipPageTableDevice", BufferDesc {
            size:   (NUM_LEVELS * GRID_VOL * std::mem::size_of::<u32>()) as u64,
            usage:  BufferUsageFlags::STORAGE_BUFFER,
            memory: MemoryType::GpuOnly,
        });
        self.clip_levels_virt  = builder.import_buffer("ClipLevels",     gb.clip_levels[next]);
        self.geom_atlas_virt   = builder.import_buffer("ClipGeomAtlas",   gb.geom_atlas_buf);
        self.weight_atlas_virt = builder.import_buffer("ClipWeightAtlas", gb.weight_atlas_buf);
        self.jobs_virt         = builder.import_buffer("ClipBakeJobs",    gb.jobs[next]);
        self.prims_virt        = builder.import_buffer("ClipPrims",       gb.prims[next]);
        self.bvh_virt          = builder.import_buffer("ClipBvh",         gb.bvh[next]);
        self.vm_ops_virt       = builder.import_buffer("ClipVmOps",       gb.vm_ops[next]);

        // Declare the CpuToGpu buffers this pass map-writes as WRITES. The actual write is
        // a host map, invisible to the frame graph — but declaring it as a write creates a
        // real write→read dependency with the bake pass (jobs/prims) and render pass (page
        // table / levels). Without this the graph sees only reads, finds no dependency, and
        // records setup + bake IN PARALLEL (rayon) — racing the `job_count` side-channel.
        builder.write_buffer(self.page_table_virt,  ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.page_table_dev,   ResourceUsage::TRANSFER_WRITE);
        builder.write_buffer(self.clip_levels_virt, ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.jobs_virt,        ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.prims_virt,       ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.bvh_virt,         ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.vm_ops_virt,      ResourceUsage::SHADER_WRITE);

        if self.first_frame.load(Ordering::Relaxed) {
            builder.write_buffer(self.geom_atlas_virt,   ResourceUsage::TRANSFER_WRITE);
            builder.write_buffer(self.weight_atlas_virt, ResourceUsage::TRANSFER_WRITE);
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        // Drain bake jobs + snapshot page table / level params under one lock.
        let levels: [ClipLevelGpu; NUM_LEVELS];
        let job_count;
        {
            let mut clip = self.shared.clipmap.write().unwrap();
            let jobs = clip.collect_frame_upload();
            levels = clip.levels_gpu();
            job_count = jobs.len();

            upload(ctx, self.page_table_virt, clip.page_table(), NUM_LEVELS * GRID_VOL);
            // Stage → device-local. The render reads `page_table_dev` (VRAM), never the
            // host-visible staging (PCIe).
            let bytes = (NUM_LEVELS * GRID_VOL * std::mem::size_of::<u32>()) as u64;
            ctx.copy_buffer(self.page_table_virt, self.page_table_dev, 0, 0, bytes);

            upload(ctx, self.clip_levels_virt, &levels, NUM_LEVELS);
            if !jobs.is_empty() { upload(ctx, self.jobs_virt, &jobs, jobs.len()); }
        }

        let scene    = self.shared.sdf_scene.read().unwrap();
        let prims    = pack_scene(&scene);
        let bvh      = pack_bvh(&scene);
        let has_terrain = scene.nodes().iter().any(|n| {
            matches!(n.primitive(), crate::SdfPrimitive::VolumeTerrain { .. })
        });
        drop(scene);

        self.shared.terrain_on.store(has_terrain as u32, Ordering::Release);
        self.shared.prim_count.store(prims.len() as u32, Ordering::Release);
        self.shared.bvh_root.store(if bvh.is_empty() { u32::MAX } else { 0 }, Ordering::Release);
        self.shared.job_count.store(job_count as u32, Ordering::Release);

        { let ops = &self.shared.vm_ops; if !ops.is_empty() { upload(ctx, self.vm_ops_virt, ops, ops.len()); } }
        if !prims.is_empty() { upload(ctx, self.prims_virt, &prims, prims.len().min(MAX_PRIMS as usize)); }
        if !bvh.is_empty()   { upload(ctx, self.bvh_virt,   &bvh,   bvh.len().min(65536)); }

        // Clear to a "far outside" sentinel: each 8-byte voxel's distance is the
        // low f16 of every u32; 0x7BFF is f16 ≈ 65504, so any unbaked/in-flight slot
        // reads as far-from-surface (no spurious surface) rather than distance 0.
        if self.first_frame.swap(false, Ordering::AcqRel) {
            ctx.clear_buffer(self.geom_atlas_virt, 0x7BFF_7BFFu32);
            ctx.clear_buffer(self.weight_atlas_virt, 0u32);   // weights default to 0
        }

        tracing::debug!("ClipmapSetupPass: {} jobs  {} prims", job_count, prims.len());
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
