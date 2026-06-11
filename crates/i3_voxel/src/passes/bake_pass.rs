use std::sync::Arc;
use std::sync::atomic::Ordering;

use i3_gfx::prelude::*;
use i3_io::asset::AssetLoader;

use super::ClipmapShared;

// ─── ClipmapBakePass ──────────────────────────────────────────────────────────────
// Evaluates SDF at 9³ voxels per pending brick.
// Dispatch: (job_count, 1, 1) workgroups of [9, 9, 9] threads.
// Same atlas format as brickmap: u8-packed, BRICK_DWORDS per brick.

pub struct ClipmapBakePass {
    shared:          Arc<ClipmapShared>,
    pipeline:        Option<BackendPipeline>,
    jobs_virt:       BufferHandle,
    prims_virt:      BufferHandle,
    geom_atlas_virt: BufferHandle,
    weight_atlas_virt: BufferHandle,
    vm_ops_virt:     BufferHandle,
}

impl ClipmapBakePass {
    pub(crate) fn new(shared: Arc<ClipmapShared>, _inv: BufferHandle) -> Self {
        Self {
            shared,
            pipeline:        None,
            jobs_virt:       BufferHandle::INVALID,
            geom_atlas_virt: BufferHandle::INVALID,
            weight_atlas_virt: BufferHandle::INVALID,
            prims_virt:      BufferHandle::INVALID,
            vm_ops_virt:     BufferHandle::INVALID,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BakePC {
    prim_count: u32, terrain_on: u32, _pad: [u32; 2],
    thr0: [f32; 4],   // snow_lo, snow_hi, slope_lo, slope_hi
    thr1: [f32; 4],   // dirt_lo, dirt_hi, tiling(unused), jitter_amp
}

impl RenderPass for ClipmapBakePass {
    fn name(&self) -> &str { "ClipmapBakePass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let Some(loader) = globals.try_consume::<Arc<AssetLoader>>("AssetLoader").cloned() else { return; };
        match loader.load::<i3_io::pipeline_asset::PipelineAsset>("clipmap_bake").wait_loaded() {
            Ok(asset) => self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode)
            ),
            Err(e) => tracing::error!("ClipmapBakePass: failed to load pipeline: {e}"),
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.jobs_virt         = builder.resolve_buffer("ClipBakeJobs");
        self.prims_virt        = builder.resolve_buffer("ClipPrims");
        self.geom_atlas_virt   = builder.resolve_buffer("ClipGeomAtlas");
        self.weight_atlas_virt = builder.resolve_buffer("ClipWeightAtlas");
        self.vm_ops_virt       = builder.resolve_buffer("ClipVmOps");

        builder.read_buffer(self.jobs_virt,          ResourceUsage::SHADER_READ);
        builder.read_buffer(self.prims_virt,         ResourceUsage::SHADER_READ);
        builder.read_buffer(self.vm_ops_virt,        ResourceUsage::SHADER_READ);
        builder.write_buffer(self.geom_atlas_virt,   ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.weight_atlas_virt, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        let job_count = self.shared.job_count.load(Ordering::Acquire);
        if job_count == 0 { return; }

        let Some(pl) = self.pipeline else { return; };
        ctx.bind_pipeline_raw(pl);

        let set = ctx.create_descriptor_set(pl, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.jobs_virt),
            DescriptorWrite::storage_buffer(0, 1, self.prims_virt),
            DescriptorWrite::storage_buffer(0, 2, self.geom_atlas_virt),
            DescriptorWrite::storage_buffer(0, 3, self.vm_ops_virt),
            DescriptorWrite::storage_buffer(0, 4, self.weight_atlas_virt),
        ]);
        ctx.bind_descriptor_set(0, set);

        let prim_count = self.shared.prim_count.load(Ordering::Acquire);
        let terrain_on = self.shared.terrain_on.load(Ordering::Acquire);
        let tm = self.shared.terrain_mat;
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &BakePC {
            prim_count, terrain_on, _pad: [0; 2],
            thr0: [tm.snow_lo, tm.snow_hi, tm.slope_lo, tm.slope_hi],
            thr1: [tm.dirt_lo, tm.dirt_hi, tm.tiling, tm.jitter_amp],
        });
        ctx.dispatch(job_count, 1, 1);
    }
}
