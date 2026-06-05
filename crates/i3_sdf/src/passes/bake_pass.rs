use std::sync::Arc;
use std::sync::atomic::Ordering;

use i3_gfx::prelude::*;
use i3_io::asset::AssetLoader;

use super::SvoShared;

// ─── SvoBakePass ──────────────────────────────────────────────────────────────
// Evaluates SDF at 9³ voxels per pending brick.
// Dispatch: (job_count, 1, 1) workgroups of [9, 9, 9] threads.
// Same atlas format as brickmap: u8-packed, BRICK_DWORDS per brick.

pub struct SvoBakePass {
    shared:         Arc<SvoShared>,
    pipeline:       Option<BackendPipeline>,
    jobs_virt:      BufferHandle,
    prims_virt:     BufferHandle,
    sdf_atlas_virt: BufferHandle,
    mat_atlas_virt: BufferHandle,
}

impl SvoBakePass {
    pub(crate) fn new(shared: Arc<SvoShared>, _inv: BufferHandle) -> Self {
        Self {
            shared,
            pipeline:       None,
            jobs_virt:      BufferHandle::INVALID,
            sdf_atlas_virt: BufferHandle::INVALID,
            mat_atlas_virt: BufferHandle::INVALID,
            prims_virt:     BufferHandle::INVALID,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BakePC { prim_count: u32, _pad: [u32; 3] }

impl RenderPass for SvoBakePass {
    fn name(&self) -> &str { "SvoBakePass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let Some(loader) = globals.try_consume::<Arc<AssetLoader>>("AssetLoader").cloned() else { return; };
        match loader.load::<i3_io::pipeline_asset::PipelineAsset>("svo_bake").wait_loaded() {
            Ok(asset) => self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode)
            ),
            Err(e) => tracing::error!("SvoBakePass: failed to load pipeline: {e}"),
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.jobs_virt      = builder.resolve_buffer("SvoBakeJobs");
        self.prims_virt     = builder.resolve_buffer("SvoPrims");
        self.sdf_atlas_virt = builder.resolve_buffer("SvoSdfAtlas");
        self.mat_atlas_virt = builder.resolve_buffer("SvoMatAtlas");

        builder.read_buffer(self.jobs_virt,       ResourceUsage::SHADER_READ);
        builder.read_buffer(self.prims_virt,      ResourceUsage::SHADER_READ);
        builder.write_buffer(self.sdf_atlas_virt, ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.mat_atlas_virt, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        let job_count = self.shared.job_count.load(Ordering::Acquire);
        if job_count == 0 { return; }

        let Some(pl) = self.pipeline else { return; };
        ctx.bind_pipeline_raw(pl);

        let set = ctx.create_descriptor_set(pl, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.jobs_virt),
            DescriptorWrite::storage_buffer(0, 1, self.prims_virt),
            DescriptorWrite::storage_buffer(0, 2, self.sdf_atlas_virt),
            DescriptorWrite::storage_buffer(0, 3, self.mat_atlas_virt),
        ]);
        ctx.bind_descriptor_set(0, set);

        let prim_count = self.shared.prim_count.load(Ordering::Acquire);
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &BakePC { prim_count, _pad: [0; 3] });
        ctx.dispatch(job_count, 1, 1);
    }
}
