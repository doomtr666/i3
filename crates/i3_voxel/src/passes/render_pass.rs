use std::sync::Arc;
use std::sync::atomic::Ordering;

use i3_gfx::prelude::*;
use i3_io::asset::AssetLoader;

use super::ClipmapShared;

// ─── ClipmapRenderPass ────────────────────────────────────────────────────────────
// Fullscreen fragment pass: DDA SVO traversal + trilinear SDF sampling.
// Set 0: SVO buffers.  Set 1: common uniform (camera, view-proj).  Set 2: bindless.

pub struct ClipmapRenderPass {
    shared:         Arc<ClipmapShared>,
    pipeline:       Option<BackendPipeline>,

    page_table_virt:     BufferHandle,
    clip_levels_virt:    BufferHandle,
    geom_atlas_virt:     BufferHandle,
    weight_atlas_virt:   BufferHandle,
    prims_virt:          BufferHandle,
    vm_ops_virt:         BufferHandle,
    common_buffer:       BufferHandle,
    bindless_set:        DescriptorSetHandle,

    gbuffer_albedo:      ImageHandle,
    gbuffer_normal:      ImageHandle,
    gbuffer_roughmetal:  ImageHandle,
    gbuffer_emissive:    ImageHandle,
    hiz_mip0:            ImageHandle,
    depth_buffer:        ImageHandle,
}

impl ClipmapRenderPass {
    pub(crate) fn new(shared: Arc<ClipmapShared>, _inv_buf: BufferHandle, _inv_img: ImageHandle) -> Self {
        Self {
            shared,
            pipeline:            None,
            page_table_virt:     BufferHandle::INVALID,
            clip_levels_virt:    BufferHandle::INVALID,
            geom_atlas_virt:     BufferHandle::INVALID,
            weight_atlas_virt:   BufferHandle::INVALID,
            prims_virt:          BufferHandle::INVALID,
            vm_ops_virt:         BufferHandle::INVALID,
            common_buffer:       BufferHandle::INVALID,
            bindless_set:        DescriptorSetHandle(0),
            gbuffer_albedo:      ImageHandle::INVALID,
            gbuffer_normal:      ImageHandle::INVALID,
            gbuffer_roughmetal:  ImageHandle::INVALID,
            gbuffer_emissive:    ImageHandle::INVALID,
            hiz_mip0:            ImageHandle::INVALID,
            depth_buffer:        ImageHandle::INVALID,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClipmapPC {
    num_levels:  u32,
    debug_flags: u32,
    prim_count:  u32,
    terrain_on:  u32,
    // Terrain texture layer bindless indices (16-aligned int4s) + blend params.
    idx_albedo:  [i32; 4],
    idx_normal:  [i32; 4],
    idx_orm:     [i32; 4],
    thr0:        [f32; 4],  // snow_lo, snow_hi, slope_lo, slope_hi
    thr1:        [f32; 4],  // dirt_lo, dirt_hi, tiling, jitter_amp
}

impl RenderPass for ClipmapRenderPass {
    fn name(&self) -> &str { "ClipmapRenderPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let Some(loader) = globals.try_consume::<Arc<AssetLoader>>("AssetLoader").cloned() else { return; };
        match loader.load::<i3_io::pipeline_asset::PipelineAsset>("clipmap_render").wait_loaded() {
            Ok(asset) => {
                let state = asset.state.as_ref().expect("missing pipeline state");
                self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                    state, &asset.reflection_data, &asset.bytecode,
                ));
            }
            Err(e) => tracing::error!("ClipmapRenderPass: failed to load pipeline: {e}"),
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.page_table_virt    = builder.resolve_buffer("ClipPageTableDevice");
        self.clip_levels_virt   = builder.resolve_buffer("ClipLevels");
        self.geom_atlas_virt    = builder.resolve_buffer("ClipGeomAtlas");
        self.weight_atlas_virt  = builder.resolve_buffer("ClipWeightAtlas");
        self.prims_virt         = builder.resolve_buffer("ClipPrims");
        self.vm_ops_virt        = builder.resolve_buffer("ClipVmOps");
        self.common_buffer      = builder.resolve_buffer("CommonBuffer");
        self.bindless_set       = *builder.consume::<DescriptorSetHandle>("BindlessSet");

        self.gbuffer_albedo     = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal     = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive   = builder.resolve_image("GBuffer_Emissive");
        self.hiz_mip0           = builder.resolve_image("HiZFinal");
        self.depth_buffer       = builder.resolve_image("DepthBuffer");

        builder.read_buffer(self.page_table_virt, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.clip_levels_virt, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.geom_atlas_virt, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.weight_atlas_virt, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.prims_virt,     ResourceUsage::SHADER_READ);
        builder.read_buffer(self.vm_ops_virt,    ResourceUsage::SHADER_READ);
        builder.read_buffer(self.common_buffer,  ResourceUsage::SHADER_READ);

        builder.write_image(self.gbuffer_albedo,     ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_normal,     ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_roughmetal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_emissive,   ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.hiz_mip0,           ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer,       ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        if !self.shared.enabled.load(Ordering::Relaxed) { return; }
        let Some(pl) = self.pipeline else { return; };

        let prim_count = self.shared.prim_count.load(Ordering::Acquire);

        ctx.bind_pipeline_raw(pl);

        let clip_set = ctx.create_descriptor_set(pl, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.page_table_virt),
            DescriptorWrite::storage_buffer(0, 1, self.geom_atlas_virt),
            DescriptorWrite::storage_buffer(0, 2, self.prims_virt),
            DescriptorWrite::storage_buffer(0, 3, self.vm_ops_virt),
            DescriptorWrite::storage_buffer(0, 4, self.weight_atlas_virt),
            DescriptorWrite::storage_buffer(0, 5, self.clip_levels_virt),
        ]);
        ctx.bind_descriptor_set(0, clip_set);

        let common_set = ctx.create_descriptor_set(pl, 1, &[
            DescriptorWrite::uniform_buffer(0, 0, self.common_buffer),
        ]);
        ctx.bind_descriptor_set(1, common_set);
        ctx.bind_descriptor_set(2, self.bindless_set);

        let tm = self.shared.terrain_mat;
        let pc = ClipmapPC {
            num_levels: crate::clipmap::NUM_LEVELS as u32,
            debug_flags: self.shared.debug_flags.load(Ordering::Relaxed),
            prim_count,
            terrain_on: self.shared.terrain_on.load(Ordering::Acquire),
            idx_albedo: tm.albedo,
            idx_normal: tm.normal,
            idx_orm:    tm.orm,
            thr0: [tm.snow_lo, tm.snow_hi, tm.slope_lo, tm.slope_hi],
            thr1: [tm.dirt_lo, tm.dirt_hi, tm.tiling, tm.jitter_amp],
        };
        ctx.push_constant_data(ShaderStageFlags::Vertex | ShaderStageFlags::Fragment, 0, &pc);
        ctx.draw(3, 0);
    }
}
