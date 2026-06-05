use std::sync::Arc;
use std::sync::atomic::Ordering;

use i3_gfx::prelude::*;
use i3_io::asset::AssetLoader;

use super::SvoShared;

// ─── SvoRenderPass ────────────────────────────────────────────────────────────
// Fullscreen fragment pass: DDA SVO traversal + trilinear SDF sampling.
// Set 0: SVO buffers.  Set 1: common uniform (camera, view-proj).  Set 2: bindless.

pub struct SvoRenderPass {
    shared:         Arc<SvoShared>,
    pipeline:       Option<BackendPipeline>,

    node_pool_virt:      BufferHandle,
    geom_atlas_virt:     BufferHandle,
    prims_virt:          BufferHandle,
    common_buffer:       BufferHandle,
    bindless_set:        DescriptorSetHandle,

    gbuffer_albedo:      ImageHandle,
    gbuffer_normal:      ImageHandle,
    gbuffer_roughmetal:  ImageHandle,
    gbuffer_emissive:    ImageHandle,
    hiz_mip0:            ImageHandle,
    depth_buffer:        ImageHandle,
}

impl SvoRenderPass {
    pub(crate) fn new(shared: Arc<SvoShared>, _inv_buf: BufferHandle, _inv_img: ImageHandle) -> Self {
        Self {
            shared,
            pipeline:            None,
            node_pool_virt:      BufferHandle::INVALID,
            geom_atlas_virt:     BufferHandle::INVALID,
            prims_virt:          BufferHandle::INVALID,
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
struct SvoPC {
    node_count:  u32,
    debug_flags: u32,
    prim_count:  u32,
    _pad:        u32,
}

impl RenderPass for SvoRenderPass {
    fn name(&self) -> &str { "SvoRenderPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let Some(loader) = globals.try_consume::<Arc<AssetLoader>>("AssetLoader").cloned() else { return; };
        match loader.load::<i3_io::pipeline_asset::PipelineAsset>("svo_render").wait_loaded() {
            Ok(asset) => {
                let state = asset.state.as_ref().expect("missing pipeline state");
                self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                    state, &asset.reflection_data, &asset.bytecode,
                ));
            }
            Err(e) => tracing::error!("SvoRenderPass: failed to load pipeline: {e}"),
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.node_pool_virt     = builder.resolve_buffer("SvoNodePool");
        self.geom_atlas_virt    = builder.resolve_buffer("SvoGeomAtlas");
        self.prims_virt         = builder.resolve_buffer("SvoPrims");
        self.common_buffer      = builder.resolve_buffer("CommonBuffer");
        self.bindless_set       = *builder.consume::<DescriptorSetHandle>("BindlessSet");

        self.gbuffer_albedo     = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal     = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive   = builder.resolve_image("GBuffer_Emissive");
        self.hiz_mip0           = builder.resolve_image("HiZFinal");
        self.depth_buffer       = builder.resolve_image("DepthBuffer");

        builder.read_buffer(self.node_pool_virt, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.geom_atlas_virt, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.prims_virt,     ResourceUsage::SHADER_READ);
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

        let node_count = {
            let tree = self.shared.svo_tree.read().unwrap();
            tree.nodes.len() as u32
        };
        let prim_count = self.shared.prim_count.load(Ordering::Acquire);

        ctx.bind_pipeline_raw(pl);

        let svo_set = ctx.create_descriptor_set(pl, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.node_pool_virt),
            DescriptorWrite::storage_buffer(0, 1, self.geom_atlas_virt),
            DescriptorWrite::storage_buffer(0, 2, self.prims_virt),
        ]);
        ctx.bind_descriptor_set(0, svo_set);

        let common_set = ctx.create_descriptor_set(pl, 1, &[
            DescriptorWrite::uniform_buffer(0, 0, self.common_buffer),
        ]);
        ctx.bind_descriptor_set(1, common_set);
        ctx.bind_descriptor_set(2, self.bindless_set);

        let pc = SvoPC {
            node_count,
            debug_flags: self.shared.debug_flags.load(Ordering::Relaxed),
            prim_count,
            _pad: 0,
        };
        ctx.push_constant_data(ShaderStageFlags::Vertex | ShaderStageFlags::Fragment, 0, &pc);
        ctx.draw(3, 0);
    }
}
