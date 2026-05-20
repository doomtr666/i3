use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

use i3_brickmap::clipmap::{BrickmapClipmapState, NUM_LEVELS};
use i3_gfx::prelude::*;
use i3_io::asset::AssetLoader;

pub struct ClipmapGBufferPass {
    pub shared:       Arc<RwLock<BrickmapClipmapState>>,
    pub enabled:      Arc<AtomicBool>,
    pub debug_flags:  Arc<AtomicU32>,

    // GBuffer / common handles
    gbuffer_albedo:     ImageHandle,
    gbuffer_normal:     ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive:   ImageHandle,
    hiz_mip0:           ImageHandle,
    depth_buffer:       ImageHandle,
    common_buffer:      BufferHandle,
    bindless_set:       DescriptorSetHandle,

    // Virtual handles (re-resolved each frame via declare)
    page_table_virt: BufferHandle,
    sdf_atlas_virt:  BufferHandle,
    mat_atlas_virt:  BufferHandle,

    pipeline: Option<BackendPipeline>,
}

impl ClipmapGBufferPass {
    pub fn new(
        shared:       Arc<RwLock<BrickmapClipmapState>>,
        enabled:      Arc<AtomicBool>,
        debug_flags:  Arc<AtomicU32>,
    ) -> Self {
        Self {
            shared,
            enabled,
            debug_flags,
            gbuffer_albedo:     ImageHandle::INVALID,
            gbuffer_normal:     ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            gbuffer_emissive:   ImageHandle::INVALID,
            hiz_mip0:           ImageHandle::INVALID,
            depth_buffer:       ImageHandle::INVALID,
            common_buffer:      BufferHandle::INVALID,
            bindless_set:       DescriptorSetHandle(0),
            page_table_virt:    BufferHandle::INVALID,
            sdf_atlas_virt:     BufferHandle::INVALID,
            mat_atlas_virt:     BufferHandle::INVALID,
            pipeline:           None,
        }
    }
}

// ─── Push constants ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
struct LevelPC {
    world_origin: [f32; 3],
    voxel_size:   f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClipmapPC {
    levels:      [LevelPC; 10],
    num_levels:  u32,
    debug_flags: u32,
    _pad:        [u32; 2],
}

// ─── RenderPass impl ──────────────────────────────────────────────────────────

impl RenderPass for ClipmapGBufferPass {
    fn name(&self) -> &str { "ClipmapGBufferPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        // Buffers are owned by ClipmapGpuBuffers and created in main.rs — no allocation here.
        let loader = globals.try_consume::<Arc<AssetLoader>>("AssetLoader").cloned();
        let Some(loader) = loader else {
            tracing::error!("ClipmapGBufferPass: no AssetLoader");
            return;
        };
        match loader.load::<i3_io::pipeline_asset::PipelineAsset>("brickmap_clipmap").wait_loaded() {
            Ok(asset) => {
                let state = asset.state.as_ref().expect("missing pipeline state");
                self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                    state,
                    &asset.reflection_data,
                    &asset.bytecode,
                ));
            }
            Err(e) => tracing::error!("ClipmapGBufferPass: failed to load pipeline: {e}"),
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // Resolve handles published by BrickmapSetupPass (same SymbolId → correct
        // RAW barrier from BakePass write to this pass's read).
        self.page_table_virt = builder.resolve_buffer("ClipmapPageTable");
        self.sdf_atlas_virt  = builder.resolve_buffer("ClipmapSdfAtlas");
        self.mat_atlas_virt  = builder.resolve_buffer("ClipmapMatAtlas");

        self.gbuffer_albedo     = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal     = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive   = builder.resolve_image("GBuffer_Emissive");
        self.hiz_mip0           = builder.resolve_image("HiZFinal");
        self.depth_buffer       = builder.resolve_image("DepthBuffer");
        self.common_buffer      = builder.resolve_buffer("CommonBuffer");
        self.bindless_set       = *builder.consume::<DescriptorSetHandle>("BindlessSet");

        builder.read_buffer(self.page_table_virt, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.sdf_atlas_virt,  ResourceUsage::SHADER_READ);
        builder.read_buffer(self.mat_atlas_virt,  ResourceUsage::SHADER_READ);
        builder.read_buffer(self.common_buffer,   ResourceUsage::SHADER_READ);
        builder.write_image(self.gbuffer_albedo,     ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_normal,     ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_roughmetal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_emissive,   ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.hiz_mip0,           ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer,       ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        if !self.enabled.load(Ordering::Relaxed) { return; }
        let Some(pipeline) = self.pipeline else { return; };

        let clipmap = self.shared.read().unwrap();

        ctx.bind_pipeline_raw(pipeline);

        let bm_set = ctx.create_descriptor_set(pipeline, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.page_table_virt),
            DescriptorWrite::storage_buffer(0, 1, self.sdf_atlas_virt),
            DescriptorWrite::storage_buffer(0, 2, self.mat_atlas_virt),
        ]);
        ctx.bind_descriptor_set(0, bm_set);

        let common_set = ctx.create_descriptor_set(pipeline, 1, &[
            DescriptorWrite::uniform_buffer(0, 0, self.common_buffer),
        ]);
        ctx.bind_descriptor_set(1, common_set);

        ctx.bind_descriptor_set(2, self.bindless_set);

        let mut pc_levels = [LevelPC { world_origin: [0.0; 3], voxel_size: 0.0 }; 10];
        for lev in 0..NUM_LEVELS {
            let ld = &clipmap.levels[lev];
            pc_levels[lev] = LevelPC {
                world_origin: ld.world_origin,
                voxel_size:   ld.voxel_size,
            };
        }
        let pc = ClipmapPC {
            levels:      pc_levels,
            num_levels:  NUM_LEVELS as u32,
            debug_flags: self.debug_flags.load(Ordering::Relaxed),
            _pad:        [0; 2],
        };
        ctx.push_constant_data(ShaderStageFlags::Vertex | ShaderStageFlags::Fragment, 0, &pc);

        ctx.draw(3, 0);
    }
}

