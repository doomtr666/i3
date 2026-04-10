use i3_gfx::prelude::*;
use std::sync::Arc;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ToneMapPushConstants {
    pub debug_mode: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

pub struct TonemapPass {
    pub sampler: SamplerHandle,
    pub backbuffer_name: String,
    pub hdr_image_name: String,

    // Resolved handles (updated in declare)
    backbuffer: ImageHandle,
    hdr_target: ImageHandle,
    exposure_buffer: BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
}

impl TonemapPass {
    pub fn new(sampler: SamplerHandle) -> Self {
        Self {
            backbuffer: ImageHandle::INVALID,
            hdr_target: ImageHandle::INVALID,
            exposure_buffer: BufferHandle::INVALID,
            sampler,
            backbuffer_name: "LDR_Target".to_string(),
            hdr_image_name: "HDR_Target".to_string(),
            pipeline: None,
        }
    }
}

impl RenderPass for TonemapPass {
    fn name(&self) -> &str {
        "ToneMapPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("tonemap")
            .wait_loaded()
        {
            let state = asset.state.as_ref().expect("Tonemap asset missing state");
            self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                state,
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        let (w, h) = (common.screen_width, common.screen_height);

        // When FXAA is active, declare LDR_Target as an intermediate image.
        // When FXAA is disabled, resolve the existing Backbuffer directly.
        self.backbuffer = if self.backbuffer_name == "Backbuffer" {
            builder.resolve_image("Backbuffer")
        } else {
            builder.declare_image_output(
                &self.backbuffer_name,
                ImageDesc::new(w, h, Format::R8G8B8A8_UNORM),
            )
        };
        self.hdr_target = builder.resolve_image(&self.hdr_image_name);
        self.exposure_buffer = builder.resolve_buffer("ExposureBuffer");

        // Read HDR target & ExposureBuffer
        builder.read_image(self.hdr_target, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.exposure_buffer, ResourceUsage::SHADER_READ);

        // Write to backbuffer
        builder.write_image(self.backbuffer, ResourceUsage::COLOR_ATTACHMENT);

        builder.descriptor_set(0, |d| {
            d.combined_image_sampler(
                self.hdr_target,
                DescriptorImageLayout::ShaderReadOnlyOptimal,
                self.sampler,
            )
            .storage_buffer(self.exposure_buffer);
        });
    }

    fn execute(
        &self,
        ctx: &mut dyn PassContext,
        _frame: &i3_gfx::graph::compiler::FrameBlackboard,
    ) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("TonemapPass::execute: pipeline not initialized!");
            return;
        };
        ctx.bind_pipeline_raw(pipeline);
        ctx.push_constant_data(
            ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
            0,
            &ToneMapPushConstants {
                debug_mode: 0,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        ctx.draw(3, 0); // Fullscreen triangle
    }
}
