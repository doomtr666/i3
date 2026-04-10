use i3_gfx::prelude::*;
use std::sync::Arc;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FxaaPushConstants {
    pub inv_screen_size: [f32; 2],
    /// Subpixel AA strength [0, 1]. Default 0.75.
    pub subpix: f32,
    /// Minimum local contrast to trigger AA. Default 0.125.
    pub edge_threshold: f32,
    /// 0 = passthrough blit, 1 = FXAA active.
    pub enabled: u32,
}

pub struct FxaaPass {
    ldr_target: ImageHandle,
    backbuffer: ImageHandle,
    sampler: SamplerHandle,
    pub enabled: bool,
    pipeline: Option<BackendPipeline>,
}

impl FxaaPass {
    pub fn new(sampler: SamplerHandle) -> Self {
        Self {
            ldr_target: ImageHandle::INVALID,
            backbuffer: ImageHandle::INVALID,
            sampler,
            enabled: true,
            pipeline: None,
        }
    }
}

impl RenderPass for FxaaPass {
    fn name(&self) -> &str {
        "FxaaPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("fxaa")
            .wait_loaded()
        {
            let state = asset.state.as_ref().expect("FXAA asset missing state");
            self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                state,
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");

        self.ldr_target = builder.resolve_image("LDR_Target");
        self.backbuffer = builder.resolve_image("Backbuffer");

        builder.read_image(self.ldr_target, ResourceUsage::SHADER_READ);
        builder.write_image(self.backbuffer, ResourceUsage::COLOR_ATTACHMENT);

        builder.descriptor_set(0, |d| {
            d.combined_image_sampler(
                self.ldr_target,
                DescriptorImageLayout::ShaderReadOnlyOptimal,
                self.sampler,
            );
        });

        let _ = common;
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("FxaaPass::execute: pipeline not initialized!");
            return;
        };
        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let push = FxaaPushConstants {
            inv_screen_size: [
                1.0 / common.screen_width as f32,
                1.0 / common.screen_height as f32,
            ],
            subpix: 0.75,
            edge_threshold: 0.125,
            enabled: self.enabled as u32,
        };

        ctx.bind_pipeline_raw(pipeline);
        ctx.push_constant_data(
            ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
            0,
            &push,
        );
        ctx.draw(3, 0);
    }
}
