use i3_gfx::prelude::*;
use std::sync::Arc;



pub struct SkyPass {
    // Resolved handles (updated in declare)
    hdr_target: ImageHandle,
    depth_buffer: ImageHandle,

    pipeline:      Option<BackendPipeline>,
    common_buffer: BufferHandle,
}
impl SkyPass {
    pub fn new() -> Self {
        Self {
            hdr_target:    ImageHandle::INVALID,
            depth_buffer:  ImageHandle::INVALID,
            common_buffer: BufferHandle::INVALID,
            pipeline:      None,
        }
    }
}

impl RenderPass for SkyPass {
    fn name(&self) -> &str {
        "SkyPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("sky")
            .wait_loaded()
        {
            let state = asset.state.as_ref().expect("Sky asset missing state");
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
        self.hdr_target = builder.declare_image_output(
            "HDR_Target",
            ImageDesc {
                width:        w,
                height:       h,
                depth:        1,
                format:       Format::R16G16B16A16_SFLOAT,
                mip_levels:   1,
                array_layers: 1,
                // STORAGE required for SSSR composite and bloom composite (RWTexture2D writes).
                usage:        ImageUsageFlags::SAMPLED
                    | ImageUsageFlags::COLOR_ATTACHMENT
                    | ImageUsageFlags::STORAGE,
                view_type:    ImageViewType::Type2D,
                swizzle:      ComponentMapping::default(),
                clear_value:  None,
            },
        );
        self.depth_buffer = builder.resolve_image("DepthBuffer");
        self.common_buffer = builder.resolve_buffer("CommonBuffer");

        builder.read_buffer(self.common_buffer, ResourceUsage::SHADER_READ);
        builder.write_image(self.hdr_target, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer, ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("SkyPass::execute: pipeline not initialized!");
            return;
        };

        let common_set = ctx.create_descriptor_set(
            pipeline,
            1,
            &[DescriptorWrite::uniform_buffer(0, 0, self.common_buffer)],
        );

        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");

        ctx.bind_pipeline_raw(pipeline);
        ctx.bind_descriptor_set(1, common_set);
        ctx.bind_descriptor_set(2, bindless_set);
        ctx.draw(3, 0); // Fullscreen triangle
    }
}
