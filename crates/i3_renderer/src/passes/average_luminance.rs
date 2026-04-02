use std::sync::Arc;
use i3_gfx::prelude::*;


#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AverageLuminancePushConstants {
    pub min_log_lum: f32,
    pub max_log_lum: f32,
    pub time_delta: f32,
    pub adaptation_rate: f32,
    pub pixel_count: f32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// Average luminance pass struct implementing the RenderPass trait.
pub struct AverageLuminancePass {
    // Resolved handles (updated in declare)
    histogram_buffer: BufferHandle,
    exposure_buffer: BufferHandle,
    history_buffer: BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
    push_constants: Option<AverageLuminancePushConstants>,
}

impl AverageLuminancePass {
    pub fn new() -> Self {
        Self {
            histogram_buffer: BufferHandle::INVALID,
            exposure_buffer: BufferHandle::INVALID,
            history_buffer: BufferHandle::INVALID,
            pipeline: None,
            push_constants: None,
        }
    }

}

impl RenderPass for AverageLuminancePass {
    fn name(&self) -> &str {
        "AverageLuminancePass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("average_luminance").wait_loaded() {
            self.pipeline = Some(backend.create_compute_pipeline_from_baked(
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        if builder.is_setup() {
            return;
        }
        // Resolve target handles by name
        self.histogram_buffer = builder.resolve_buffer("HistogramBuffer");
        self.exposure_buffer = builder.resolve_buffer("ExposureBuffer");
        self.history_buffer = builder.read_buffer_history("ExposureBuffer");

        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        let dt = *builder.consume::<f32>("TimeDelta");

        self.push_constants = Some(AverageLuminancePushConstants {
            min_log_lum: -10.0,
            max_log_lum: 10.0,
            time_delta: dt,
            adaptation_rate: 2.0,
            pixel_count: (common.screen_width * common.screen_height) as f32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        });

        // Read histogram, read history, write exposure
        builder.read_buffer(self.histogram_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.history_buffer, ResourceUsage::SHADER_READ);
        builder.write_buffer(self.exposure_buffer, ResourceUsage::SHADER_WRITE);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.histogram_buffer,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.history_buffer,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 2,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.exposure_buffer,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
            ],
        );
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("AverageLuminancePass::execute: pipeline not initialized!");
            return;
        };
        ctx.bind_pipeline_raw(pipeline);
        if let Some(constants) = self.push_constants {
            ctx.push_constant_data(ShaderStageFlags::Compute, 0, &constants);

            // Only 1 workgroup needed to process the 256-bin histogram
            ctx.dispatch(1, 1, 1);
        }
    }
}
