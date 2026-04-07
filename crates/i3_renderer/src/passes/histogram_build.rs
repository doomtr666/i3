use i3_gfx::prelude::*;
use std::sync::Arc;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HistogramPushConstants {
    pub min_log_lum: f32,
    pub max_log_lum: f32,
    pub time_delta: f32,
    pub pad: u32,
}

/// Histogram build pass struct implementing the RenderPass trait.
pub struct HistogramBuildPass {
    pub hdr_image_name: String,

    // Resolved handles (updated in declare)
    hdr_image: ImageHandle,
    histogram_buffer: BufferHandle,
    exposure_buffer: BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
}

impl HistogramBuildPass {
    pub fn new() -> Self {
        Self {
            hdr_image: ImageHandle::INVALID,
            histogram_buffer: BufferHandle::INVALID,
            exposure_buffer: BufferHandle::INVALID,
            hdr_image_name: "HDR_Target".to_string(),
            pipeline: None,
        }
    }
}

impl RenderPass for HistogramBuildPass {
    fn name(&self) -> &str {
        "HistogramBuildPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("histogram_build")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.hdr_image = builder.resolve_image(&self.hdr_image_name);
        self.histogram_buffer = builder.resolve_buffer("HistogramBuffer");
        self.exposure_buffer = builder.read_buffer_history("ExposureBuffer");

        // Read HDR image and exposure buffer
        builder.read_image(self.hdr_image, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.exposure_buffer, ResourceUsage::SHADER_READ);

        // Write to histogram buffer
        builder.write_buffer(self.histogram_buffer, ResourceUsage::SHADER_WRITE);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::Texture,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.hdr_image,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: None,
                    }),
                    accel_struct_info: None,
                },
                DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.histogram_buffer,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                    accel_struct_info: None,
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
                    accel_struct_info: None,
                },
            ],
        );
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("HistogramBuildPass::execute: pipeline not initialized!");
            return;
        };
        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let dt = *frame.consume::<f32>("TimeDelta");
        let push_constants = HistogramPushConstants {
            min_log_lum: -10.0,
            max_log_lum: 10.0,
            time_delta: dt,
            pad: 0,
        };
        ctx.bind_pipeline_raw(pipeline);
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &push_constants);
        // Assuming GROUP_SIZE = 16
        let group_count_x = (common.screen_width + 15) / 16;
        let group_count_y = (common.screen_height + 15) / 16;
        ctx.dispatch(group_count_x, group_count_y, 1);
    }
}
