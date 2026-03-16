use i3_gfx::prelude::*;


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

    // Resolved handles (updated in record)
    hdr_image: ImageHandle,
    histogram_buffer: BufferHandle,
    exposure_buffer: BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
    width: u32,
    height: u32,
    push_constants: Option<HistogramPushConstants>,
}

impl HistogramBuildPass {
    pub fn new() -> Self {
        Self {
            hdr_image: ImageHandle::INVALID,
            histogram_buffer: BufferHandle::INVALID,
            exposure_buffer: BufferHandle::INVALID,
            hdr_image_name: "HDR_Target".to_string(),
            pipeline: None,
            width: 0,
            height: 0,
            push_constants: None,
        }
    }

    pub fn init_from_baked(
        &mut self,
        backend: &mut dyn RenderBackend,
        asset: &i3_io::pipeline_asset::PipelineAsset,
    ) {
        if self.pipeline.is_some() {
            return;
        }

        self.pipeline = Some(backend.create_compute_pipeline_from_baked(
            &asset.reflection_data,
            &asset.bytecode,
        ));
    }
}

impl RenderPass for HistogramBuildPass {
    fn name(&self) -> &str {
        "HistogramBuildPass"
    }

    fn init(&mut self, _backend: &mut dyn RenderBackend) {
        // Handled by init_from_baked
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        // Resolve target handles by name
        self.hdr_image = builder.resolve_image(&self.hdr_image_name);
        self.histogram_buffer = builder.resolve_buffer("HistogramBuffer");
        self.exposure_buffer = builder.read_buffer_history("ExposureBuffer");

        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        self.width = common.screen_width;
        self.height = common.screen_height;

        self.push_constants = Some(HistogramPushConstants {
            min_log_lum: -10.0,
            max_log_lum: 10.0,
            time_delta: *builder.consume::<f32>("TimeDelta"),
            pad: 0,
        });

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
        let pipeline = self
            .pipeline
            .expect("HistogramBuildPass pipeline not initialized");
        ctx.bind_pipeline_raw(pipeline);
        if let Some(constants) = self.push_constants {
            ctx.push_constant_data(ShaderStageFlags::Compute, 0, &constants);

            // Assuming GROUP_SIZE = 16
            let group_count_x = (self.width + 15) / 16;
            let group_count_y = (self.height + 15) / 16;
            ctx.dispatch(group_count_x, group_count_y, 1);
        }
    }
}
