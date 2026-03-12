use i3_gfx::prelude::*;
use i3_slang::prelude::*;

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
    shader: Option<ShaderModule>,
    pipeline: Option<BackendPipeline>,
    width: u32,
    height: u32,
    push_constants: Option<HistogramPushConstants>,
}

impl HistogramBuildPass {
    pub fn new() -> Self {
        let dummy_image = ImageHandle(SymbolId(0));
        let dummy_buffer = BufferHandle(SymbolId(0));
        Self {
            hdr_image: dummy_image,
            histogram_buffer: dummy_buffer,
            exposure_buffer: dummy_buffer,
            hdr_image_name: "HDR_Target".to_string(),
            shader: None,
            pipeline: None,
            width: 0,
            height: 0,
            push_constants: None,
        }
    }

    pub fn create_pipeline_info(&self) -> ComputePipelineCreateInfo {
        ComputePipelineCreateInfo {
            shader_module: self.shader.clone().expect("Shader not compiled"),
        }
    }
}

impl RenderPass for HistogramBuildPass {
    fn name(&self) -> &str {
        "HistogramBuildPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend) {
        if self.pipeline.is_some() {
            return;
        }

        // 1. Compile Shader
        let slang = SlangCompiler::new().expect("Failed to create Slang compiler");
        let shader_dir = "crates/i3_renderer/shaders";

        self.shader = Some(
            slang
                .compile_file("histogram_build", ShaderTarget::Spirv, &[shader_dir])
                .expect("Failed to compile HistogramBuild shader"),
        );

        // 2. Create Pipeline
        let info = self.create_pipeline_info();
        self.pipeline = Some(backend.create_compute_pipeline(&info));
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        // Resolve target handles by name
        self.hdr_image = builder.resolve_image(&self.hdr_image_name);
        self.histogram_buffer = builder.resolve_buffer("HistogramBuffer");
        self.exposure_buffer = builder.resolve_buffer("ExposureBuffer");

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
