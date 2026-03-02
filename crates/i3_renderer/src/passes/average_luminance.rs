use i3_gfx::prelude::*;
use i3_slang::prelude::*;

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
    pub histogram_buffer: BufferHandle,
    pub exposure_buffer: BufferHandle,

    // Persistence
    shader: Option<ShaderModule>,
    pipeline: Option<BackendPipeline>,
    push_constants: Option<AverageLuminancePushConstants>,
}

impl AverageLuminancePass {
    pub fn new(histogram_buffer: BufferHandle, exposure_buffer: BufferHandle) -> Self {
        Self {
            histogram_buffer,
            exposure_buffer,
            shader: None,
            pipeline: None,
            push_constants: None,
        }
    }

    pub fn create_pipeline_info(&self) -> ComputePipelineCreateInfo {
        ComputePipelineCreateInfo {
            shader_module: self.shader.clone().expect("Shader not compiled"),
        }
    }
}

impl RenderPass for AverageLuminancePass {
    fn name(&self) -> &str {
        "AverageLuminancePass"
    }

    fn domain(&self) -> PassDomain {
        PassDomain::Compute
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
                .compile_file("average_luminance", ShaderTarget::Spirv, &[shader_dir])
                .expect("Failed to compile AverageLuminance shader"),
        );

        // 2. Create Pipeline
        let info = self.create_pipeline_info();
        self.pipeline = Some(backend.create_compute_pipeline(&info));
    }

    fn record(&mut self, builder: &mut PassBuilder) {
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

        // Read histogram, read/write exposure
        builder.read_buffer(self.histogram_buffer, ResourceUsage::SHADER_READ);
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
            .expect("AverageLuminancePass pipeline not initialized");
        ctx.bind_pipeline_raw(pipeline);
        if let Some(constants) = self.push_constants {
            ctx.push_constant_data(ShaderStageFlags::Compute, 0, &constants);

            // Only 1 workgroup needed to process the 256-bin histogram
            ctx.dispatch(1, 1, 1);
        }
    }
}
