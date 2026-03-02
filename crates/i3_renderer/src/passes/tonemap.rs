use i3_gfx::prelude::*;
use i3_slang::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ToneMapPushConstants {
    pub debug_mode: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

/// Tonemap pass struct implementing the RenderPass trait.
pub struct TonemapPass {
    pub backbuffer: ImageHandle,
    pub hdr_target: ImageHandle,
    pub exposure_buffer: BufferHandle,
    pub sampler: SamplerHandle,

    // Persistence
    shader: Option<ShaderModule>,
    pipeline: Option<BackendPipeline>,
    push_constants: Option<ToneMapPushConstants>,
}

impl TonemapPass {
    pub fn new(
        backbuffer: ImageHandle,
        hdr_target: ImageHandle,
        exposure_buffer: BufferHandle,
        sampler: SamplerHandle,
    ) -> Self {
        Self {
            backbuffer,
            hdr_target,
            exposure_buffer,
            sampler,
            shader: None,
            pipeline: None,
            push_constants: None,
        }
    }

    pub fn create_pipeline_info(&self) -> GraphicsPipelineCreateInfo {
        GraphicsPipelineCreateInfo {
            shader_module: self.shader.clone().expect("Shader not compiled"),
            render_targets: RenderTargetsInfo {
                color_targets: vec![RenderTargetInfo {
                    format: Format::B8G8R8A8_SRGB, // Backbuffer format
                    ..Default::default()
                }],
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

impl RenderPass for TonemapPass {
    fn name(&self) -> &str {
        "ToneMapPass"
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
                .compile_file("tonemap", ShaderTarget::Spirv, &[shader_dir])
                .expect("Failed to compile Tonemap shader"),
        );

        // 2. Create Pipeline
        let info = self.create_pipeline_info();
        self.pipeline = Some(backend.create_graphics_pipeline(&info));
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        self.push_constants = Some(ToneMapPushConstants {
            debug_mode: 0,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        });

        // Read HDR target & ExposureBuffer
        builder.read_image(self.hdr_target, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.exposure_buffer, ResourceUsage::SHADER_READ);

        // Write to backbuffer
        builder.write_image(self.backbuffer, ResourceUsage::COLOR_ATTACHMENT);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.hdr_target,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(self.sampler),
                    }),
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
        let pipeline = self.pipeline.expect("TonemapPass pipeline not initialized");
        ctx.bind_pipeline_raw(pipeline);

        if let Some(constants) = self.push_constants {
            ctx.push_constant_data(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                &constants,
            );
            ctx.draw(3, 0); // Fullscreen triangle

            // This is the final pass targeting the backbuffer, so we must present it here.
            ctx.present(self.backbuffer);
        }
    }
}
