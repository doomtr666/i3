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
    pub sampler: SamplerHandle,
    pub backbuffer_name: String,
    pub hdr_image_name: String,

    // Resolved handles (updated in record)
    backbuffer: ImageHandle,
    hdr_target: ImageHandle,
    exposure_buffer: BufferHandle,

    // Persistence
    shader: Option<ShaderModule>,
    pipeline: Option<BackendPipeline>,
    push_constants: Option<ToneMapPushConstants>,
}

impl TonemapPass {
    pub fn new(sampler: SamplerHandle) -> Self {
        let dummy_image = ImageHandle(SymbolId(0));
        let dummy_buffer = BufferHandle(SymbolId(0));
        Self {
            backbuffer: dummy_image,
            hdr_target: dummy_image,
            exposure_buffer: dummy_buffer,
            sampler,
            backbuffer_name: "Backbuffer".to_string(),
            hdr_image_name: "HDR_Target".to_string(),
            shader: None,
            pipeline: None,
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

        let state = asset.state.as_ref().expect("Tonemap asset missing state");
        self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
            state,
            &asset.reflection_data,
            &asset.bytecode,
        ));
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
        // Resolve target handles by name
        self.backbuffer = builder.resolve_image(&self.backbuffer_name);
        self.hdr_target = builder.resolve_image(&self.hdr_image_name);
        self.exposure_buffer = builder.resolve_buffer("ExposureBuffer");

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
