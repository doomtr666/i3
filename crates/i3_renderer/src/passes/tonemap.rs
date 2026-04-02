use std::sync::Arc;
use i3_gfx::prelude::*;


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
    push_constants: Option<ToneMapPushConstants>,
}

impl TonemapPass {
    pub fn new(sampler: SamplerHandle) -> Self {
        Self {
            backbuffer: ImageHandle::INVALID,
            hdr_target: ImageHandle::INVALID,
            exposure_buffer: BufferHandle::INVALID,
            sampler,
            backbuffer_name: "Backbuffer".to_string(),
            hdr_image_name: "HDR_Target".to_string(),
            pipeline: None,
            push_constants: None,
        }
    }

}

impl RenderPass for TonemapPass {
    fn name(&self) -> &str {
        "ToneMapPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("tonemap").wait_loaded() {
            let state = asset.state.as_ref().expect("Tonemap asset missing state");
            self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                state,
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
        let Some(pipeline) = self.pipeline else {
            tracing::error!("TonemapPass::execute: pipeline not initialized!");
            return;
        };
        ctx.bind_pipeline_raw(pipeline);

        if let Some(constants) = self.push_constants {
            ctx.push_constant_data(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                &constants,
            );
            ctx.draw(3, 0); // Fullscreen triangle

            // This is NO LONGER the final pass because of Egui/Debug UI.
            // Present is now handled by a dedicated PresentPass.
        }
    }
}
