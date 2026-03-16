use i3_gfx::prelude::*;
use i3_slang::prelude::*;
use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeferredResolvePushConstants {
    pub inv_view_proj: glm::Mat4,
    pub inv_projection: glm::Mat4,
    pub camera_pos: glm::Vec3,
    pub near_plane: f32,
    pub grid_size: [u32; 3],
    pub far_plane: f32,
    pub screen_dimensions: [f32; 2],
    pub debug_mode: u32,
    pub _pad: u32,
}

/// Deferred resolve pass struct implementing the RenderPass trait.
pub struct DeferredResolvePass {
    pub sampler: SamplerHandle,

    // Resolved handles (updated in record)
    hdr_target: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive: ImageHandle,
    depth_buffer: ImageHandle,
    lights: BufferHandle,
    cluster_grid: BufferHandle,
    cluster_light_indices: BufferHandle,
    exposure_buffer: BufferHandle,

    // Persistence
    shader: Option<ShaderModule>,
    pipeline: Option<BackendPipeline>,
    push_constants: Option<DeferredResolvePushConstants>,
}

impl DeferredResolvePass {
    pub fn new(sampler: SamplerHandle) -> Self {
        let dummy_image = ImageHandle(SymbolId(0));
        let dummy_buffer = BufferHandle(SymbolId(0));
        Self {
            sampler,
            hdr_target: dummy_image,
            gbuffer_albedo: dummy_image,
            gbuffer_normal: dummy_image,
            gbuffer_roughmetal: dummy_image,
            gbuffer_emissive: dummy_image,
            depth_buffer: dummy_image,
            lights: dummy_buffer,
            cluster_grid: dummy_buffer,
            cluster_light_indices: dummy_buffer,
            exposure_buffer: dummy_buffer,
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

        let state = asset.state.as_ref().expect("DeferredResolve asset missing state");
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
                    format: Format::R16G16B16A16_SFLOAT,
                    ..Default::default()
                }],
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

impl RenderPass for DeferredResolvePass {
    fn name(&self) -> &str {
        "DeferredResolvePass"
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
                .compile_file("deferred_resolve", ShaderTarget::Spirv, &[shader_dir])
                .expect("Failed to compile DeferredResolve shader"),
        );

        // 2. Create Pipeline
        let info = self.create_pipeline_info();
        self.pipeline = Some(backend.create_graphics_pipeline(&info));
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        // Resolve target handles by name
        self.hdr_target = builder.resolve_image("HDR_Target");
        self.gbuffer_albedo = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive = builder.resolve_image("GBuffer_Emissive");
        self.depth_buffer = builder.resolve_image("DepthBuffer");

        self.lights = builder.resolve_buffer("LightBuffer");
        self.cluster_grid = builder.resolve_buffer("ClusterGrid");
        self.cluster_light_indices = builder.resolve_buffer("ClusterLightIndices");
        self.exposure_buffer = builder.resolve_buffer("ExposureBuffer");

        let (common, grid_size, debug_mode) = {
            let c = *builder.consume::<crate::render_graph::CommonData>("Common");
            let g = *builder.consume::<[u32; 3]>("ClusterGridSize");
            let d = *builder.consume::<u32>("DebugChannel");
            (c, g, d)
        };

        self.push_constants = Some(DeferredResolvePushConstants {
            inv_view_proj: common.view_projection.try_inverse().unwrap_or_default(),
            inv_projection: common.inv_projection,
            camera_pos: common.camera_pos,
            near_plane: common.near_plane,
            grid_size,
            far_plane: common.far_plane,
            screen_dimensions: [common.screen_width as f32, common.screen_height as f32],
            debug_mode,
            _pad: 0,
        });

        // Read GBuffers and buffers
        builder.read_image(self.gbuffer_albedo, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_emissive, ResourceUsage::SHADER_READ);
        builder.read_image(self.depth_buffer, ResourceUsage::SHADER_READ);

        builder.read_buffer(self.lights, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.cluster_grid, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.cluster_light_indices, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.exposure_buffer, ResourceUsage::SHADER_READ);

        // Write to HDR target
        builder.write_image(self.hdr_target, ResourceUsage::COLOR_ATTACHMENT);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.gbuffer_albedo,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(self.sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.gbuffer_normal,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(self.sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 2,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.gbuffer_roughmetal,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(self.sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 3,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.gbuffer_emissive,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(self.sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 4,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: self.depth_buffer,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(self.sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 5,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.lights,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 6,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.cluster_grid,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 7,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.cluster_light_indices,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 8,
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
            .expect("DeferredResolvePass pipeline not initialized");
        ctx.bind_pipeline_raw(pipeline);

        if let Some(constants) = self.push_constants {
            ctx.push_constant_data(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                &constants,
            );
            ctx.draw(3, 0); // Fullscreen triangle
        }
    }
}
