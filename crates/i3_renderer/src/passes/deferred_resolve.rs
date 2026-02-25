use i3_gfx::prelude::*;
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
    pub pipeline: PipelineHandle,
    pub hdr_target: ImageHandle,
    pub gbuffer_albedo: ImageHandle,
    pub gbuffer_normal: ImageHandle,
    pub gbuffer_roughmetal: ImageHandle,
    pub gbuffer_emissive: ImageHandle,
    pub depth_buffer: ImageHandle,
    pub lights: BufferHandle,
    pub cluster_grid: BufferHandle,
    pub cluster_light_indices: BufferHandle,
    pub sampler: SamplerHandle,
    pub exposure_buffer: BufferHandle,
    pub push_constants: DeferredResolvePushConstants,
}

impl DeferredResolvePass {
    pub fn new(
        pipeline: PipelineHandle,
        hdr_target: ImageHandle,
        gbuffer_albedo: ImageHandle,
        gbuffer_normal: ImageHandle,
        gbuffer_roughmetal: ImageHandle,
        gbuffer_emissive: ImageHandle,
        depth_buffer: ImageHandle,
        lights: BufferHandle,
        cluster_grid: BufferHandle,
        cluster_light_indices: BufferHandle,
        sampler: SamplerHandle,
        exposure_buffer: BufferHandle,
        push_constants: DeferredResolvePushConstants,
    ) -> Self {
        Self {
            pipeline,
            hdr_target,
            gbuffer_albedo,
            gbuffer_normal,
            gbuffer_roughmetal,
            gbuffer_emissive,
            depth_buffer,
            lights,
            cluster_grid,
            cluster_light_indices,
            sampler,
            exposure_buffer,
            push_constants,
        }
    }
}

impl RenderPass for DeferredResolvePass {
    fn name(&self) -> &str {
        "DeferredResolvePass"
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.bind_pipeline(self.pipeline);

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
        ctx.push_constant_data(
            ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
            0,
            &self.push_constants,
        );
        ctx.draw(3, 0); // Fullscreen triangle
    }
}
