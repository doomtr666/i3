use i3_gfx::prelude::*;

/// Which GBuffer channel to display in the debug visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugChannel {
    Albedo,
    Normal,
    Roughness,
    Metallic,
    Emissive,
    Depth,
}

/// Push constants for the debug visualization pass.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DebugVizPushConstants {
    /// Channel selector (maps to DebugChannel enum as u32).
    pub channel: u32,
    /// Padding to align to 4 bytes.
    pub _pad: [u32; 3],
}

/// Records the debug visualization pass into the FrameGraph.
///
/// Draws a fullscreen triangle that samples the selected GBuffer channel
/// and writes to the backbuffer.
pub fn record_debug_viz_pass(
    builder: &mut PassBuilder,
    pipeline: PipelineHandle,
    backbuffer: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive: ImageHandle,
    sampler: SamplerHandle,
    channel: DebugChannel,
) {
    builder.add_node("DebugVizPass", move |sub| {
        sub.bind_pipeline(pipeline);

        // Read GBuffer targets
        sub.read_image(gbuffer_albedo, ResourceUsage::SHADER_READ);
        sub.read_image(gbuffer_normal, ResourceUsage::SHADER_READ);
        sub.read_image(gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        sub.read_image(gbuffer_emissive, ResourceUsage::SHADER_READ);

        // Write to backbuffer
        sub.write_image(backbuffer, ResourceUsage::COLOR_ATTACHMENT);

        // Bind GBuffer textures via push descriptors
        sub.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: gbuffer_albedo,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: gbuffer_normal,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 2,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: gbuffer_roughmetal,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(sampler),
                    }),
                },
                DescriptorWrite {
                    binding: 3,
                    array_element: 0,
                    descriptor_type: BindingType::CombinedImageSampler,
                    buffer_info: None,
                    image_info: Some(DescriptorImageInfo {
                        image: gbuffer_emissive,
                        image_layout: DescriptorImageLayout::ShaderReadOnlyOptimal,
                        sampler: Some(sampler),
                    }),
                },
            ],
        );

        let push = DebugVizPushConstants {
            channel: channel as u32,
            _pad: [0; 3],
        };

        move |ctx: &mut dyn PassContext| {
            let pc_bytes = unsafe {
                std::slice::from_raw_parts(
                    &push as *const DebugVizPushConstants as *const u8,
                    std::mem::size_of::<DebugVizPushConstants>(),
                )
            };
            ctx.push_constants(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                pc_bytes,
            );
            ctx.draw(3, 0); // Fullscreen triangle
            ctx.present(backbuffer);
        }
    });
}
