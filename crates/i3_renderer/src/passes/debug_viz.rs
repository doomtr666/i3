use i3_gfx::prelude::*;
use std::sync::Arc;

/// Which GBuffer channel to display in the debug visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugChannel {
    Albedo = 0,
    Normal = 1,
    Roughness = 2,
    Metallic = 3,
    Emissive = 4,
    Depth = 5,
    AO = 6,
    SsrResolved = 7,
    BloomBuffer = 8,
    SsrRaw = 9,
    Lit = 10,
    LightDensity = 11,
    ClusterGrid = 12,
    SsrUpsampled = 13,
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
/// Debug visualization pass struct implementing the RenderPass trait.
pub struct DebugVizPass {
    pub channel: DebugChannel,
    pub backbuffer_name: String,

    // Resolved handles (updated in declare)
    backbuffer: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive: ImageHandle,
    gbuffer_depth: ImageHandle,
    ao_resolved:   ImageHandle,
    ssr_resolved:  ImageHandle,
    ssr_raw:       ImageHandle,
    ssr_upsampled: ImageHandle,
    bloom_buf:     ImageHandle,


    // Persistence
    pipeline: Option<BackendPipeline>,
}

impl DebugVizPass {
    pub fn new() -> Self {
        Self {
            backbuffer: ImageHandle::INVALID,
            gbuffer_albedo: ImageHandle::INVALID,
            gbuffer_normal: ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            gbuffer_emissive: ImageHandle::INVALID,
            gbuffer_depth: ImageHandle::INVALID,
            ao_resolved:   ImageHandle::INVALID,
            ssr_resolved:  ImageHandle::INVALID,
            ssr_raw:       ImageHandle::INVALID,
            ssr_upsampled: ImageHandle::INVALID,
            bloom_buf:     ImageHandle::INVALID,

            channel: DebugChannel::Lit,
            backbuffer_name: "Backbuffer".to_string(),
            pipeline: None,
        }
    }
}

impl RenderPass for DebugVizPass {
    fn name(&self) -> &str {
        "DebugVizPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("debug_viz")
            .wait_loaded()
        {
            let state = asset.state.as_ref().expect("DebugViz asset missing state");
            self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                state,
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // Resolve target handles by name
        self.gbuffer_albedo = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive = builder.resolve_image("GBuffer_Emissive");
        self.gbuffer_depth = builder.resolve_image("DepthBuffer");
        self.ao_resolved = builder.resolve_image("AO_Resolved");
        self.backbuffer  = builder.resolve_image(&self.backbuffer_name);

        // SSR_Resolved/SSR_Raw are only declared when the lighting path ran.
        if self.channel == DebugChannel::SsrResolved {
            self.ssr_resolved = builder.resolve_image("SSR_Resolved");
            builder.read_image(self.ssr_resolved, ResourceUsage::SHADER_READ);
        }
        if self.channel == DebugChannel::SsrRaw {
            self.ssr_raw = builder.resolve_image("SSR_Raw");
            builder.read_image(self.ssr_raw, ResourceUsage::SHADER_READ);
        }
        if self.channel == DebugChannel::SsrUpsampled {
            self.ssr_upsampled = builder.resolve_image("SSR_Upsampled");
            builder.read_image(self.ssr_upsampled, ResourceUsage::SHADER_READ);
        }
        // Bloom_Buffer only exists when BloomPass is enabled.
        if self.channel == DebugChannel::BloomBuffer {
            self.bloom_buf = builder.resolve_image("Bloom_Buffer");
            builder.read_image(self.bloom_buf, ResourceUsage::SHADER_READ);
        }


        // Read GBuffer targets
        builder.read_image(self.gbuffer_albedo,    ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal,    ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal,ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_emissive,  ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_depth,     ResourceUsage::SHADER_READ);
        builder.read_image(self.ao_resolved,       ResourceUsage::SHADER_READ);

        // Write to backbuffer, then transition to PresentSrc
        builder.write_image(self.backbuffer, ResourceUsage::COLOR_ATTACHMENT);
        builder.present_image(self.backbuffer);

        // Bindings 6 and 7 must always be written even if the shader won't read them.
        // Use ao_resolved as a harmless dummy when the channel is something else.
        let ssr_handle = if self.channel == DebugChannel::SsrResolved {
            self.ssr_resolved
        } else if self.channel == DebugChannel::SsrRaw {
            self.ssr_raw
        } else if self.channel == DebugChannel::SsrUpsampled {
            self.ssr_upsampled
        } else {
            self.ao_resolved
        };

        let bloom_handle = if self.channel == DebugChannel::BloomBuffer {
            self.bloom_buf
        } else {
            self.ao_resolved
        };

        // Bind GBuffer textures via push descriptors
        builder.descriptor_set(0, |d| {
            d.sampled_image(self.gbuffer_albedo,     DescriptorImageLayout::ShaderReadOnlyOptimal);
            d.sampled_image(self.gbuffer_normal,     DescriptorImageLayout::ShaderReadOnlyOptimal);
            d.sampled_image(self.gbuffer_roughmetal, DescriptorImageLayout::ShaderReadOnlyOptimal);
            d.sampled_image(self.gbuffer_emissive,   DescriptorImageLayout::ShaderReadOnlyOptimal);
            d.sampled_image(self.gbuffer_depth,      DescriptorImageLayout::ShaderReadOnlyOptimal);
            d.sampled_image(self.ao_resolved,        DescriptorImageLayout::ShaderReadOnlyOptimal);
            d.sampled_image(ssr_handle,              DescriptorImageLayout::ShaderReadOnlyOptimal);
            d.sampled_image(bloom_handle,            DescriptorImageLayout::ShaderReadOnlyOptimal);
        });
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("DebugVizPass::execute: pipeline not initialized!");
            return;
        };
        ctx.bind_pipeline_raw(pipeline);
        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");
        ctx.bind_descriptor_set(2, bindless_set);
        let channel = *frame.consume::<u32>("DebugChannel");
        let push = DebugVizPushConstants {
            channel,
            _pad: [0; 3],
        };
        ctx.push_constant_data(
            ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
            0,
            &push,
        );
        ctx.draw(3, 0); // Fullscreen triangle
        ctx.present(self.backbuffer);
    }
}
