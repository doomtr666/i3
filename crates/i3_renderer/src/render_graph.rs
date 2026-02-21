use i3_gfx::prelude::*;

use crate::passes::debug_viz::{self, DebugChannel};
use crate::passes::gbuffer::{self, GBufferPushConstants};
use crate::scene::{Mesh, SceneProvider};

/// A single draw command extracted from the scene for the GBuffer pass.
#[derive(Clone, Copy)]
pub struct DrawCommand {
    pub mesh: Mesh,
    pub push_constants: GBufferPushConstants,
}

/// The default render graph for deferred clustered shading.
///
/// Owns pipelines and samplers. Geometry comes from the SceneProvider.
pub struct DefaultRenderGraph {
    gbuffer_pipeline: PipelineHandle,
    debug_viz_pipeline: PipelineHandle,
    sampler: SamplerHandle,
    pub debug_channel: DebugChannel,
}

/// Configuration for GBuffer target dimensions.
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
}

impl DefaultRenderGraph {
    /// Creates the render graph resources (pipelines only, no geometry).
    pub fn new(
        backend: &mut dyn RenderBackend,
        gbuffer_shader: ShaderModule,
        debug_viz_shader: ShaderModule,
        config: &RenderConfig,
    ) -> Self {
        // Create GBuffer pipeline (4 MRT + depth)
        let gbuffer_pipeline_info = GraphicsPipelineCreateInfo {
            shader_module: gbuffer_shader,
            vertex_input: VertexInputState {
                bindings: vec![VertexInputBinding {
                    binding: 0,
                    stride: std::mem::size_of::<gbuffer::GBufferVertex>() as u32,
                    input_rate: VertexInputRate::Vertex,
                }],
                attributes: vec![
                    VertexInputAttribute {
                        location: 0,
                        binding: 0,
                        format: VertexFormat::Float3,
                        offset: 0,
                    },
                    VertexInputAttribute {
                        location: 1,
                        binding: 0,
                        format: VertexFormat::Float3,
                        offset: 12,
                    },
                    VertexInputAttribute {
                        location: 2,
                        binding: 0,
                        format: VertexFormat::Float3,
                        offset: 24,
                    },
                ],
            },
            render_targets: RenderTargetsInfo {
                color_targets: vec![
                    RenderTargetInfo {
                        format: Format::R8G8B8A8_SRGB,
                        ..Default::default()
                    },
                    RenderTargetInfo {
                        format: Format::R16G16_SFLOAT,
                        ..Default::default()
                    },
                    RenderTargetInfo {
                        format: Format::R8G8_UNORM,
                        ..Default::default()
                    },
                    RenderTargetInfo {
                        format: Format::R11G11B10_UFLOAT,
                        ..Default::default()
                    },
                ],
                depth_stencil_format: Some(Format::D32_FLOAT),
                logic_op: None,
            },
            rasterization_state: RasterizationState {
                cull_mode: CullMode::Back,
                front_face: FrontFace::CounterClockwise,
                ..Default::default()
            },
            depth_stencil_state: DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: true,
                depth_compare_op: CompareOp::Less,
                ..Default::default()
            },
            ..Default::default()
        };

        let backend_gbuffer = backend.create_graphics_pipeline(&gbuffer_pipeline_info);
        let gbuffer_pipeline = PipelineHandle(SymbolId(backend_gbuffer.0));

        // Create debug viz pipeline (fullscreen, no vertex input, 1 color target)
        let debug_viz_pipeline_info = GraphicsPipelineCreateInfo {
            shader_module: debug_viz_shader,
            vertex_input: VertexInputState::default(),
            render_targets: RenderTargetsInfo {
                color_targets: vec![RenderTargetInfo {
                    format: Format::B8G8R8A8_SRGB,
                    ..Default::default()
                }],
                depth_stencil_format: None,
                logic_op: None,
            },
            rasterization_state: RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            },
            ..Default::default()
        };

        let backend_debug = backend.create_graphics_pipeline(&debug_viz_pipeline_info);
        let debug_viz_pipeline = PipelineHandle(SymbolId(backend_debug.0));

        // Nearest-neighbor sampler for pixel-accurate GBuffer inspection
        let sampler = backend.create_sampler(&SamplerDesc {
            min_filter: Filter::Nearest,
            mag_filter: Filter::Nearest,
            ..Default::default()
        });

        let _ = config; // Will be used when we add resize support

        Self {
            gbuffer_pipeline,
            debug_viz_pipeline,
            sampler,
            debug_channel: DebugChannel::Albedo,
        }
    }

    /// Records the full render graph for one frame.
    ///
    /// Extracts draw commands from the scene, then records GBuffer + debug viz passes.
    pub fn record(
        &self,
        graph: &mut FrameGraph,
        window: WindowHandle,
        scene: &dyn SceneProvider,
        view_projection: nalgebra_glm::Mat4,
    ) {
        // Extract draw commands from scene (before the 'static closure)
        let draw_commands: Vec<DrawCommand> = scene
            .iter_objects()
            .map(|(_, obj)| {
                let mesh = *scene.mesh(obj.mesh_id);
                DrawCommand {
                    mesh,
                    push_constants: GBufferPushConstants {
                        view_projection,
                        model: obj.world_transform,
                    },
                }
            })
            .collect();

        let pipeline = self.gbuffer_pipeline;
        let debug_pipeline = self.debug_viz_pipeline;
        let sampler = self.sampler;
        let channel = self.debug_channel;

        graph.record(move |builder| {
            let backbuffer = builder.acquire_backbuffer(window);

            // Declare GBuffer transient targets
            let gbuffer_albedo = builder.declare_image(
                "GBuffer_Albedo",
                ImageDesc {
                    width: 1280,
                    height: 720,
                    depth: 1,
                    format: Format::R8G8B8A8_SRGB,
                    mip_levels: 1,
                    array_layers: 1,
                    usage: ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
                    view_type: ImageViewType::Type2D,
                    swizzle: ComponentMapping::default(),
                },
            );
            let gbuffer_normal = builder.declare_image(
                "GBuffer_Normal",
                ImageDesc {
                    width: 1280,
                    height: 720,
                    depth: 1,
                    format: Format::R16G16_SFLOAT,
                    mip_levels: 1,
                    array_layers: 1,
                    usage: ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
                    view_type: ImageViewType::Type2D,
                    swizzle: ComponentMapping::default(),
                },
            );
            let gbuffer_roughmetal = builder.declare_image(
                "GBuffer_RoughMetal",
                ImageDesc {
                    width: 1280,
                    height: 720,
                    depth: 1,
                    format: Format::R8G8_UNORM,
                    mip_levels: 1,
                    array_layers: 1,
                    usage: ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
                    view_type: ImageViewType::Type2D,
                    swizzle: ComponentMapping::default(),
                },
            );
            let gbuffer_emissive = builder.declare_image(
                "GBuffer_Emissive",
                ImageDesc {
                    width: 1280,
                    height: 720,
                    depth: 1,
                    format: Format::R11G11B10_UFLOAT,
                    mip_levels: 1,
                    array_layers: 1,
                    usage: ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
                    view_type: ImageViewType::Type2D,
                    swizzle: ComponentMapping::default(),
                },
            );
            let depth_buffer = builder.declare_image(
                "DepthBuffer",
                ImageDesc {
                    width: 1280,
                    height: 720,
                    depth: 1,
                    format: Format::D32_FLOAT,
                    mip_levels: 1,
                    array_layers: 1,
                    usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                    view_type: ImageViewType::Type2D,
                    swizzle: ComponentMapping::default(),
                },
            );

            // GBuffer pass — draw all objects from scene
            gbuffer::record_gbuffer_pass(
                builder,
                pipeline,
                depth_buffer,
                gbuffer_albedo,
                gbuffer_normal,
                gbuffer_roughmetal,
                gbuffer_emissive,
                &draw_commands,
            );

            // Debug visualization — fullscreen pass reading GBuffer → backbuffer
            debug_viz::record_debug_viz_pass(
                builder,
                debug_pipeline,
                backbuffer,
                gbuffer_albedo,
                gbuffer_normal,
                gbuffer_roughmetal,
                gbuffer_emissive,
                sampler,
                channel,
            );
        });
    }
}
