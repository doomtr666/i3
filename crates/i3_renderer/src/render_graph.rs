use i3_gfx::prelude::*;

use crate::passes::debug_viz::{self, DebugChannel};
use crate::passes::gbuffer::{self, GBufferPushConstants};

/// The default render graph for deferred clustered shading.
///
/// Phase 1: GBuffer with hardcoded cubes + debug visualization.
/// Owns the GPU resources (pipelines, buffers) and records passes into
/// the FrameGraph each frame.
pub struct DefaultRenderGraph {
    gbuffer_pipeline: PipelineHandle,
    debug_viz_pipeline: PipelineHandle,
    cube_vertex_buffer: BackendBuffer,
    cube_index_buffer: BackendBuffer,
    sampler: SamplerHandle,
    pub debug_channel: DebugChannel,
}

/// Configuration for GBuffer target dimensions.
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
}

impl DefaultRenderGraph {
    /// Creates the render graph resources.
    ///
    /// Call once at init time. Creates pipelines and uploads geometry.
    pub fn new(
        backend: &mut dyn RenderBackend,
        gbuffer_shader: ShaderModule,
        debug_viz_shader: ShaderModule,
        config: &RenderConfig,
    ) -> Self {
        // Upload cube geometry
        let (vertices, indices) = gbuffer::generate_cube();

        let vb = backend.create_buffer(&BufferDesc {
            size: (vertices.len() * std::mem::size_of::<gbuffer::GBufferVertex>()) as u64,
            usage: BufferUsageFlags::VERTEX_BUFFER,
            memory: MemoryType::CpuToGpu,
        });
        let vb_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * std::mem::size_of::<gbuffer::GBufferVertex>(),
            )
        };
        backend
            .upload_buffer(vb, vb_bytes, 0)
            .expect("Failed to upload cube vertices");

        let ib = backend.create_buffer(&BufferDesc {
            size: (indices.len() * std::mem::size_of::<u16>()) as u64,
            usage: BufferUsageFlags::INDEX_BUFFER,
            memory: MemoryType::CpuToGpu,
        });
        let ib_bytes = unsafe {
            std::slice::from_raw_parts(
                indices.as_ptr() as *const u8,
                indices.len() * std::mem::size_of::<u16>(),
            )
        };
        backend
            .upload_buffer(ib, ib_bytes, 0)
            .expect("Failed to upload cube indices");

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
            cube_vertex_buffer: vb,
            cube_index_buffer: ib,
            sampler,
            debug_channel: DebugChannel::Albedo,
        }
    }

    /// Records the full render graph for one frame.
    ///
    /// Phase 1: GBuffer pass → debug visualization pass.
    pub fn record(
        &self,
        graph: &mut FrameGraph,
        window: WindowHandle,
        view_projection: nalgebra_glm::Mat4,
    ) {
        let pipeline = self.gbuffer_pipeline;
        let debug_pipeline = self.debug_viz_pipeline;
        let vb = self.cube_vertex_buffer;
        let ib = self.cube_index_buffer;
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

            // GBuffer pass — draw cubes
            let push = GBufferPushConstants {
                view_projection,
                model: nalgebra_glm::identity(),
            };

            gbuffer::record_gbuffer_pass(
                builder,
                pipeline,
                vb,
                ib,
                depth_buffer,
                gbuffer_albedo,
                gbuffer_normal,
                gbuffer_roughmetal,
                gbuffer_emissive,
                push,
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
