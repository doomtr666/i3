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
    deferred_resolve_pipeline: PipelineHandle,
    cluster_build_pipeline: PipelineHandle,
    light_cull_pipeline: PipelineHandle,
    histogram_build_pipeline: PipelineHandle,
    average_luminance_pipeline: PipelineHandle,
    tonemap_pipeline: PipelineHandle,
    sky_pipeline: PipelineHandle,
    sampler: SamplerHandle,
    pub debug_channel: DebugChannel,
    pub gpu_buffers: crate::gpu_buffers::GpuBuffers,
    pub temporal_registry: i3_gfx::graph::temporal::TemporalRegistry,
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
        deferred_resolve_shader: ShaderModule,
        cluster_build_shader: ShaderModule,
        light_cull_shader: ShaderModule,
        histogram_build_shader: ShaderModule,
        average_luminance_shader: ShaderModule,
        tonemap_shader: ShaderModule,
        sky_shader: ShaderModule,
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

        let deferred_resolve_pipeline_info = GraphicsPipelineCreateInfo {
            shader_module: deferred_resolve_shader,
            vertex_input: VertexInputState::default(),
            render_targets: RenderTargetsInfo {
                color_targets: vec![RenderTargetInfo {
                    format: Format::R16G16B16A16_SFLOAT,
                    ..Default::default()
                }],
                depth_stencil_format: Some(Format::D32_FLOAT),
                logic_op: None,
            },
            rasterization_state: RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            },
            depth_stencil_state: DepthStencilState {
                depth_test_enable: false,
                depth_write_enable: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let backend_deferred_resolve =
            backend.create_graphics_pipeline(&deferred_resolve_pipeline_info);
        let deferred_resolve_pipeline = PipelineHandle(SymbolId(backend_deferred_resolve.0));

        // Create Sky graphics pipeline
        let sky_pipeline_info = GraphicsPipelineCreateInfo {
            shader_module: sky_shader,
            vertex_input: VertexInputState::default(),
            render_targets: RenderTargetsInfo {
                color_targets: vec![RenderTargetInfo {
                    format: Format::R16G16B16A16_SFLOAT,
                    ..Default::default()
                }],
                depth_stencil_format: Some(Format::D32_FLOAT),
                logic_op: None,
            },
            rasterization_state: RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            },
            depth_stencil_state: DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: false,
                depth_compare_op: CompareOp::Equal,
                ..Default::default()
            },
            ..Default::default()
        };
        let backend_sky = backend.create_graphics_pipeline(&sky_pipeline_info);
        let sky_pipeline = PipelineHandle(SymbolId(backend_sky.0));

        // Create cluster build compute pipeline
        let cluster_build_info = ComputePipelineCreateInfo {
            shader_module: cluster_build_shader,
        };
        let backend_cluster = backend.create_compute_pipeline(&cluster_build_info);
        let cluster_build_pipeline = PipelineHandle(SymbolId(backend_cluster.0));

        // Create light cull compute pipeline
        let light_cull_info = ComputePipelineCreateInfo {
            shader_module: light_cull_shader,
        };
        let backend_light_cull = backend.create_compute_pipeline(&light_cull_info);
        let light_cull_pipeline = PipelineHandle(SymbolId(backend_light_cull.0));

        // Create histogram build compute pipeline
        let histogram_build_info = ComputePipelineCreateInfo {
            shader_module: histogram_build_shader,
        };
        let backend_histogram_build = backend.create_compute_pipeline(&histogram_build_info);
        let histogram_build_pipeline = PipelineHandle(SymbolId(backend_histogram_build.0));

        // Create average luminance compute pipeline
        let average_luminance_info = ComputePipelineCreateInfo {
            shader_module: average_luminance_shader,
        };
        let backend_average_luminance = backend.create_compute_pipeline(&average_luminance_info);
        let average_luminance_pipeline = PipelineHandle(SymbolId(backend_average_luminance.0));

        // Create tonemap graphics pipeline
        let tonemap_pipeline_info = GraphicsPipelineCreateInfo {
            shader_module: tonemap_shader,
            vertex_input: VertexInputState {
                bindings: vec![],
                attributes: vec![],
            },
            render_targets: RenderTargetsInfo {
                color_targets: vec![RenderTargetInfo {
                    format: Format::B8G8R8A8_SRGB,
                    blend: None,
                    write_mask: ColorComponentFlags::RGBA,
                }],
                depth_stencil_format: None,
                logic_op: None,
            },
            rasterization_state: RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            },
            depth_stencil_state: DepthStencilState {
                depth_test_enable: false,
                depth_write_enable: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let backend_tonemap = backend.create_graphics_pipeline(&tonemap_pipeline_info);
        let tonemap_pipeline = PipelineHandle(SymbolId(backend_tonemap.0));

        // Nearest-neighbor sampler for pixel-accurate GBuffer inspection
        let sampler = backend.create_sampler(&SamplerDesc {
            min_filter: Filter::Nearest,
            mag_filter: Filter::Nearest,
            ..Default::default()
        });

        let _ = config; // Will be used when we add resize support
        let gpu_buffers = crate::gpu_buffers::GpuBuffers::allocate(backend);

        Self {
            gbuffer_pipeline,
            debug_viz_pipeline,
            deferred_resolve_pipeline,
            cluster_build_pipeline,
            light_cull_pipeline,
            histogram_build_pipeline,
            average_luminance_pipeline,
            tonemap_pipeline,
            sky_pipeline,
            sampler,
            debug_channel: DebugChannel::Lit,
            gpu_buffers,
            temporal_registry: i3_gfx::graph::temporal::TemporalRegistry::new(),
        }
    }

    /// Synchronizes scene data (lights, materials, etc.) to persistent GPU buffers.
    /// Should be called once per frame before recording the graph.
    pub fn sync(
        &mut self,
        backend: &mut dyn RenderBackend,
        scene: &dyn crate::scene::SceneProvider,
    ) {
        self.temporal_registry.advance_frame();

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct GpuLightData {
            position: [f32; 3],
            radius: f32,
            color: [f32; 3],
            intensity: f32,
            direction: [f32; 3],
            light_type: u32,
        }

        let mut gpu_lights = Vec::with_capacity(1024);
        for (_, light) in scene.iter_lights() {
            if gpu_lights.len() >= 1024 {
                break;
            }
            let light_type = match light.light_type {
                crate::scene::LightType::Point => 0,
                crate::scene::LightType::Directional => 1,
                crate::scene::LightType::Spot => 2,
            };
            gpu_lights.push(GpuLightData {
                position: [light.position.x, light.position.y, light.position.z],
                radius: light.radius,
                color: [light.color.x, light.color.y, light.color.z],
                intensity: light.intensity,
                direction: [light.direction.x, light.direction.y, light.direction.z],
                light_type,
            });
        }

        if !gpu_lights.is_empty() {
            let size = std::mem::size_of::<GpuLightData>() * gpu_lights.len();
            let data =
                unsafe { std::slice::from_raw_parts(gpu_lights.as_ptr() as *const u8, size) };
            let _ = backend.upload_buffer(self.gpu_buffers.light_buffer, data, 0);
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
        view: nalgebra_glm::Mat4,
        projection: nalgebra_glm::Mat4,
        near_plane: f32,
        far_plane: f32,
        screen_width: u32,
        screen_height: u32,
        dt: f32,
    ) {
        let view_projection = projection * view;
        let inv_projection = projection
            .try_inverse()
            .unwrap_or_else(nalgebra_glm::identity);
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
        let deferred_resolve_pipeline = self.deferred_resolve_pipeline;
        let cluster_build_pipeline = self.cluster_build_pipeline;
        let light_cull_pipeline = self.light_cull_pipeline;
        let histogram_build_pipeline = self.histogram_build_pipeline;
        let average_luminance_pipeline = self.average_luminance_pipeline;
        let tonemap_pipeline = self.tonemap_pipeline;
        let sky_pipeline = self.sky_pipeline;
        let sampler = self.sampler;
        let channel = self.debug_channel;
        let light_buffer_physical = self.gpu_buffers.light_buffer;
        // Save the light count from the scene
        let light_count = scene.iter_lights().count().min(1024) as u32;

        let (sun_dir, sun_int) = scene
            .iter_lights()
            .find(|(_, l)| l.light_type == crate::scene::LightType::Directional)
            .map(|(_, l)| (l.direction, l.intensity))
            .unwrap_or((nalgebra_glm::vec3(0.0, -1.0, 0.0), 1.0));

        graph.record(move |builder| {
            let backbuffer = builder.acquire_backbuffer(window);
            let light_buffer = builder.import_buffer("LightBuffer", light_buffer_physical);

            // Cluster Build Pass
            let grid_x = (screen_width + 63) / 64;
            let grid_y = (screen_height + 63) / 64;
            let grid_z = 16;

            let max_clusters = (grid_x * grid_y * grid_z) as u64;
            let cluster_aabbs = builder.declare_buffer(
                "ClusterAABBs",
                BufferDesc {
                    size: max_clusters * 32,
                    usage: BufferUsageFlags::STORAGE_BUFFER,
                    memory: MemoryType::GpuOnly,
                },
            );
            let cluster_grid = builder.declare_buffer(
                "ClusterGrid",
                BufferDesc {
                    size: max_clusters * 8,
                    usage: BufferUsageFlags::STORAGE_BUFFER,
                    memory: MemoryType::GpuOnly,
                },
            );
            let cluster_light_indices = builder.declare_buffer(
                "ClusterLightIndices",
                BufferDesc {
                    size: max_clusters * 64 * 4,
                    usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
                    memory: MemoryType::GpuOnly,
                },
            );

            // Clear the first u32 of cluster_light_indices
            builder.add_node("ClearClusterIndices", move |b| {
                b.write_buffer(cluster_light_indices, ResourceUsage::TRANSFER_WRITE);
                move |ctx| {
                    ctx.clear_buffer(cluster_light_indices, 0);
                }
            });

            // Cluster Build Pass
            let grid_x = (screen_width + 63) / 64;
            let grid_y = (screen_height + 63) / 64;
            let grid_z = 16;

            crate::passes::cluster_build::record_cluster_build_pass(
                builder,
                cluster_build_pipeline,
                cluster_aabbs,
                &crate::passes::cluster_build::ClusterBuildPushConstants {
                    inv_projection,
                    grid_size: [grid_x, grid_y, grid_z],
                    near_plane,
                    far_plane,
                    screen_dimensions: [screen_width as f32, screen_height as f32],
                    pad: 0,
                },
            );

            // Light Cull Pass
            crate::passes::light_cull::record_light_cull_pass(
                builder,
                light_cull_pipeline,
                cluster_aabbs,
                light_buffer,
                cluster_grid,
                cluster_light_indices,
                &crate::passes::light_cull::LightCullPushConstants {
                    view_matrix: view,
                    grid_size: [grid_x, grid_y, grid_z],
                    light_count,
                },
            );

            // Declare GBuffer transient targets
            let gbuffer_albedo = builder.declare_image(
                "GBuffer_Albedo",
                ImageDesc {
                    width: screen_width,
                    height: screen_height,
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
                    width: screen_width,
                    height: screen_height,
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
                    width: screen_width,
                    height: screen_height,
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
                    width: screen_width,
                    height: screen_height,
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
                    width: screen_width,
                    height: screen_height,
                    depth: 1,
                    format: Format::D32_FLOAT,
                    mip_levels: 1,
                    array_layers: 1,
                    usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::SAMPLED,
                    view_type: ImageViewType::Type2D,
                    swizzle: ComponentMapping::default(),
                },
            );

            let camera_pos = view
                .try_inverse()
                .map(|v| v.column(3).xyz())
                .unwrap_or_else(|| nalgebra_glm::vec3(0.0, 0.0, 0.0));

            let hdr_target = builder.declare_image(
                "HDR_Target",
                ImageDesc {
                    width: screen_width,
                    height: screen_height,
                    depth: 1,
                    format: Format::R16G16B16A16_SFLOAT,
                    mip_levels: 1,
                    array_layers: 1,
                    usage: ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
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

            // Sky Pass — draw sky on the background
            crate::passes::sky::record_sky_pass(
                builder,
                sky_pipeline,
                hdr_target,
                depth_buffer,
                &crate::passes::sky::SkyPushConstants {
                    inv_view_proj: view_projection
                        .try_inverse()
                        .unwrap_or_else(nalgebra_glm::identity),
                    camera_pos,
                    _pad0: 0.0,
                    sun_direction: sun_dir,
                    sun_intensity: sun_int,
                },
            );

            let exposure_buffer = builder.declare_buffer_history(
                "ExposureBuffer",
                i3_gfx::graph::types::BufferDesc {
                    size: 8, // 2 f32s: PrevExposure, NewExposure
                    usage: i3_gfx::graph::types::BufferUsageFlags::STORAGE_BUFFER,
                    memory: i3_gfx::graph::types::MemoryType::GpuOnly,
                },
            );

            let histogram_buffer = builder.declare_buffer(
                "HistogramBuffer",
                i3_gfx::graph::types::BufferDesc {
                    size: 256 * 4, // 256 bins * 4 bytes
                    usage: i3_gfx::graph::types::BufferUsageFlags::STORAGE_BUFFER
                        | i3_gfx::graph::types::BufferUsageFlags::TRANSFER_DST,
                    memory: i3_gfx::graph::types::MemoryType::GpuOnly,
                },
            );

            if channel == DebugChannel::Lit
                || channel == DebugChannel::LightDensity
                || channel == DebugChannel::ClusterGrid
            {
                let debug_mode = match channel {
                    DebugChannel::Lit => 0,
                    DebugChannel::LightDensity => 1,
                    DebugChannel::ClusterGrid => 2,
                    _ => 0,
                };
                crate::passes::deferred_resolve::record_deferred_resolve_pass(
                    builder,
                    deferred_resolve_pipeline,
                    hdr_target,
                    gbuffer_albedo,
                    gbuffer_normal,
                    gbuffer_roughmetal,
                    gbuffer_emissive,
                    depth_buffer,
                    light_buffer,
                    cluster_grid,
                    cluster_light_indices,
                    sampler,
                    exposure_buffer,
                    &crate::passes::deferred_resolve::DeferredResolvePushConstants {
                        inv_view_proj: view_projection
                            .try_inverse()
                            .unwrap_or_else(nalgebra_glm::identity),
                        inv_projection,
                        camera_pos,
                        near_plane,
                        grid_size: [grid_x, grid_y, grid_z],
                        far_plane,
                        screen_dimensions: [screen_width as f32, screen_height as f32],
                        debug_mode,
                        _pad: 0,
                    },
                );

                builder.add_node("ClearHistogram", move |b| {
                    b.write_buffer(histogram_buffer, ResourceUsage::TRANSFER_WRITE);
                    move |ctx| {
                        ctx.clear_buffer(histogram_buffer, 0);
                    }
                });

                crate::passes::histogram_build::record_histogram_build_pass(
                    builder,
                    histogram_build_pipeline,
                    hdr_target,
                    histogram_buffer,
                    exposure_buffer,
                    screen_width,
                    screen_height,
                    &crate::passes::histogram_build::HistogramPushConstants {
                        min_log_lum: -10.0,
                        max_log_lum: 10.0,
                        time_delta: dt,
                        pad: 0,
                    },
                );

                crate::passes::average_luminance::record_average_luminance_pass(
                    builder,
                    average_luminance_pipeline,
                    histogram_buffer,
                    exposure_buffer,
                    &crate::passes::average_luminance::AverageLuminancePushConstants {
                        min_log_lum: -10.0,
                        max_log_lum: 10.0,
                        time_delta: dt,
                        adaptation_rate: 2.0,
                        pixel_count: (screen_width * screen_height) as f32,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    },
                );

                crate::passes::tonemap::record_tonemap_pass(
                    builder,
                    tonemap_pipeline,
                    backbuffer,
                    hdr_target,
                    exposure_buffer,
                    sampler,
                    &crate::passes::tonemap::ToneMapPushConstants {
                        debug_mode: 0,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    },
                );
            } else {
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
            }
        });
    }
}
