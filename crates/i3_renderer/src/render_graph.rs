use crate::groups::{ClusteringGroup, PostProcessGroup};
use crate::passes::debug_viz::{DebugChannel, DebugVizPass};
use crate::passes::deferred_resolve::DeferredResolvePass;
use crate::passes::gbuffer::{self, DrawCommand, GBufferPass};
use crate::passes::sky::SkyPass;
use crate::scene::SceneProvider;
use i3_gfx::prelude::*;
use std::sync::{Arc, Mutex};

/// Shared data published to the FrameGraph blackboard.
#[derive(Debug, Clone, Copy)]
pub struct CommonData {
    pub view: nalgebra_glm::Mat4,
    pub projection: nalgebra_glm::Mat4,
    pub view_projection: nalgebra_glm::Mat4,
    pub inv_projection: nalgebra_glm::Mat4,
    pub near_plane: f32,
    pub far_plane: f32,
    pub screen_width: u32,
    pub screen_height: u32,
    pub camera_pos: nalgebra_glm::Vec3,
    pub light_count: u32,
}

/// Helper pass to clear a buffer.
struct ClearBufferPass {
    pub name: String,
    pub buffer: BufferHandle,
}

impl RenderPass for ClearBufferPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.write_buffer(self.buffer, ResourceUsage::TRANSFER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        ctx.clear_buffer(self.buffer, 0);
    }
}

/// The default render graph for deferred clustered shading.
///
/// Owns persistent passes and groups. Geometry comes from the SceneProvider.
pub struct DefaultRenderGraph {
    // Persistent Passes
    pub gbuffer_pass: Arc<Mutex<GBufferPass>>,
    pub sky_pass: Arc<Mutex<SkyPass>>,
    pub clustering_group: Arc<Mutex<ClusteringGroup>>,
    pub deferred_resolve_pass: Arc<Mutex<DeferredResolvePass>>,
    pub post_process_group: Arc<Mutex<PostProcessGroup>>,
    pub debug_viz_pass: Arc<Mutex<DebugVizPass>>,

    pub sampler: SamplerHandle,
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
    /// Creates the render graph resources (passes and groups).
    pub fn new(_backend: &mut dyn RenderBackend, _config: &RenderConfig) -> Self {
        // Create shared sampler
        let sampler = _backend.create_sampler(&SamplerDesc {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Mock handles for initialization (will be declared properly in record)
        let dummy_image = ImageHandle(SymbolId(0));
        let dummy_buffer = BufferHandle(SymbolId(0));

        let gbuffer_pass = Arc::new(Mutex::new(GBufferPass::new(
            dummy_image,
            dummy_image,
            dummy_image,
            dummy_image,
            dummy_image,
        )));

        let sky_pass = Arc::new(Mutex::new(SkyPass::new(dummy_image, dummy_image)));

        let clustering_group = Arc::new(Mutex::new(ClusteringGroup::new(
            dummy_buffer,
            dummy_buffer,
            dummy_buffer,
            dummy_buffer,
            [1, 1, 1],
        )));

        let deferred_resolve_pass = Arc::new(Mutex::new(DeferredResolvePass::new(
            dummy_image,
            dummy_image,
            dummy_image,
            dummy_image,
            dummy_image,
            dummy_image,
            dummy_buffer,
            dummy_buffer,
            dummy_buffer,
            sampler,
            dummy_buffer,
        )));

        let post_process_group = Arc::new(Mutex::new(PostProcessGroup::new(
            dummy_image,
            dummy_image,
            dummy_buffer,
            dummy_buffer,
            sampler,
        )));

        let debug_viz_pass = Arc::new(Mutex::new(DebugVizPass::new(
            dummy_image,
            dummy_image,
            dummy_image,
            dummy_image,
            dummy_image,
            sampler,
            DebugChannel::Lit,
        )));

        Self {
            gbuffer_pass,
            sky_pass,
            clustering_group,
            deferred_resolve_pass,
            post_process_group,
            debug_viz_pass,
            sampler,
            debug_channel: DebugChannel::Lit,
            gpu_buffers: crate::gpu_buffers::GpuBuffers::allocate(_backend),
            temporal_registry: i3_gfx::graph::temporal::TemporalRegistry::new(),
        }
    }
}

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

impl DefaultRenderGraph {
    pub fn sync(&mut self, backend: &mut dyn RenderBackend, scene: &dyn SceneProvider) {
        self.temporal_registry.advance_frame();

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
            let _ = backend.upload_buffer_slice(self.gpu_buffers.light_buffer, &gpu_lights, 0);
        }
    }

    fn record_clustering(
        &self,
        builder: &mut PassBuilder,
        light_buffer: BufferHandle,
    ) -> (BufferHandle, BufferHandle, BufferHandle, u32, u32, u32) {
        let common = *builder.consume::<CommonData>("Common");

        // Cluster Build Pass
        let grid_x = (common.screen_width + 63) / 64;
        let grid_y = (common.screen_height + 63) / 64;
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
                size: max_clusters * 256 * 4,
                usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
                memory: MemoryType::GpuOnly,
            },
        );

        // Clear the first u32 of cluster_light_indices
        builder.add_pass(ClearBufferPass {
            name: "ClearClusterIndices".to_string(),
            buffer: cluster_light_indices,
        });

        // Update persistent clustering group with current handles and logic
        {
            let group = self.clustering_group.lock().unwrap();

            let mut build = group.cluster_build_pass.lock().unwrap();
            build.cluster_aabbs = cluster_aabbs;
            build.push_constants = crate::passes::cluster_build::ClusterBuildPushConstants {
                inv_projection: common.inv_projection,
                grid_size: [grid_x, grid_y, grid_z],
                near_plane: common.near_plane,
                far_plane: common.far_plane,
                screen_dimensions: [common.screen_width as f32, common.screen_height as f32],
                pad: 0,
            };

            let mut cull = group.light_cull_pass.lock().unwrap();
            cull.cluster_aabbs = cluster_aabbs;
            cull.lights = light_buffer;
            cull.cluster_grid = cluster_grid;
            cull.cluster_light_indices = cluster_light_indices;
            cull.push_constants = crate::passes::light_cull::LightCullPushConstants {
                view_matrix: common.view,
                grid_size: [grid_x, grid_y, grid_z],
                light_count: common.light_count,
            };
        }

        builder.add_pass(self.clustering_group.clone());

        (
            cluster_aabbs,
            cluster_grid,
            cluster_light_indices,
            grid_x,
            grid_y,
            grid_z,
        )
    }

    fn record_gbuffer(
        &self,
        builder: &mut PassBuilder,
    ) -> (
        ImageHandle,
        ImageHandle,
        ImageHandle,
        ImageHandle,
        ImageHandle,
    ) {
        let common = *builder.consume::<CommonData>("Common");

        let albedo = builder.declare_image(
            "GBuffer_Albedo",
            ImageDesc::new(
                common.screen_width,
                common.screen_height,
                Format::R8G8B8A8_SRGB,
            ),
        );
        let normal = builder.declare_image(
            "GBuffer_Normal",
            ImageDesc::new(
                common.screen_width,
                common.screen_height,
                Format::R16G16_SFLOAT,
            ),
        );
        let roughmetal = builder.declare_image(
            "GBuffer_RoughMetal",
            ImageDesc::new(
                common.screen_width,
                common.screen_height,
                Format::R8G8_UNORM,
            ),
        );
        let emissive = builder.declare_image(
            "GBuffer_Emissive",
            ImageDesc::new(
                common.screen_width,
                common.screen_height,
                Format::R11G11B10_UFLOAT,
            ),
        );
        let depth = builder.declare_image(
            "DepthBuffer",
            ImageDesc {
                width: common.screen_width,
                height: common.screen_height,
                depth: 1,
                format: Format::D32_FLOAT,
                mip_levels: 1,
                array_layers: 1,
                usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::SAMPLED,
                view_type: ImageViewType::Type2D,
                swizzle: ComponentMapping::default(),
            },
        );

        {
            let mut pass = self.gbuffer_pass.lock().unwrap();
            pass.depth_buffer = depth;
            pass.gbuffer_albedo = albedo;
            pass.gbuffer_normal = normal;
            pass.gbuffer_roughmetal = roughmetal;
            pass.gbuffer_emissive = emissive;
        }

        builder.add_pass(self.gbuffer_pass.clone());

        (albedo, normal, roughmetal, emissive, depth)
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

        let light_count = scene.light_count().min(1024) as u32;
        let camera_pos = view
            .try_inverse()
            .map(|v| v.column(3).xyz())
            .unwrap_or_else(|| nalgebra_glm::vec3(0.0, 0.0, 0.0));

        let common = CommonData {
            view,
            projection,
            view_projection,
            inv_projection,
            near_plane,
            far_plane,
            screen_width,
            screen_height,
            camera_pos,
            light_count,
        };

        // Extract draw commands from scene
        let draw_commands: Vec<DrawCommand> = scene
            .iter_objects()
            .map(|(_, obj)| {
                let mesh = *scene.mesh(obj.mesh_id);
                DrawCommand {
                    mesh,
                    push_constants: gbuffer::GBufferPushConstants {
                        view_projection,
                        model: obj.world_transform,
                    },
                }
            })
            .collect();

        let channel = self.debug_channel;
        let light_buffer_physical = self.gpu_buffers.light_buffer;

        let (sun_dir, sun_int, sun_col) = scene
            .iter_lights()
            .find(|(_, l)| l.light_type == crate::scene::LightType::Directional)
            .map(|(_, l)| (l.direction, l.intensity, l.color))
            .unwrap_or((
                nalgebra_glm::vec3(0.0, -1.0, 0.0),
                1.0,
                nalgebra_glm::vec3(1.0, 0.9, 0.8),
            ));

        graph.record(move |builder| {
            builder.publish("Common", common);
            builder.publish("GBufferCommands", draw_commands);
            builder.publish("SunDirection", sun_dir);
            builder.publish("SunIntensity", sun_int);
            builder.publish("SunColor", sun_col);
            builder.publish("TimeDelta", dt);

            let backbuffer = builder.acquire_backbuffer(window);
            let light_buffer = builder.import_buffer("LightBuffer", light_buffer_physical);

            // 1. Clustering & Culling
            let (_cluster_aabbs, cluster_grid, cluster_light_indices, grid_x, grid_y, grid_z) =
                self.record_clustering(builder, light_buffer);

            builder.publish("ClusterGridSize", [grid_x, grid_y, grid_z]);
            builder.publish("DebugChannel", channel as u32);

            // 2. GBuffer Generation
            let (
                gbuffer_albedo,
                gbuffer_normal,
                gbuffer_roughmetal,
                gbuffer_emissive,
                depth_buffer,
            ) = self.record_gbuffer(builder);

            let hdr_target = builder.declare_image(
                "HDR_Target",
                ImageDesc::new(screen_width, screen_height, Format::R16G16B16A16_SFLOAT),
            );

            {
                let mut pass = self.sky_pass.lock().unwrap();
                pass.hdr_target = hdr_target;
                pass.depth_buffer = depth_buffer;
            }
            builder.add_pass(self.sky_pass.clone());

            let exposure_buffer = builder.declare_buffer_history(
                "ExposureBuffer",
                BufferDesc {
                    size: 8,
                    usage: BufferUsageFlags::STORAGE_BUFFER,
                    memory: MemoryType::GpuOnly,
                },
            );

            let histogram_buffer = builder.declare_buffer(
                "HistogramBuffer",
                BufferDesc {
                    size: 256 * 4,
                    usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
                    memory: MemoryType::GpuOnly,
                },
            );

            if channel == DebugChannel::Lit
                || channel == DebugChannel::LightDensity
                || channel == DebugChannel::ClusterGrid
            {
                // 4. Deferred Lighting
                let hdr_final = self.record_lighting(
                    builder,
                    hdr_target,
                    exposure_buffer,
                    gbuffer_albedo,
                    gbuffer_normal,
                    gbuffer_roughmetal,
                    gbuffer_emissive,
                    depth_buffer,
                    light_buffer,
                    cluster_grid,
                    cluster_light_indices,
                );

                // Use the exposure buffer from record_lighting?
                // Wait, record_lighting declarations might conflict.
                // I'll update record_lighting to take them as params or resolve them.

                // 5. Post Processing
                self.record_post_process(
                    builder,
                    hdr_final,
                    backbuffer,
                    exposure_buffer,
                    histogram_buffer,
                    dt,
                );
            } else {
                {
                    let mut pass = self.debug_viz_pass.lock().unwrap();
                    pass.backbuffer = backbuffer;
                    pass.gbuffer_albedo = gbuffer_albedo;
                    pass.gbuffer_normal = gbuffer_normal;
                    pass.gbuffer_roughmetal = gbuffer_roughmetal;
                    pass.gbuffer_emissive = gbuffer_emissive;
                    pass.channel = channel;
                }
                builder.add_pass(self.debug_viz_pass.clone());
            }
        });
    }

    fn record_lighting(
        &self,
        builder: &mut PassBuilder,
        hdr_target: ImageHandle,
        exposure_buffer: BufferHandle,
        gbuffer_albedo: ImageHandle,
        gbuffer_normal: ImageHandle,
        gbuffer_roughmetal: ImageHandle,
        gbuffer_emissive: ImageHandle,
        depth_buffer: ImageHandle,
        lights: BufferHandle,
        cluster_grid: BufferHandle,
        cluster_light_indices: BufferHandle,
    ) -> ImageHandle {
        {
            let mut pass = self.deferred_resolve_pass.lock().unwrap();
            pass.hdr_target = hdr_target;
            pass.gbuffer_albedo = gbuffer_albedo;
            pass.gbuffer_normal = gbuffer_normal;
            pass.gbuffer_roughmetal = gbuffer_roughmetal;
            pass.gbuffer_emissive = gbuffer_emissive;
            pass.depth_buffer = depth_buffer;
            pass.lights = lights;
            pass.cluster_grid = cluster_grid;
            pass.cluster_light_indices = cluster_light_indices;
            pass.exposure_buffer = exposure_buffer;
        }

        builder.add_pass(self.deferred_resolve_pass.clone());

        hdr_target
    }

    fn record_post_process(
        &self,
        builder: &mut PassBuilder,
        hdr_target: ImageHandle,
        backbuffer: ImageHandle,
        exposure_buffer: BufferHandle,
        histogram_buffer: BufferHandle,
        _dt: f32,
    ) {
        builder.add_pass(ClearBufferPass {
            name: "ClearHistogram".to_string(),
            buffer: histogram_buffer,
        });

        {
            let group = self.post_process_group.lock().unwrap();

            let mut hist = group.histogram_build_pass.lock().unwrap();
            hist.hdr_image = hdr_target;
            hist.histogram_buffer = histogram_buffer;
            hist.exposure_buffer = exposure_buffer;

            let mut avg = group.average_luminance_pass.lock().unwrap();
            avg.histogram_buffer = histogram_buffer;
            avg.exposure_buffer = exposure_buffer;

            let mut tone = group.tonemap_pass.lock().unwrap();
            tone.backbuffer = backbuffer;
            tone.hdr_target = hdr_target;
            tone.exposure_buffer = exposure_buffer;
        }

        builder.add_pass(self.post_process_group.clone());
    }
}
