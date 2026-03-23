use crate::groups::{ClusteringGroup, PostProcessGroup, sync::SyncGroup};
use crate::passes::debug_viz::{DebugChannel, DebugVizPass};
use crate::passes::deferred_resolve::DeferredResolvePass;
use crate::passes::cull::DrawCallGenPass;
use crate::passes::gbuffer::GBufferPass;
use crate::passes::sky::SkyPass;
use crate::scene::SceneProvider;
use i3_gfx::prelude::*;
use i3_egui::UiSystem;
use std::sync::Arc;

/// Shared data published to the FrameGraph blackboard.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CommonData {
    pub view: nalgebra_glm::Mat4,
    pub projection: nalgebra_glm::Mat4,
    pub view_projection: nalgebra_glm::Mat4,
    pub inv_projection: nalgebra_glm::Mat4,
    pub inv_view_projection: nalgebra_glm::Mat4,
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

    fn init(&mut self, _backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {}

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.write_buffer(self.buffer, ResourceUsage::TRANSFER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        ctx.clear_buffer(self.buffer, 0);
    }
}

/// Helper pass to present the backbuffer.
struct PresentPass {
    pub backbuffer: ImageHandle,
}

impl RenderPass for PresentPass {
    fn name(&self) -> &str {
        "PresentPass"
    }

    fn init(&mut self, _backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {}

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.read_image(self.backbuffer, ResourceUsage::PRESENT);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        ctx.present(self.backbuffer);
    }
}

/// The default render graph for deferred clustered shading.
///
/// Owns persistent passes and groups. Geometry comes from the SceneProvider.
pub struct DefaultRenderGraph {
    pub graph: FrameGraph,
    pub gbuffer_pass: GBufferPass,
    pub draw_call_gen_pass: DrawCallGenPass,
    pub sky_pass: SkyPass,
    pub sync_group: SyncGroup,
    pub clustering_group: ClusteringGroup,
    pub deferred_resolve_pass: DeferredResolvePass,
    pub post_process_group: PostProcessGroup,
    pub debug_viz_pass: DebugVizPass,

    pub linear_sampler: SamplerHandle,
    pub material_sampler: SamplerHandle,
    pub debug_channel: DebugChannel,
    pub gpu_buffers: crate::gpu_buffers::GpuBuffers,
    pub temporal_registry: i3_gfx::graph::temporal::TemporalRegistry,
    pub bindless_manager: crate::bindless::BindlessManager,
    pub ui: Option<Arc<i3_egui::UiSystem>>,

    // Scene data cached during sync() for record()
    pub scene_mesh_descriptors: Vec<(u32, crate::scene::GpuMeshDescriptor)>,
}

/// Configuration for GBuffer target dimensions.
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
}

impl DefaultRenderGraph {
    /// Creates the render graph resources (passes and groups).
    pub fn new(_backend: &mut dyn RenderBackend, _config: &RenderConfig) -> Self {
        // Linear sampler for post-processing (ClampToEdge, no anisotropy)
        let linear_sampler = _backend.create_sampler(&SamplerDesc {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            anisotropy: 1,
            ..Default::default()
        });

        // Material sampler for bindless textures (Repeat, Anisotropy 16)
        let material_sampler = _backend.create_sampler(&SamplerDesc {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            anisotropy: 16,
            ..Default::default()
        });

        let gbuffer_pass = GBufferPass::new();
        let draw_call_gen_pass = DrawCallGenPass::new();
        let sky_pass = SkyPass::new();
        let clustering_group = ClusteringGroup::new();
        let deferred_resolve_pass = DeferredResolvePass::new(linear_sampler);
        let post_process_group = PostProcessGroup::new(linear_sampler);
        let debug_viz_pass = DebugVizPass::new(linear_sampler, DebugChannel::Lit);

        let graph = FrameGraph::new();
        let gpu_buffers = crate::gpu_buffers::GpuBuffers::allocate(_backend);

        let sync_group = SyncGroup::new();

        let mut bindless_manager = crate::bindless::BindlessManager::new(
            1000, // Capacity for 1000 bindless global textures
            material_sampler,
        );
        bindless_manager.bindless_set = _backend.get_bindless_set_handle();

        // Register a default 1x1 white texture at index 0
        let white_image = _backend.create_image(&ImageDesc {
            width: 1,
            height: 1,
            depth: 1,
            format: Format::R8G8B8A8_UNORM,
            usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
            mip_levels: 1,
            array_layers: 1,
            view_type: ImageViewType::Type2D,
            swizzle: Default::default(),
        });
        _backend
            .upload_image(white_image, &[255, 255, 255, 255], 0, 0)
            .unwrap();
        bindless_manager.register_physical_texture(_backend, white_image);

        Self {
            graph,
            gbuffer_pass,
            draw_call_gen_pass,
            sky_pass,
            sync_group,
            clustering_group,
            deferred_resolve_pass,
            post_process_group,
            debug_viz_pass,
            linear_sampler,
            material_sampler,
            debug_channel: DebugChannel::Lit,
            gpu_buffers,
            temporal_registry: i3_gfx::graph::temporal::TemporalRegistry::new(),
            bindless_manager,
            ui: None,
            scene_mesh_descriptors: Vec::new(),
        }
    }

    /// Initializes the render graph: handles cooperative asset loading and pipeline binding.
    /// External services (AssetLoader, UiSystem) should be published to the graph's blackboard before calling this.
    pub fn init(&mut self, backend: &mut dyn RenderBackend) {
        // System bundle discovery
        let exe_path = std::env::current_exe().unwrap();
        let exe_dir = exe_path.parent().unwrap();

        let catalog_path_exe = exe_dir.join("system.i3c");
        let (catalog_path, blob_path) = if catalog_path_exe.exists() {
            (catalog_path_exe, exe_dir.join("system.i3b"))
        } else {
            let local_assets = std::path::Path::new("assets");
            (local_assets.join("system.i3c"), local_assets.join("system.i3b"))
        };

        // Cooperative Asset Loading: 
        let loader = self.graph.try_consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader")
            .cloned()
            .unwrap_or_else(|| {
                let loader = Arc::new(i3_io::asset::AssetLoader::new(Arc::new(i3_io::vfs::Vfs::new())));
                self.graph.publish("AssetLoader", loader.clone());
                loader
            });

        // Ensure system bundle is mounted
        let vfs = loader.vfs();
        if catalog_path.exists() && blob_path.exists() {
            if let Ok(bundle) = i3_io::vfs::BundleBackend::mount(
                catalog_path.to_str().unwrap(), 
                blob_path.to_str().unwrap()
            ) {
                let _ = vfs.mount(Box::new(bundle));
                tracing::info!("System bundle cooperatively mounted from {:?}", catalog_path.parent().unwrap());
            }
        }

        // 1. Setup Phase: Register all potential passes
        let gbuffer_pass = &mut self.gbuffer_pass;
        let draw_call_gen_pass = &mut self.draw_call_gen_pass;
        let sky_pass = &mut self.sky_pass;
        let sync_group = &mut self.sync_group;
        let clustering_group = &mut self.clustering_group;
        let deferred_resolve_pass = &mut self.deferred_resolve_pass;
        let post_process_group = &mut self.post_process_group;
        let debug_viz_pass = &mut self.debug_viz_pass;

        self.graph.setup(|builder| {
            builder.add_pass(gbuffer_pass);
            builder.add_pass(draw_call_gen_pass);
            builder.add_pass(sky_pass);
            builder.add_pass(sync_group);
            builder.add_pass(clustering_group);
            builder.add_pass(deferred_resolve_pass);
            builder.add_pass(post_process_group);
            builder.add_pass(debug_viz_pass);

            // Add a dummy egui pass for initialization
            if let Some(ui) = builder.try_consume::<Arc<UiSystem>>("UiSystem") {
                if let Some(egui_pass) = ui.create_pass(ImageHandle::INVALID) {
                    builder.add_owned_pass(egui_pass);
                }
            }
        });

        // 2. Initialization Phase: Load shaders/pipelines
        self.graph.init_all(backend);
    }

    /// Proxy for publishing a service to the global blackboard.
    pub fn publish<T: 'static + Send + Sync>(&mut self, name: &str, data: T) {
        self.graph.publish(name, data);
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

        // Prefetch mesh descriptors (they need RenderBackend to resolve BDA)
        self.scene_mesh_descriptors = scene.iter_mesh_descriptors(backend).collect();

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

    fn record_clustering(&mut self, builder: &mut PassBuilder) -> (u32, u32, u32) {
        let common = *builder.consume::<CommonData>("Common");

        // Cluster Build Pass
        let grid_x = (common.screen_width + 63) / 64;
        let grid_y = (common.screen_height + 63) / 64;
        let grid_z = 16;

        let max_clusters = (grid_x * grid_y * grid_z) as u64;
        builder.declare_buffer(
            "ClusterAABBs",
            BufferDesc {
                size: max_clusters * 32,
                usage: BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::GpuOnly,
            },
        );
        builder.declare_buffer(
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
        builder.add_owned_pass(ClearBufferPass {
            name: "ClearClusterIndices".to_string(),
            buffer: cluster_light_indices,
        });

        builder.add_pass(&mut self.clustering_group);

        (grid_x, grid_y, grid_z)
    }

    fn record_gbuffer(&mut self, builder: &mut PassBuilder) {
        let common = *builder.consume::<CommonData>("Common");

        builder.declare_image(
            "GBuffer_Albedo",
            ImageDesc::new(
                common.screen_width,
                common.screen_height,
                Format::R8G8B8A8_SRGB,
            ),
        );
        builder.declare_image(
            "GBuffer_Normal",
            ImageDesc::new(
                common.screen_width,
                common.screen_height,
                Format::R16G16_SFLOAT,
            ),
        );
        builder.declare_image(
            "GBuffer_RoughMetal",
            ImageDesc::new(
                common.screen_width,
                common.screen_height,
                Format::R8G8_UNORM,
            ),
        );
        builder.declare_image(
            "GBuffer_Emissive",
            ImageDesc::new(
                common.screen_width,
                common.screen_height,
                Format::R11G11B10_UFLOAT,
            ),
        );
        builder.declare_image(
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

        builder.add_pass(&mut self.gbuffer_pass);
    }

    /// Records the full render graph for one frame.
    ///
    /// Extracts draw commands from the scene, then records GBuffer + debug viz passes.
    pub fn record(
        &mut self,
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
        let view_projection = projection
            * view;
        let inv_projection = projection
            .try_inverse()
            .unwrap_or_else(nalgebra_glm::identity);
        let inv_view_projection = view_projection
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
            inv_view_projection,
            near_plane,
            far_plane,
            screen_width,
            screen_height,
            camera_pos,
            light_count,
        };

        let channel = self.debug_channel;
        let light_buffer_physical = self.gpu_buffers.light_buffer;

        let sun_light = scene.sun();
        let sun_dir = sun_light.direction;
        let sun_int = sun_light.intensity;
        let sun_col = sun_light.color;

        let scene_mesh_descriptors = self.scene_mesh_descriptors.clone();
        let scene_instances: Vec<crate::scene::GpuInstanceData> = scene
            .iter_instances()
            .collect();
        let scene_materials: Vec<(u32, crate::scene::MaterialData)> = scene
            .iter_materials()
            .map(|(id, data)| (id.0, data.clone()))
            .collect();

        graph.record(|builder| {
            builder.publish("Common", common);
            builder.publish("BindlessSet", self.bindless_manager.bindless_set);
            builder.publish("SceneMeshDescriptors", scene_mesh_descriptors);
            builder.publish("SceneInstances", scene_instances);
            builder.publish("SceneMaterials", scene_materials);
            builder.publish("SunDirection", sun_dir);
            builder.publish("SunIntensity", sun_int);
            builder.publish("SunColor", sun_col);
            builder.publish("TimeDelta", dt);

            let backbuffer = builder.acquire_backbuffer(window);
            builder.publish("Backbuffer", backbuffer);
            builder.import_buffer("LightBuffer", light_buffer_physical);

            let mesh_descriptor_buffer_physical = self.gpu_buffers.mesh_descriptor_buffer;
            builder.import_buffer("MeshDescriptorBuffer", mesh_descriptor_buffer_physical);

            let instance_buffer_physical = self.gpu_buffers.instance_buffer;
            builder.import_buffer("InstanceBuffer", instance_buffer_physical);
            
            let draw_call_buffer_physical = self.gpu_buffers.draw_call_buffer;
            builder.import_buffer("DrawCallBuffer", draw_call_buffer_physical);

            let draw_count_buffer_physical = self.gpu_buffers.draw_count_buffer;
            builder.import_buffer("DrawCountBuffer", draw_count_buffer_physical);

            let material_buffer_physical = self.gpu_buffers.material_buffer;
            builder.import_buffer("MaterialBuffer", material_buffer_physical);

            // 0. Sync CPU scene delta to GPU
            builder.add_pass(&mut self.sync_group);

            // 1. Draw Call Generation & Culling
            builder.add_pass(&mut self.draw_call_gen_pass);

            // 1. Clustering & Culling
            let (grid_x, grid_y, grid_z) = self.record_clustering(builder);

            builder.publish("ClusterGridSize", [grid_x, grid_y, grid_z]);
            builder.publish("DebugChannel", channel as u32);

            // 2. GBuffer Generation
            self.record_gbuffer(builder);

            builder.declare_image(
                "HDR_Target",
                ImageDesc::new(screen_width, screen_height, Format::R16G16B16A16_SFLOAT),
            );

            builder.add_pass(&mut self.sky_pass);

            builder.declare_buffer_history(
                "ExposureBuffer",
                BufferDesc {
                    size: 8,
                    usage: BufferUsageFlags::STORAGE_BUFFER,
                    memory: MemoryType::GpuOnly,
                },
            );

            builder.declare_buffer(
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
                self.record_lighting(builder);

                // 5. Post Processing
                self.record_post_process(builder);
            } else {
                builder.add_pass(&mut self.debug_viz_pass);
            }

            // 6. Egui UI
            if let Some(ui) = builder.try_consume::<Arc<UiSystem>>("UiSystem") {
                if let Some(egui_pass) = ui.create_pass(backbuffer) {
                    builder.add_owned_pass(egui_pass);
                }
            }

            // 7. Final Presentation
            builder.add_owned_pass(PresentPass { backbuffer });
        });
    }

    fn record_lighting(&mut self, builder: &mut PassBuilder) -> ImageHandle {
        builder.add_pass(&mut self.deferred_resolve_pass);
        builder.resolve_image("HDR_Target")
    }

    fn record_post_process(&mut self, builder: &mut PassBuilder) {
        let histogram_buffer = builder.resolve_buffer("HistogramBuffer");
        builder.add_owned_pass(ClearBufferPass {
            name: "ClearHistogram".to_string(),
            buffer: histogram_buffer,
        });

        builder.add_pass(&mut self.post_process_group);
    }
}
