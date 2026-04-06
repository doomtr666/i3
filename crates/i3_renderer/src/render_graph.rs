use crate::prelude::*;
use crate::passes::cull::DrawCallGenPass;
use crate::passes::debug_viz::{DebugChannel, DebugVizPass};
use crate::passes::deferred_resolve::DeferredResolvePass;
use crate::passes::gbuffer::GBufferPass;
use crate::passes::sky::SkyPass;
use i3_egui::prelude::*;
use i3_gfx::prelude::*;
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
pub struct ClearBufferPass {
    pub name: String,
    pub buffer: BufferHandle,
}

impl RenderPass for ClearBufferPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn init(&mut self, _backend: &mut dyn RenderBackend, _globals: &mut PassBuilder) {}

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.write_buffer(self.buffer, ResourceUsage::TRANSFER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &i3_gfx::graph::compiler::FrameBlackboard) {
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

    fn declare(&mut self, builder: &mut PassBuilder) {
        builder.present_image(self.backbuffer);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &i3_gfx::graph::compiler::FrameBlackboard) {
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
    pub light_buffer: i3_gfx::graph::backend::BackendBuffer,
    pub temporal_registry: i3_gfx::graph::temporal::TemporalRegistry,
    pub bindless_manager: crate::bindless::BindlessManager,
    pub ui: Option<Arc<i3_egui::UiSystem>>,

    pub accel_struct_system: crate::passes::accel_struct::AccelStructSystem,
    pub blas_update_pass: crate::passes::accel_struct::BlasUpdatePass,
    pub tlas_rebuild_pass: crate::passes::accel_struct::TlasRebuildPass,

    // Scene data cached during sync() for declare()
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

        let max_lights: u64 = 1024;
        let light_buffer = _backend.create_buffer(&BufferDesc {
            size: max_lights * std::mem::size_of::<crate::scene::LightData>() as u64,
            usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
            memory: MemoryType::CpuToGpu,
        });
        #[cfg(debug_assertions)]
        _backend.set_buffer_name(light_buffer, "LightBuffer");

        let sync_group = SyncGroup::new();

        let mut bindless_manager = crate::bindless::BindlessManager::new(
            1000, // Capacity for 1000 bindless global textures
            material_sampler,
        );
        bindless_manager.bindless_set = _backend.get_bindless_set_handle();
        
        // Update the global sampler in the bindless set
        _backend.update_bindless_sampler(material_sampler, bindless_manager.bindless_set, 1);

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
            clear_value: None,
        });
        _backend
            .upload_image(white_image, &[255, 255, 255, 255], 0, 0, 1, 1, 0, 0)
            .unwrap();
        bindless_manager.register_physical_texture(_backend, white_image);

        let accel_struct_system = crate::passes::accel_struct::AccelStructSystem::new();
        let blas_update_pass = crate::passes::accel_struct::BlasUpdatePass::new();
        let tlas_rebuild_pass = crate::passes::accel_struct::TlasRebuildPass::new();

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
            light_buffer,
            temporal_registry: i3_gfx::graph::temporal::TemporalRegistry::new(),
            bindless_manager,
            ui: None,
            accel_struct_system,
            blas_update_pass,
            tlas_rebuild_pass,
            scene_mesh_descriptors: Vec::new(),
        }
    }

    /// Resets scene-specific state in the render graph (e.g., for scene switching).
    pub fn clear_scene(&mut self, backend: &mut dyn RenderBackend) {
        self.accel_struct_system.reset(backend);
        self.blas_update_pass.builds.clear();
        self.tlas_rebuild_pass.reset();
        self.bindless_manager.clear();
        self.scene_mesh_descriptors.clear();
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
            (
                local_assets.join("system.i3c"),
                local_assets.join("system.i3b"),
            )
        };

        // Cooperative Asset Loading:
        let loader = self
            .graph
            .try_consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader")
            .cloned()
            .unwrap_or_else(|| {
                let loader = Arc::new(i3_io::asset::AssetLoader::new(Arc::new(
                    i3_io::vfs::Vfs::new(),
                )));
                self.graph.publish("AssetLoader", loader.clone());
                loader
            });

        // Ensure system bundle is mounted
        let vfs = loader.vfs();
        if catalog_path.exists() && blob_path.exists() {
            if let Ok(bundle) = i3_io::vfs::BundleBackend::mount(
                catalog_path.to_str().unwrap(),
                blob_path.to_str().unwrap(),
            ) {
                let _ = vfs.mount(Box::new(bundle));
                tracing::info!(
                    "System bundle mounted from {:?}",
                    catalog_path.parent().unwrap()
                );
            }
        }

        // Initialize all passes directly — no declare() needed, avoids per-frame symbol requirements.
        // Each group's init() recursively calls init() on its children.
        self.graph.init_pass_direct(&mut self.sync_group,            backend);
        self.graph.init_pass_direct(&mut self.blas_update_pass,      backend);
        self.graph.init_pass_direct(&mut self.gbuffer_pass,          backend);
        self.graph.init_pass_direct(&mut self.draw_call_gen_pass,    backend);
        self.graph.init_pass_direct(&mut self.sky_pass,              backend);
        self.graph.init_pass_direct(&mut self.clustering_group,      backend);
        self.graph.init_pass_direct(&mut self.tlas_rebuild_pass,     backend);
        self.graph.init_pass_direct(&mut self.deferred_resolve_pass, backend);
        self.graph.init_pass_direct(&mut self.post_process_group,    backend);
        self.graph.init_pass_direct(&mut self.debug_viz_pass,        backend);

        // Egui pipeline lives in an Arc<Mutex<EguiRenderer>> shared across all per-frame passes.
        // Call init() once on a dummy pass so the pipeline gets loaded into that shared renderer.
        // Resolve ui first (read-only), drop the borrow, then mutably use self.graph.
        let egui_dummy = self.graph
            .try_consume::<Arc<i3_egui::UiSystem>>("UiSystem")
            .cloned()
            .and_then(|ui| ui.create_pass(i3_gfx::graph::types::ImageHandle::INVALID));
        if let Some(mut egui_pass) = egui_dummy {
            self.graph.init_pass_direct(&mut egui_pass, backend);
        }
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

        // Ray Tracing Synchronization
        if backend.capabilities().ray_tracing {
            // 1. Initialize TLAS if missing
            if self.accel_struct_system.tlas.is_none() {
                self.accel_struct_system.tlas = Some(backend.create_tlas(&TlasCreateInfo {
                    max_instances: 262144,
                    flags: AccelStructBuildFlags::PREFER_FAST_TRACE,
                }));
            }

            // 2. Manage BLAS lifecycle — create new BLAS for any mesh not yet in cache
            let mut builds = Vec::new();
            for (id, _) in &self.scene_mesh_descriptors {
                if !self.accel_struct_system.blas_cache.contains_key(id) {
                    let mesh = scene.mesh(*id);
                    let blas = backend.create_blas(&BlasCreateInfo {
                        geometries: vec![BlasGeometryDesc {
                            vertex_buffer: mesh.vertex_buffer,
                            vertex_offset: 0,
                            vertex_count: mesh.vertex_count,
                            vertex_stride: mesh.stride,
                            vertex_format: Format::R32G32B32_SFLOAT,
                            index_buffer: mesh.index_buffer,
                            index_offset: 0,
                            index_count: mesh.index_count,
                            index_type: mesh.index_type,
                        }],
                        flags: AccelStructBuildFlags::PREFER_FAST_TRACE,
                    });
                    self.accel_struct_system.blas_cache.insert(*id, blas);
                    builds.push(blas);
                }
            }
            self.blas_update_pass.builds = builds;

            // 3. Build TLAS instance list and push to the pass
            let mut tlas_instances = Vec::new();
            for inst in scene.iter_instances() {
                if let Some(&blas) = self.accel_struct_system.blas_cache.get(&inst.mesh_idx) {
                    let mut transform = [0.0f32; 12];
                    for i in 0..3 {
                        for j in 0..4 {
                            transform[i * 4 + j] = inst.world_transform[(i, j)];
                        }
                    }
                    tlas_instances.push(TlasInstanceDesc {
                        transform,
                        instance_id: tlas_instances.len() as u32,
                        mask: 0xFF,
                        sbt_offset: 0,
                        flags: 0,
                        blas,
                    });
                }
            }
            self.tlas_rebuild_pass.tlas = self.accel_struct_system.tlas;
            self.tlas_rebuild_pass.instances = tlas_instances;
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
            let _ = backend.upload_buffer_slice(self.light_buffer, &gpu_lights, 0);
        }
    }

    /// All-in-one per-frame entry point.
    ///
    /// Syncs GPU data, builds a `FrameBlackboard` with per-frame CPU data, declares and
    /// executes the render graph. Replaces the caller-side sync/declare/compile/execute loop.
    pub fn render<B: i3_gfx::graph::backend::RenderBackendInternal>(
        &mut self,
        backend: &mut B,
        window: WindowHandle,
        scene: &dyn SceneProvider,
        view: nalgebra_glm::Mat4,
        projection: nalgebra_glm::Mat4,
        near_plane: f32,
        far_plane: f32,
        screen_width: u32,
        screen_height: u32,
        dt: f32,
    ) -> Result<Option<u64>, i3_gfx::graph::types::GraphError> {
        // 1. GPU sync
        self.sync(backend, scene);

        // 2. Build CommonData
        let view_projection = projection * view;
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

        let sun_light = scene.sun();

        // 3. Populate FrameBlackboard for execute() access
        let mut frame_data = i3_gfx::graph::compiler::FrameBlackboard::new();
        frame_data.publish("Common",       common);
        frame_data.publish("SunDirection", sun_light.direction);
        frame_data.publish("SunIntensity", sun_light.intensity);
        frame_data.publish("SunColor",     sun_light.color);
        frame_data.publish("TimeDelta",    dt);
        frame_data.publish("DebugChannel", self.debug_channel as u32);
        frame_data.publish("BindlessSet",  self.bindless_manager.bindless_set);

        // 4. Declare graph (symbol-table path stays for resource declarations)
        let mut graph = FrameGraph::new();
        // Forward global services already published on self.graph
        if let Some(loader) = self.graph.try_consume::<std::sync::Arc<i3_io::asset::AssetLoader>>("AssetLoader").cloned() {
            graph.publish("AssetLoader", loader);
        }
        if let Some(ui) = self.graph.try_consume::<std::sync::Arc<i3_egui::UiSystem>>("UiSystem").cloned() {
            graph.publish("UiSystem", ui);
        }
        self.declare(&mut graph, window, scene, view, projection, near_plane, far_plane, screen_width, screen_height, dt);

        // 5. Compile + execute
        let compiled = graph.compile(&backend.capabilities());
        compiled.execute(backend, &frame_data, Some(&mut self.temporal_registry))
    }

    /// Records the full render graph for one frame.
    ///
    /// Extracts draw commands from the scene, then records GBuffer + debug viz passes.
    pub fn declare(
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
        let view_projection = projection * view;
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
        let light_buffer_physical = self.light_buffer;

        let sun_light = scene.sun();
        let sun_dir = sun_light.direction;
        let sun_int = sun_light.intensity;
        let sun_col = sun_light.color;

        let scene_mesh_descriptors = self.scene_mesh_descriptors.clone();
        let scene_instances: Vec<crate::scene::GpuInstanceData> = scene.iter_instances().collect();
        let scene_materials: Vec<(u32, crate::scene::MaterialData)> = scene
            .iter_materials()
            .map(|(id, data)| (id.0, data.clone()))
            .collect();

        graph.declare(|builder| {
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
            // LightBuffer is owned here; mesh/instance/material/draw buffers are imported by their passes
            builder.import_buffer("LightBuffer", light_buffer_physical);

            // 0. Sync CPU scene delta to GPU (imports MeshDescriptorBuffer, InstanceBuffer, MaterialBuffer)
            builder.add_pass(&mut self.sync_group);
            builder.add_pass(&mut self.blas_update_pass);

            // 1. Draw Call Generation (imports DrawCallBuffer + DrawCountBuffer, adds ClearDrawCount child)
            builder.add_pass(&mut self.draw_call_gen_pass);

            // 1c. TLAS Rebuild
            builder.add_pass(&mut self.tlas_rebuild_pass);

            builder.publish("DebugChannel", channel as u32);

            // 1. Clustering (buffers declared as outputs inside ClusteringGroup)
            builder.add_pass(&mut self.clustering_group);

            // 2. GBuffer (images declared as outputs inside GBufferPass)
            builder.add_pass(&mut self.gbuffer_pass);

            // 3. Sky (HDR_Target declared as output inside SkyPass)
            builder.add_pass(&mut self.sky_pass);

            builder.declare_buffer_history(
                "ExposureBuffer",
                BufferDesc {
                    size: 8,
                    usage: BufferUsageFlags::STORAGE_BUFFER,
                    memory: MemoryType::GpuOnly,
                },
            );

            if channel == DebugChannel::Lit
                || channel == DebugChannel::LightDensity
                || channel == DebugChannel::ClusterGrid
            {
                // 4. Deferred Lighting
                self.record_lighting(builder);

                // 5. Post Processing (HistogramBuffer declared as output inside PostProcessGroup)
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
        // HistogramBuffer declared as output (with clear) inside PostProcessGroup
        builder.add_pass(&mut self.post_process_group);
    }
}
