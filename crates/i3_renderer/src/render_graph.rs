use crate::passes::cull::DrawCallGenPass;
use crate::passes::debug_viz::{DebugChannel, DebugVizPass};
use crate::passes::deferred_resolve::DeferredResolvePass;
use crate::passes::gbuffer::GBufferPass;
use crate::passes::sky::SkyPass;
use crate::prelude::*;
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

/// Indices of IBL textures in the global bindless array.
#[derive(Debug, Clone, Copy)]
pub struct IblIndices {
    pub lut_index: u32,
    pub irr_index: u32,
    pub pref_index: u32,
    pub env_index: u32,
    pub intensity_scale: f32,
}

impl Default for IblIndices {
    fn default() -> Self {
        Self {
            lut_index: !0,
            irr_index: !0,
            pref_index: !0,
            env_index: !0,
            intensity_scale: 1.0,
        }
    }
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

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
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

    fn execute(
        &self,
        ctx: &mut dyn PassContext,
        _frame: &i3_gfx::graph::compiler::FrameBlackboard,
    ) {
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

    // GFX-10: compiled topology cache — rebuilt only on structural events.
    compiled: Option<i3_gfx::graph::compiler::CompiledGraph>,
    last_screen_size: (u32, u32),
    last_compiled_debug_channel: DebugChannel,
    graph_dirty: bool,
    // True if the current compiled graph was built while sync passes were dirty
    // (i.e. it contains transient staging buffer nodes). Force one extra recompile
    // on the first clean frame so the compiled graph has no staging buffers.
    compiled_had_staging: bool,

    pub accel_struct_system: crate::passes::accel_struct::AccelStructSystem,
    pub blas_update_pass: crate::passes::accel_struct::BlasUpdatePass,
    pub tlas_rebuild_pass: crate::passes::accel_struct::TlasRebuildPass,

    // Scene data cached during sync() for declare() and dirty checking
    pub scene_mesh_descriptors: Vec<(u32, crate::scene::GpuMeshDescriptor)>,
    pub cached_instances: Vec<crate::scene::GpuInstanceData>,
    pub cached_materials: Vec<(u32, crate::scene::MaterialData)>,

    pub ibl_images: Vec<i3_gfx::graph::backend::BackendImage>,
    pub ibl_sun: i3_io::ibl::IblSunData,
    pub ibl_indices: IblIndices,
    /// 1×1 white texture always at bindless index 0 — re-registered after each clear.
    white_image: i3_gfx::graph::backend::BackendImage,
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
        let sky_pass = SkyPass::new(linear_sampler);
        let clustering_group = ClusteringGroup::new();
        let deferred_resolve_pass = DeferredResolvePass::new(linear_sampler);
        let post_process_group = PostProcessGroup::new(linear_sampler);
        let debug_viz_pass = DebugVizPass::new(linear_sampler, DebugChannel::Lit);

        let graph = FrameGraph::new();

        let light_buffer = _backend.create_buffer(&BufferDesc {
            size: crate::constants::MAX_LIGHTS
                * std::mem::size_of::<crate::scene::LightData>() as u64,
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

        // NOTE: white_image stored in struct so it can be re-registered after bindless clear.
        // Fallback 1x1 black env texture — always bound to sky descriptor set 1
        // so the slot is never garbage when no IBL is loaded.
        let fallback_env = _backend.create_image(&ImageDesc {
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
            .upload_image(fallback_env, &[0, 0, 0, 255], 0, 0, 1, 1, 0, 0)
            .unwrap();

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
            cached_instances: Vec::new(),
            cached_materials: Vec::new(),

            ibl_images: Vec::new(),
            ibl_sun: i3_io::ibl::IblSunData {
                direction: [0.0, 1.0, 0.0],
                intensity: 1.0,
                color: [0.0; 3],
                _pad: 0.0,
            },
            ibl_indices: IblIndices::default(),
            white_image,
            compiled: None,
            last_screen_size: (0, 0),
            last_compiled_debug_channel: DebugChannel::Lit,
            graph_dirty: true,
            compiled_had_staging: false,
        }
    }

    /// Resets scene-specific state in the render graph (e.g., for scene switching).
    /// Also calls `scene.cleanup_gpu()` to free any GPU buffers owned by the scene.
    pub fn clear_scene(
        &mut self,
        backend: &mut dyn RenderBackend,
        scene: &mut dyn crate::scene::SceneProvider,
    ) {
        scene.cleanup_gpu(backend);
        self.accel_struct_system.reset(backend);
        self.blas_update_pass.builds.clear();
        self.tlas_rebuild_pass.reset();
        self.bindless_manager.clear();
        self.restore_permanent_bindless(backend);
        self.scene_mesh_descriptors.clear();
        // Force recompile: new scene may have different BLAS/mesh counts
        self.graph_dirty = true;
        self.compiled = None;
        self.compiled_had_staging = false;
    }

    /// Re-registers permanent bindless textures (white + IBL) after a bindless clear.
    /// Must be called immediately after `bindless_manager.clear()` to guarantee stable indices.
    fn restore_permanent_bindless(&mut self, backend: &mut dyn RenderBackend) {
        // Index 0 — white fallback (must always be first)
        self.bindless_manager.register_physical_texture(backend, self.white_image);

        // Indices 1-4 — IBL textures (lut, irr, pref, env) if loaded
        if self.ibl_images.len() == 4 {
            let lut_index  = self.bindless_manager.register_physical_texture(backend, self.ibl_images[0]);
            let irr_index  = self.bindless_manager.register_physical_texture(backend, self.ibl_images[1]);
            let pref_index = self.bindless_manager.register_physical_texture(backend, self.ibl_images[2]);
            let env_index  = self.bindless_manager.register_physical_texture(backend, self.ibl_images[3]);
            self.ibl_indices = IblIndices {
                lut_index,
                irr_index,
                pref_index,
                env_index,
                intensity_scale: self.ibl_sun.intensity,
            };
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
        self.graph.init_pass_direct(&mut self.sync_group, backend);
        self.graph
            .init_pass_direct(&mut self.blas_update_pass, backend);
        self.graph.init_pass_direct(&mut self.gbuffer_pass, backend);
        self.graph
            .init_pass_direct(&mut self.draw_call_gen_pass, backend);
        self.graph.init_pass_direct(&mut self.sky_pass, backend);
        self.graph
            .init_pass_direct(&mut self.clustering_group, backend);
        self.graph
            .init_pass_direct(&mut self.tlas_rebuild_pass, backend);
        self.graph
            .init_pass_direct(&mut self.deferred_resolve_pass, backend);
        self.graph
            .init_pass_direct(&mut self.post_process_group, backend);
        self.graph
            .init_pass_direct(&mut self.debug_viz_pass, backend);

        // Egui pipeline lives in an Arc<Mutex<EguiRenderer>> shared across all per-frame passes.
        // Call init() once on a dummy pass so the pipeline gets loaded into that shared renderer.
        // Resolve ui first (read-only), drop the borrow, then mutably use self.graph.
        let egui_dummy = self
            .graph
            .try_consume::<Arc<i3_egui::UiSystem>>("UiSystem")
            .cloned()
            .and_then(|ui| ui.create_pass(i3_gfx::graph::types::ImageHandle::INVALID));
        if let Some(mut egui_pass) = egui_dummy {
            self.graph.init_pass_direct(&mut egui_pass, backend);
        }

        // --- IBL Loading
        let mut ibl_images = Vec::new();
        if let Ok(asset) = loader
            .load::<i3_io::ibl::IblAsset>("horn-koppe_spring_1k")
            .wait_loaded()
        {
            let h = &asset.header;

            let offset = 0;

            // 1. LUT (R16G16_SFLOAT — supports COLOR_ATTACHMENT, no override needed)
            let lut_img = backend.create_image(&ImageDesc::new(
                h.lut_width,
                h.lut_height,
                Format::R16G16_SFLOAT,
            ));
            backend
                .upload_image(
                    lut_img,
                    &asset.data[offset..offset + h.lut_data_size as usize],
                    0,
                    0,
                    h.lut_width,
                    h.lut_height,
                    0,
                    0,
                )
                .unwrap();

            // 2. Irradiance (B10G11R11 — compressed-like, force SAMPLED only)
            let irr_img = {
                let mut desc =
                    ImageDesc::new(h.irr_width, h.irr_height, Format::B10G11R11_UFLOAT_PACK32);
                desc.usage = ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST;
                backend.create_image(&desc)
            };
            let offset = h.lut_data_size as usize;
            backend
                .upload_image(
                    irr_img,
                    &asset.data[offset..offset + h.irr_data_size as usize],
                    0,
                    0,
                    h.irr_width,
                    h.irr_height,
                    0,
                    0,
                )
                .unwrap();

            // 3. Pre-filtered (B10G11R11 multi-mip — force SAMPLED only)
            let mut pref_img_desc =
                ImageDesc::new(h.pref_width, h.pref_height, Format::B10G11R11_UFLOAT_PACK32);
            pref_img_desc.mip_levels = h.pref_mip_levels;
            pref_img_desc.usage = ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST;

            let pref_img = backend.create_image(&pref_img_desc);
            let mut offset = (h.lut_data_size + h.irr_data_size) as usize;
            for m in 0..h.pref_mip_levels {
                let ms = (h.pref_width >> m).max(1);
                let size = ms * ms * 4; // R11G11B10 is 4 bytes
                backend
                    .upload_image(
                        pref_img,
                        &asset.data[offset..offset + size as usize],
                        0,
                        0,
                        ms,
                        ms,
                        m,
                        0,
                    )
                    .unwrap();
                offset += size as usize;
            }

            // 4. Env Equirect (BC6H — compressed, COLOR_ATTACHMENT illegal, force SAMPLED only)
            let env_img = {
                let mut desc = ImageDesc::new(h.env_width, h.env_height, Format::BC6H_UF16);
                desc.usage = ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST;
                backend.create_image(&desc)
            };
            let offset = (h.lut_data_size + h.irr_data_size + h.pref_data_size) as usize;
            backend
                .upload_image(
                    env_img,
                    &asset.data[offset..offset + h.env_data_size as usize],
                    0,
                    0,
                    h.env_width,
                    h.env_height,
                    0,
                    0,
                )
                .unwrap();

            ibl_images = vec![lut_img, irr_img, pref_img, env_img];
            self.ibl_sun = h.sun;

            // Register all 4 IBL textures into the bindless array
            let lut_index = self
                .bindless_manager
                .register_physical_texture(backend, lut_img);
            let irr_index = self
                .bindless_manager
                .register_physical_texture(backend, irr_img);
            let pref_index = self
                .bindless_manager
                .register_physical_texture(backend, pref_img);
            let env_index = self
                .bindless_manager
                .register_physical_texture(backend, env_img);
            self.ibl_indices = IblIndices {
                lut_index,
                irr_index,
                pref_index,
                env_index,
                intensity_scale: h.intensity_scale,
            };

            tracing::info!(
                "IBL loaded: Sun intensity={}, dir={:?}, bindless indices: lut={} irr={} pref={} env={}",
                h.sun.intensity,
                h.sun.direction,
                lut_index,
                irr_index,
                pref_index,
                env_index
            );
        }
        self.ibl_images = ibl_images;
    }

    /// Forces a full redeclare + recompile on the next render() call.
    /// Call this after any structural change not auto-detected by render()
    /// (e.g. UiSystem attach/detach, RT enablement toggle).
    pub fn mark_dirty(&mut self) {
        self.graph_dirty = true;
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
                    max_instances: crate::constants::MAX_INSTANCES as u32,
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

        let mut gpu_lights = Vec::with_capacity(crate::constants::MAX_LIGHTS as usize);
        for (_, light) in scene.iter_lights() {
            if gpu_lights.len() >= crate::constants::MAX_LIGHTS as usize {
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

        // Cache scene data for declare() and dirty-check access in render()
        self.cached_instances = scene.iter_instances().collect();
        self.cached_materials = scene
            .iter_materials()
            .map(|(id, data)| (id.0, data.clone()))
            .collect();
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
        let light_count = scene
            .light_count()
            .min(crate::constants::MAX_LIGHTS as usize) as u32;
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
        frame_data.publish("Common", common);
        frame_data.publish("SunDirection", sun_light.direction);
        frame_data.publish("SunIntensity", sun_light.intensity);
        frame_data.publish("SunColor", sun_light.color);
        frame_data.publish("TimeDelta", dt);
        frame_data.publish("DebugChannel", self.debug_channel as u32);
        frame_data.publish("BindlessSet", self.bindless_manager.bindless_set);

        frame_data.publish("IblIndices", self.ibl_indices);
        frame_data.publish("LinearSampler", self.linear_sampler);

        // 4. Detect structural changes that require recompilation.
        let new_size = (screen_width, screen_height);
        if new_size != self.last_screen_size {
            self.last_screen_size = new_size;
            self.graph_dirty = true;
        }
        if self.debug_channel != self.last_compiled_debug_channel {
            self.last_compiled_debug_channel = self.debug_channel;
            self.graph_dirty = true;
        }

        // 5. Dirty-check sync passes against cached scene data — no declare() side effects.
        // Sync passes add/remove transient staging buffers when dirty, changing graph topology.
        let mesh_dirty = self.scene_mesh_descriptors.len()
            != self
                .sync_group
                .mesh_registry_sync
                .mesh_descriptors_cache
                .len()
            || self
                .scene_mesh_descriptors
                .iter()
                .zip(
                    self.sync_group
                        .mesh_registry_sync
                        .mesh_descriptors_cache
                        .iter(),
                )
                .any(|(a, b)| a != b);

        let inst_dirty = self.cached_instances.len()
            != self.sync_group.instance_sync.instances_cache.len()
            || self
                .cached_instances
                .iter()
                .zip(self.sync_group.instance_sync.instances_cache.iter())
                .any(|(a, b)| a != b);

        let mat_dirty = self.cached_materials.len()
            != self.sync_group.material_sync.materials_cache.len()
            || self
                .cached_materials
                .iter()
                .zip(self.sync_group.material_sync.materials_cache.iter())
                .any(|(a, b)| a != b);

        let sync_dirty = mesh_dirty || inst_dirty || mat_dirty;

        // 6. Declare + compile only when topology must change.
        // Also recompile once on the first clean frame after a staging-dirty compile,
        // so the cached graph no longer contains transient staging buffer nodes.
        // Egui primitives are captured inside EguiPass at declare() time (VB/IB sizing).
        // When there's pending output, declare() must run again to get fresh primitives.
        let egui_dirty = self
            .graph
            .try_consume::<Arc<i3_egui::UiSystem>>("UiSystem")
            .map(|ui| ui.has_pending_output())
            .unwrap_or(false);

        let need_compile = self.graph_dirty
            || sync_dirty
            || egui_dirty
            || self.compiled.is_none()
            || (!sync_dirty && self.compiled_had_staging);

        if need_compile {
            tracing::debug!(
                graph_dirty = self.graph_dirty,
                sync_dirty,
                had_staging = self.compiled_had_staging,
                "Recompiling render graph"
            );
            let mut graph = FrameGraph::new();
            if let Some(loader) = self
                .graph
                .try_consume::<std::sync::Arc<i3_io::asset::AssetLoader>>("AssetLoader")
                .cloned()
            {
                graph.publish("AssetLoader", loader);
            }
            if let Some(ui) = self
                .graph
                .try_consume::<std::sync::Arc<i3_egui::UiSystem>>("UiSystem")
                .cloned()
            {
                graph.publish("UiSystem", ui);
            }
            self.declare(
                &mut graph,
                window,
                scene,
                view,
                projection,
                near_plane,
                far_plane,
                screen_width,
                screen_height,
                dt,
            );
            self.compiled = Some(graph.compile(&backend.capabilities()));
            self.graph_dirty = false;
            self.compiled_had_staging = sync_dirty;
        }

        // 7. Execute the cached compiled graph.
        self.compiled.as_mut().unwrap().execute(
            backend,
            &frame_data,
            Some(&mut self.temporal_registry),
        )
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

        let light_count = scene
            .light_count()
            .min(crate::constants::MAX_LIGHTS as usize) as u32;
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
        let scene_instances = self.cached_instances.clone();
        let scene_materials = self.cached_materials.clone();

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
