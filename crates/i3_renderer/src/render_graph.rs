use crate::constants::div_ceil;
use crate::groups::gtao_group::GtaoGroup;
use crate::groups::rtao_group::RtaoGroup;
use crate::passes::cull::DrawCallGenPass;
use crate::passes::debug_draw::DebugDrawPass;
use crate::passes::debug_viz::{DebugChannel, DebugVizPass};
use crate::passes::deferred_resolve::DeferredResolvePass;
use crate::passes::gbuffer::GBufferPass;
use crate::passes::sky::SkyPass;
use crate::prelude::*;
use i3_egui::prelude::*;
use i3_gfx::prelude::*;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AoMode {
    None,
    Gtao,
    Rtao,
}

/// Indices of global samplers in the bindless sampler array.
#[derive(Debug, Clone, Copy)]
pub struct GlobalSamplerIndices {
    pub linear: u32,
    pub nearest: u32,
    pub aniso: u32,
}

/// Shared data published to the FrameGraph blackboard.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CommonData {
    pub view: nalgebra_glm::Mat4,
    pub projection: nalgebra_glm::Mat4,
    pub view_projection: nalgebra_glm::Mat4,
    pub inv_view: nalgebra_glm::Mat4,
    pub inv_projection: nalgebra_glm::Mat4,
    pub inv_view_projection: nalgebra_glm::Mat4,
    pub prev_view_projection: nalgebra_glm::Mat4,

    pub camera_pos: nalgebra_glm::Vec3,
    pub near_plane: f32,
    pub sun_dir: nalgebra_glm::Vec3,
    pub sun_intensity: f32,
    pub sun_color: nalgebra_glm::Vec3,
    pub far_plane: f32,

    pub screen_width: u32,
    pub screen_height: u32,
    pub light_count: u32,
    pub frame_index: u32,

    pub ibl_lut_index: u32,
    pub ibl_irr_index: u32,
    pub ibl_pref_index: u32,
    pub ibl_env_index: u32,

    pub blue_noise_index: u32,
    pub debug_channel: u32,
    pub ibl_intensity: f32,
    pub dt: f32,
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

/// Trivial pass that declares AO_Resolved and writes 1.0 (AO disabled).
struct AoNonePass {
    ao_resolved: ImageHandle,
    pipeline: Option<i3_gfx::graph::backend::BackendPipeline>,
}

impl AoNonePass {
    fn new() -> Self {
        Self {
            ao_resolved: ImageHandle::INVALID,
            pipeline: None,
        }
    }
}

impl RenderPass for AoNonePass {
    fn name(&self) -> &str {
        "AoNonePass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("ao_none")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let depth = builder.resolve_image("DepthBuffer");
        let desc = builder.get_image_desc(depth);
        let img_desc = ImageDesc {
            width: desc.width,
            height: desc.height,
            depth: 1,
            format: Format::R32_SFLOAT,
            mip_levels: 1,
            array_layers: 1,
            usage: ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
            view_type: ImageViewType::Type2D,
            swizzle: ComponentMapping::default(),
            clear_value: None,
        };
        self.ao_resolved = builder.declare_image_history_output("AO_Resolved", img_desc);
        builder.write_image(self.ao_resolved, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            return;
        };
        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let (w, h) = (common.screen_width, common.screen_height);
        ctx.bind_pipeline_raw(pipeline);
        let ds = ctx.create_descriptor_set(
            pipeline,
            0,
            &[i3_gfx::graph::backend::DescriptorWrite::storage_image(
                0,
                0,
                self.ao_resolved,
                i3_gfx::graph::backend::DescriptorImageLayout::General,
            )],
        );
        ctx.bind_descriptor_set(0, ds);
        ctx.dispatch(div_ceil(w, 8), div_ceil(h, 8), 1);
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
    pub debug_draw_pass: DebugDrawPass,

    pub linear_sampler: SamplerHandle,
    pub nearest_sampler: SamplerHandle,
    pub material_sampler: SamplerHandle,
    pub debug_channel: DebugChannel,
    pub fxaa_enabled: bool,
    pub ao_mode: AoMode,
    pub gtao_group: GtaoGroup,
    pub rtao_group: RtaoGroup,
    ao_none_pass: AoNonePass,
    prev_ao_mode: AoMode,
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
    pub hiz_build_final: crate::passes::hiz_build::HiZBuildPass,
    pub hdr_mip_gen_pass: crate::passes::hdr_mip_gen::HdrMipGenPass,
    pub sssr_sample_pass: crate::passes::sssr::SssrSamplePass,
    pub sssr_bilateral_pass: crate::passes::sssr::SssrBilateralUpsamplePass,
    pub sssr_composite_pass: crate::passes::sssr::SssrCompositePass,
    pub bloom_pass: crate::passes::bloom::BloomPass,

    // Scene data cached during sync() for declare() and dirty checking
    pub scene_mesh_descriptors: Vec<(u32, crate::scene::GpuMeshDescriptor)>,
    pub cached_instances: Vec<crate::scene::GpuInstanceData>,
    pub cached_materials: Vec<(u32, crate::scene::MaterialData)>,

    // Previous frame view-projection — one-frame lag for temporal reprojection.
    prev_view_projection: nalgebra_glm::Mat4,

    pub ibl_images: Vec<i3_gfx::graph::backend::BackendImage>,
    pub ibl_sun: i3_io::ibl::IblSunData,
    pub ibl_indices: IblIndices,
    /// 1×1 white texture always at bindless index 0 — re-registered after each clear.
    white_image: i3_gfx::graph::backend::BackendImage,
    /// 1024×1024 RGBA16_UNORM blue-noise texture — always registered right after white.
    blue_noise_image: i3_gfx::graph::backend::BackendImage,
    /// Bindless index of the blue-noise texture (stable across frames).
    pub blue_noise_index: u32,
    pub sampler_indices: GlobalSamplerIndices,

    pub common_buffer: i3_gfx::graph::backend::BackendBuffer,
    pub frame_index: u32,
    pub last_dt: f32,
}

/// Configuration for GBuffer target dimensions.
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
}

impl DefaultRenderGraph {
    /// Creates the render graph resources (passes and groups).
    pub fn new(_backend: &mut dyn RenderBackend, _config: &RenderConfig) -> Self {
        // 1. Create Physical Samplers
        let linear_sampler = _backend.create_sampler(&SamplerDesc {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            anisotropy: 1,
            ..Default::default()
        });

        let material_sampler = _backend.create_sampler(&SamplerDesc {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            anisotropy: 16,
            ..Default::default()
        });

        let nearest_sampler = _backend.create_sampler(&SamplerDesc {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            ..Default::default()
        });

        // Nearest sampler with transparent-black border (returns 0.0 for out-of-bounds UVs).
        let nearest_border_sampler = _backend.create_sampler(&SamplerDesc {
            address_mode_u: AddressMode::ClampToBorder,
            address_mode_v: AddressMode::ClampToBorder,
            address_mode_w: AddressMode::ClampToBorder,
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            border_color: Some(BorderColor::FloatTransparentBlack),
            ..Default::default()
        });

        // Linear sampler with transparent-black border.
        // Used by the Hi-Z occlusion test so that UV corners landing exactly on a texel
        // boundary don't snap hard to an adjacent texel (nearest-neighbor flicker).
        // Bilinear interpolation of a MAX pyramid is conservative: the blended value is
        // ≤ the true MAX → we under-cull at boundaries but never falsely cull.
        let linear_border_sampler = _backend.create_sampler(&SamplerDesc {
            address_mode_u: AddressMode::ClampToBorder,
            address_mode_v: AddressMode::ClampToBorder,
            address_mode_w: AddressMode::ClampToBorder,
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            border_color: Some(BorderColor::FloatTransparentBlack),
            ..Default::default()
        });

        // 2. Register Samplers in Bindless Set
        let bindless_set = _backend.get_bindless_set_handle();
        _backend.update_bindless_sampler(linear_sampler, 0, bindless_set, 1);
        _backend.update_bindless_sampler(nearest_sampler, 1, bindless_set, 1);
        _backend.update_bindless_sampler(material_sampler, 2, bindless_set, 1);
        _backend.update_bindless_sampler(nearest_border_sampler, 3, bindless_set, 1);
        _backend.update_bindless_sampler(linear_border_sampler, 4, bindless_set, 1);

        let sampler_indices = GlobalSamplerIndices {
            linear: 0,
            nearest: 1,
            aniso: 2,
        };

        // 3. Rule #1: Empty Constructors
        let gbuffer_pass = GBufferPass::new();
        let draw_call_gen_pass = DrawCallGenPass::new();
        let sky_pass = SkyPass::new();
        let clustering_group = ClusteringGroup::new();
        let deferred_resolve_pass = DeferredResolvePass::new();
        let post_process_group = PostProcessGroup::new();
        let debug_viz_pass = DebugVizPass::new();
        let debug_draw_pass = DebugDrawPass::new();

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
        bindless_manager.bindless_set = DescriptorSetHandle(_backend.get_bindless_set_handle());

        // Update the global sampler in the bindless set
        _backend.update_bindless_sampler(
            material_sampler,
            sampler_indices.aniso,
            bindless_manager.bindless_set.0,
            1,
        );

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

        // Placeholder 1×1 RGBA16_UNORM — replaced with the baked 1024×1024 NoiseAsset in init().
        let blue_noise_image = _backend.create_image(&ImageDesc {
            width: 1,
            height: 1,
            depth: 1,
            format: Format::R16G16B16A16_UNORM,
            usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
            mip_levels: 1,
            array_layers: 1,
            view_type: ImageViewType::Type2D,
            swizzle: Default::default(),
            clear_value: None,
        });
        _backend
            .upload_image(blue_noise_image, &[0u8; 8], 0, 0, 1, 1, 0, 0)
            .unwrap();
        let blue_noise_index =
            bindless_manager.register_physical_texture(_backend, blue_noise_image);

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
        let hiz_build_final = crate::passes::hiz_build::HiZBuildPass::new_final();
        let hdr_mip_gen_pass = crate::passes::hdr_mip_gen::HdrMipGenPass::new();
        let mut sssr_sample_pass = crate::passes::sssr::SssrSamplePass::new();
        sssr_sample_pass.blue_noise_index = blue_noise_index;
        let sssr_bilateral_pass = crate::passes::sssr::SssrBilateralUpsamplePass::new();
        let sssr_composite_pass = crate::passes::sssr::SssrCompositePass::new();
        let bloom_pass = crate::passes::bloom::BloomPass::new();
        let mut gtao_group = GtaoGroup::new();
        gtao_group.gtao_pass.blue_noise_index = blue_noise_index;

        let mut this = Self {
            graph,
            gbuffer_pass,
            draw_call_gen_pass,
            sky_pass,
            sync_group,
            clustering_group,
            deferred_resolve_pass,
            post_process_group,
            debug_viz_pass,
            debug_draw_pass,
            linear_sampler,
            material_sampler,
            nearest_sampler,
            debug_channel: DebugChannel::Lit,
            fxaa_enabled: true,
            ao_mode: AoMode::Gtao,
            gtao_group,
            rtao_group: RtaoGroup::new(),
            ao_none_pass: AoNonePass::new(),
            prev_ao_mode: AoMode::None,
            light_buffer,
            temporal_registry: i3_gfx::graph::temporal::TemporalRegistry::new(),
            bindless_manager,
            sampler_indices,
            ui: None,
            accel_struct_system,
            blas_update_pass,
            tlas_rebuild_pass,
            hiz_build_final,
            hdr_mip_gen_pass,
            sssr_sample_pass,
            sssr_bilateral_pass,
            sssr_composite_pass,
            bloom_pass,
            scene_mesh_descriptors: Vec::new(),
            cached_instances: Vec::new(),
            cached_materials: Vec::new(),

            prev_view_projection: nalgebra_glm::identity(),

            ibl_images: Vec::new(),
            ibl_sun: i3_io::ibl::IblSunData {
                direction: [0.0, 1.0, 0.0],
                intensity: 1.0,
                color: [0.0; 3],
                _pad: 0.0,
            },
            ibl_indices: IblIndices::default(),
            white_image,
            blue_noise_image,
            blue_noise_index,
            compiled: None,
            last_screen_size: (0, 0),
            last_compiled_debug_channel: DebugChannel::Lit,
            graph_dirty: true,
            compiled_had_staging: false,

            common_buffer: _backend.create_buffer(&BufferDesc {
                size: std::mem::size_of::<crate::frame_constant_data::GpuCommonData>() as u64,
                usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
                memory: MemoryType::GpuOnly,
            }),
            frame_index: 0,
            last_dt: 0.016,
        };

        // 4. Initialization: Run pass-specific initialization logic
        this.graph.publish(
            "BindlessSet",
            DescriptorSetHandle(_backend.get_bindless_set_handle()),
        );

        this
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
        self.bindless_manager
            .register_physical_texture(backend, self.white_image);

        // Indices 1-4 — IBL textures (lut, irr, pref, env) if loaded
        if self.ibl_images.len() == 4 {
            let lut_index = self
                .bindless_manager
                .register_physical_texture(backend, self.ibl_images[0]);
            let irr_index = self
                .bindless_manager
                .register_physical_texture(backend, self.ibl_images[1]);
            let pref_index = self
                .bindless_manager
                .register_physical_texture(backend, self.ibl_images[2]);
            let env_index = self
                .bindless_manager
                .register_physical_texture(backend, self.ibl_images[3]);
            // Preserve the existing intensity_scale (HDR→physical calibration factor, default 1.0).
            // Do NOT use ibl_sun.intensity here — that's sun radiance, not ambient scale.
            let scale = self.ibl_indices.intensity_scale;
            self.ibl_indices = IblIndices {
                lut_index,
                irr_index,
                pref_index,
                env_index,
                intensity_scale: scale,
            };
        }

        // Blue-noise texture — always re-registered after IBL so its index stays stable.
        self.blue_noise_index = self
            .bindless_manager
            .register_physical_texture(backend, self.blue_noise_image);
        self.sssr_sample_pass.blue_noise_index = self.blue_noise_index;
        self.gtao_group.gtao_pass.blue_noise_index = self.blue_noise_index;
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
            .init_pass_direct(&mut self.hiz_build_final, backend);
        self.graph
            .init_pass_direct(&mut self.hdr_mip_gen_pass, backend);
        self.graph
            .init_pass_direct(&mut self.draw_call_gen_pass, backend);
        self.graph.init_pass_direct(&mut self.sky_pass, backend);
        self.graph
            .init_pass_direct(&mut self.clustering_group, backend);
        self.graph
            .init_pass_direct(&mut self.tlas_rebuild_pass, backend);
        self.graph.init_pass_direct(&mut self.ao_none_pass, backend);
        self.graph.init_pass_direct(&mut self.gtao_group, backend);
        self.graph.init_pass_direct(&mut self.rtao_group, backend);
        self.graph
            .init_pass_direct(&mut self.sssr_sample_pass, backend);
        self.graph
            .init_pass_direct(&mut self.sssr_bilateral_pass, backend);
        self.graph
            .init_pass_direct(&mut self.sssr_composite_pass, backend);
        self.graph.init_pass_direct(&mut self.bloom_pass, backend);
        self.graph
            .init_pass_direct(&mut self.deferred_resolve_pass, backend);
        self.graph
            .init_pass_direct(&mut self.post_process_group, backend);
        self.graph
            .init_pass_direct(&mut self.debug_viz_pass, backend);
        self.graph
            .init_pass_direct(&mut self.debug_draw_pass, backend);

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

            tracing::debug!(
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

        // Load baked 1024×1024 RGBA16_UNORM blue-noise texture from the system bundle.
        if let Ok(noise) = loader
            .load::<i3_io::noise_asset::NoiseAsset>("white_noise")
            .wait_loaded()
        {
            let n = noise.header.width;
            let img = backend.create_image(&ImageDesc {
                width: n,
                height: n,
                depth: 1,
                format: Format::R16G16B16A16_UNORM,
                usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
                mip_levels: 1,
                array_layers: 1,
                view_type: ImageViewType::Type2D,
                swizzle: Default::default(),
                clear_value: None,
            });
            backend
                .upload_image(img, &noise.data, 0, 0, n, n, 0, 0)
                .unwrap();
            self.blue_noise_image = img;
            self.blue_noise_index = self
                .bindless_manager
                .register_physical_texture(backend, img);
            self.sssr_sample_pass.blue_noise_index = self.blue_noise_index;
            self.gtao_group.gtao_pass.blue_noise_index = self.blue_noise_index;
            tracing::info!(
                "Blue noise loaded: {}×{} RGBA16_UNORM, bindless index {}",
                n,
                n,
                self.blue_noise_index
            );
        } else {
            tracing::warn!("Blue noise asset 'blue_noise' not found — using placeholder");
        }
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
    pub fn sync(
        &mut self,
        backend: &mut dyn RenderBackend,
        window: WindowHandle,
        scene: &dyn SceneProvider,
    ) {
        let capacity = backend.swapchain_image_count(window);
        self.temporal_registry.advance_frame(capacity);

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

        let ibl_sun_active = !self.ibl_images.is_empty();
        let mut gpu_lights = Vec::with_capacity(crate::constants::MAX_LIGHTS as usize);

        // When IBL is loaded, inject its extracted sun as the sole directional light.
        if ibl_sun_active {
            let s = &self.ibl_sun;
            gpu_lights.push(GpuLightData {
                position: [0.0, 0.0, 0.0],
                radius: 0.0,
                color: s.color,
                intensity: s.intensity,
                direction: s.direction,
                light_type: 1, // Directional
            });
        }

        for (_, light) in scene.iter_lights() {
            if gpu_lights.len() >= crate::constants::MAX_LIGHTS as usize {
                break;
            }
            // Skip scene directional lights when IBL sun takes over.
            if ibl_sun_active && light.light_type == crate::scene::LightType::Directional {
                continue;
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
        self.last_dt = dt;
        self.sync(backend, window, scene);

        // 2. Build CommonData
        let view_projection = projection * view;

        let inv_view = view.try_inverse().unwrap_or_else(nalgebra_glm::identity);
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

        let (sun_dir, sun_int, sun_col) = if !self.ibl_images.is_empty() {
            (
                nalgebra_glm::make_vec3(&self.ibl_sun.direction),
                self.ibl_sun.intensity,
                nalgebra_glm::make_vec3(&self.ibl_sun.color),
            )
        } else {
            (
                nalgebra_glm::vec3(0.0, -1.0, 0.0),
                1.0,
                nalgebra_glm::vec3(1.0, 1.0, 1.0),
            )
        };

        let common = CommonData {
            view,
            projection,
            view_projection,
            inv_view,
            inv_projection,
            inv_view_projection,
            prev_view_projection: self.prev_view_projection,
            near_plane,
            far_plane,
            screen_width,
            screen_height,
            camera_pos,
            light_count,
            frame_index: self.frame_index,
            dt,
            sun_dir,
            sun_intensity: sun_int,
            sun_color: sun_col,
            ibl_lut_index: self.ibl_indices.lut_index,
            ibl_irr_index: self.ibl_indices.irr_index,
            ibl_pref_index: self.ibl_indices.pref_index,
            ibl_env_index: self.ibl_indices.env_index,
            ibl_intensity: self.ibl_indices.intensity_scale,
            debug_channel: self.debug_channel as u32,
            blue_noise_index: self.blue_noise_index,
        };

        // Update GPU-side CommonData
        let gpu_common = crate::frame_constant_data::GpuCommonData {
            view: common.view.into(),
            projection: common.projection.into(),
            view_projection: common.view_projection.into(),
            inv_projection: common.inv_projection.into(),
            inv_view: common.inv_view.into(),
            inv_view_projection: common.inv_view_projection.into(),
            prev_view_projection: common.prev_view_projection.into(),

            camera_pos: [
                common.camera_pos.x,
                common.camera_pos.y,
                common.camera_pos.z,
                common.near_plane,
            ],
            sun_dir: [
                common.sun_dir.x,
                common.sun_dir.y,
                common.sun_dir.z,
                common.sun_intensity,
            ],
            sun_color: [
                common.sun_color.x,
                common.sun_color.y,
                common.sun_color.z,
                common.far_plane,
            ],

            screen_size: [
                common.screen_width,
                common.screen_height,
                common.light_count,
                common.frame_index,
            ],
            ibl_indices: [
                common.ibl_lut_index,
                common.ibl_irr_index,
                common.ibl_pref_index,
                common.ibl_env_index,
            ],
            extra_indices: [common.blue_noise_index, common.debug_channel, 0, 0],
            time_params: [common.dt, common.ibl_intensity, 0.0, 0.0],
        };
        backend
            .upload_buffer_data(self.common_buffer, &gpu_common, 0)
            .unwrap();

        // When IBL is loaded, its extracted sun overrides any scene directional light.
        let (sun_dir, sun_int, sun_col) = if !self.ibl_images.is_empty() {
            (
                nalgebra_glm::make_vec3(&self.ibl_sun.direction),
                self.ibl_sun.intensity,
                nalgebra_glm::make_vec3(&self.ibl_sun.color),
            )
        } else {
            let l = scene.sun();
            (l.direction, l.intensity, l.color)
        };

        // 3. Populate FrameBlackboard for execute() access
        let mut frame_data = i3_gfx::graph::compiler::FrameBlackboard::new();
        frame_data.publish("Common", common);
        frame_data.publish("CommonBuffer", self.common_buffer);
        frame_data.publish("PrevViewProjection", self.prev_view_projection);
        frame_data.publish("SunDirection", sun_dir);
        frame_data.publish("SunIntensity", sun_int);
        frame_data.publish("SunColor", sun_col);
        frame_data.publish("TimeDelta", dt);
        frame_data.publish("DebugChannel", self.debug_channel as u32);
        frame_data.publish("BindlessSet", self.bindless_manager.bindless_set);

        // Update prev VP for the next frame (after publishing this frame's prev)
        self.prev_view_projection = view_projection;

        frame_data.publish("IblIndices", self.ibl_indices);

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
        self.post_process_group.fxaa_pass.enabled = self.fxaa_enabled;

        // AO mode switch → recompile graph (different group added to the graph).
        if self.ao_mode != self.prev_ao_mode {
            self.graph_dirty = true;
            self.prev_ao_mode = self.ao_mode;
        }

        self.rtao_group.rtao_pass.blue_noise_index = self.blue_noise_index;
        self.gtao_group.tick();
        self.rtao_group.tick();
        self.sssr_sample_pass.tick();

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

            // Publish global persistent resources to the graph blackboard
            graph.publish("BindlessSet", self.bindless_manager.bindless_set);

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
        let res = self.compiled.as_mut().unwrap().execute(
            backend,
            &frame_data,
            Some(&mut self.temporal_registry),
        );

        self.frame_index += 1;
        res
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
        let inv_view = view.try_inverse().unwrap_or_else(nalgebra_glm::identity);
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

        let (sun_dir, sun_int, sun_col) = if !self.ibl_images.is_empty() {
            (
                nalgebra_glm::make_vec3(&self.ibl_sun.direction),
                self.ibl_sun.intensity,
                nalgebra_glm::make_vec3(&self.ibl_sun.color),
            )
        } else {
            (
                nalgebra_glm::vec3(0.0, -1.0, 0.0),
                1.0,
                nalgebra_glm::vec3(1.0, 1.0, 1.0),
            )
        };

        let common = CommonData {
            view,
            projection,
            view_projection,
            inv_view,
            inv_projection,
            inv_view_projection,
            prev_view_projection: self.prev_view_projection,
            near_plane,
            far_plane,
            screen_width,
            screen_height,
            camera_pos,
            light_count,
            frame_index: self.frame_index,
            dt: self.last_dt, // or self.dt if available
            sun_dir,
            sun_intensity: sun_int,
            sun_color: sun_col,
            ibl_lut_index: self.ibl_indices.lut_index,
            ibl_irr_index: self.ibl_indices.irr_index,
            ibl_pref_index: self.ibl_indices.pref_index,
            ibl_env_index: self.ibl_indices.env_index,
            ibl_intensity: self.ibl_indices.intensity_scale,
            debug_channel: self.debug_channel as u32,
            blue_noise_index: self.blue_noise_index,
        };

        let channel = self.debug_channel;
        let light_buffer_physical = self.light_buffer;

        let (sun_dir, sun_int, sun_col) = if !self.ibl_images.is_empty() {
            (
                nalgebra_glm::make_vec3(&self.ibl_sun.direction),
                self.ibl_sun.intensity,
                nalgebra_glm::make_vec3(&self.ibl_sun.color),
            )
        } else {
            let l = scene.sun();
            (l.direction, l.intensity, l.color)
        };

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
            builder.import_buffer("CommonBuffer", self.common_buffer);

            // 0. Sync CPU scene delta to GPU (imports MeshDescriptorBuffer, InstanceBuffer, MaterialBuffer)
            builder.add_pass(&mut self.sync_group);
            builder.add_pass(&mut self.blas_update_pass);

            // 1. Frustum cull → DrawCallBuffer
            builder.add_pass(&mut self.draw_call_gen_pass);

            // 1b. TLAS Rebuild
            builder.add_pass(&mut self.tlas_rebuild_pass);

            builder.publish("DebugChannel", channel as u32);

            // 1. Clustering (buffers declared as outputs inside ClusteringGroup)
            builder.add_pass(&mut self.clustering_group);

            // 2. GBuffer (images declared as outputs inside GBufferPass)
            builder.add_pass(&mut self.gbuffer_pass);

            // 2b. Hi-Z Pyramid — Final (DepthBuffer → HiZFinal, for screen-space effects)
            builder.add_pass(&mut self.hiz_build_final);

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

            // 3b. AO — exactly one group active; each group owns AO_Resolved.
            match self.ao_mode {
                AoMode::None => builder.add_pass(&mut self.ao_none_pass),
                AoMode::Gtao => builder.add_pass(&mut self.gtao_group),
                AoMode::Rtao => builder.add_pass(&mut self.rtao_group),
            }

            // Always sync the channel onto the pass before declare() runs.
            self.debug_viz_pass.channel = channel;

            if channel == DebugChannel::Lit
                || channel == DebugChannel::LightDensity
                || channel == DebugChannel::ClusterGrid
                || channel == DebugChannel::SsrRaw
                || channel == DebugChannel::SsrUpsampled
                || channel == DebugChannel::BloomBuffer
            {
                // 4. Deferred Lighting (includes HdrMipsPass + SSR + Bloom)
                self.record_lighting(builder);

                if channel == DebugChannel::Lit
                    || channel == DebugChannel::LightDensity
                    || channel == DebugChannel::ClusterGrid
                {
                    // 5. Post Processing
                    self.record_post_process(builder);
                } else {
                    // SsrRaw / SsrUpsampled / BloomBuffer: show buffer directly, skip tonemap.
                    builder.add_pass(&mut self.debug_viz_pass);
                }
            } else {
                builder.add_pass(&mut self.debug_viz_pass);
            }

            // 6. Debug line overlay (no-op when no lines are queued)
            builder.add_pass(&mut self.debug_draw_pass);

            // 7. Egui UI
            if let Some(ui) = builder.try_consume::<Arc<UiSystem>>("UiSystem") {
                if let Some(egui_pass) = ui.create_pass(backbuffer) {
                    builder.add_owned_pass(egui_pass);
                }
            }

            // 8. Final Presentation
            builder.add_owned_pass(PresentPass { backbuffer });
        });
    }

    fn record_lighting(&mut self, builder: &mut PassBuilder) -> ImageHandle {
        builder.add_pass(&mut self.deferred_resolve_pass);
        // Generate HDR mip chain after deferred resolve, before SSR sampling.
        builder.add_pass(&mut self.hdr_mip_gen_pass);
        builder.add_pass(&mut self.sssr_sample_pass);
        // Bilateral upsample: low-res SSR_Raw → full-res SSR_Upsampled (copy at factor=1).
        builder.add_pass(&mut self.sssr_bilateral_pass);
        // Composite SSR_Upsampled onto HDR_Target (in-place, mip 0).
        builder.add_pass(&mut self.sssr_composite_pass);
        // Bloom — bright-pass filter → downsample → upsample → additive composite.
        builder.add_pass(&mut self.bloom_pass);
        builder.resolve_image("HDR_Target")
    }

    fn record_post_process(&mut self, builder: &mut PassBuilder) {
        // Tonemap always writes to LDR_Target; FXAA always runs (passthrough when disabled).
        builder.add_pass(&mut self.post_process_group);
    }
}
