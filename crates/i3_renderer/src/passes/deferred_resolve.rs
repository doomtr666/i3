use i3_gfx::prelude::*;
use std::sync::Arc;

use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DeferredResolvePushConstants {
    pub inv_view_proj: glm::Mat4,
    pub inv_projection: glm::Mat4,
    pub camera_pos: glm::Vec3,
    pub near_plane: f32,
    pub grid_size: [u32; 3],
    pub far_plane: f32,
    pub screen_dimensions: [f32; 2],
    pub debug_mode: u32,
    pub _pad: u32,
}

pub struct DeferredResolvePass {
    pub sampler: SamplerHandle,
    tlas_handle: AccelerationStructureHandle,

    // Resolved handles (updated in declare)
    hdr_target: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive: ImageHandle,
    depth_buffer: ImageHandle,
    lights: BufferHandle,
    cluster_grid: BufferHandle,
    cluster_light_indices: BufferHandle,
    exposure_buffer: BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
}

impl DeferredResolvePass {
    pub fn new(sampler: SamplerHandle) -> Self {
        Self {
            sampler,
            hdr_target: ImageHandle::INVALID,
            gbuffer_albedo: ImageHandle::INVALID,
            gbuffer_normal: ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            gbuffer_emissive: ImageHandle::INVALID,
            depth_buffer: ImageHandle::INVALID,
            lights: BufferHandle::INVALID,
            cluster_grid: BufferHandle::INVALID,
            cluster_light_indices: BufferHandle::INVALID,
            exposure_buffer: BufferHandle::INVALID,
            tlas_handle: AccelerationStructureHandle::INVALID,
            pipeline: None,
        }
    }
}

impl RenderPass for DeferredResolvePass {
    fn name(&self) -> &str {
        "DeferredResolvePass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("deferred_resolve")
            .wait_loaded()
        {
            let state = asset
                .state
                .as_ref()
                .expect("DeferredResolve asset missing state");
            self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                state,
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // Resolve target handles by name
        self.hdr_target = builder.resolve_image("HDR_Target");
        self.gbuffer_albedo = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive = builder.resolve_image("GBuffer_Emissive");
        self.depth_buffer = builder.resolve_image("DepthBuffer");

        self.lights = builder.resolve_buffer("LightBuffer");
        self.cluster_grid = builder.resolve_buffer("ClusterGrid");
        self.cluster_light_indices = builder.resolve_buffer("ClusterLightIndices");
        self.exposure_buffer = builder.read_buffer_history("ExposureBuffer");

        // Resolve TLAS from symbol table (published by TlasRebuildPass::declare()).
        self.tlas_handle = builder
            .try_resolve_acceleration_structure("TLAS")
            .unwrap_or(AccelerationStructureHandle::INVALID);
        if self.tlas_handle != AccelerationStructureHandle::INVALID {
            builder.read_acceleration_structure(self.tlas_handle, ResourceUsage::SHADER_READ);
        }

        // Read GBuffers and buffers
        builder.read_image(self.gbuffer_albedo, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_normal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_roughmetal, ResourceUsage::SHADER_READ);
        builder.read_image(self.gbuffer_emissive, ResourceUsage::SHADER_READ);
        builder.read_image(self.depth_buffer, ResourceUsage::SHADER_READ);

        builder.read_buffer(self.lights, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.cluster_grid, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.cluster_light_indices, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.exposure_buffer, ResourceUsage::SHADER_READ);

        // Write to HDR target
        builder.write_image(self.hdr_target, ResourceUsage::COLOR_ATTACHMENT);

        builder.descriptor_set(0, |d| {
            d.combined_image_sampler(
                self.gbuffer_albedo,
                DescriptorImageLayout::ShaderReadOnlyOptimal,
                self.sampler,
            );
            d.combined_image_sampler(
                self.gbuffer_normal,
                DescriptorImageLayout::ShaderReadOnlyOptimal,
                self.sampler,
            );
            d.combined_image_sampler(
                self.gbuffer_roughmetal,
                DescriptorImageLayout::ShaderReadOnlyOptimal,
                self.sampler,
            );
            d.combined_image_sampler(
                self.gbuffer_emissive,
                DescriptorImageLayout::ShaderReadOnlyOptimal,
                self.sampler,
            );
            d.combined_image_sampler(
                self.depth_buffer,
                DescriptorImageLayout::ShaderReadOnlyOptimal,
                self.sampler,
            );
            d.storage_buffer(self.lights);
            d.storage_buffer(self.cluster_grid);
            d.storage_buffer(self.cluster_light_indices);
            d.storage_buffer(self.exposure_buffer);

            if self.tlas_handle != AccelerationStructureHandle::INVALID {
                d.bind(9).acceleration_structure(self.tlas_handle);
            }
        });
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("DeferredResolvePass::execute: pipeline not initialized!");
            return;
        };
        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        let debug_mode = *frame.consume::<u32>("DebugChannel");
        let grid_x = (common.screen_width + 63) / 64;
        let grid_y = (common.screen_height + 63) / 64;
        let grid_size = [grid_x, grid_y, 16u32];
        let push_constants = DeferredResolvePushConstants {
            inv_view_proj: common.inv_view_projection,
            inv_projection: common.inv_projection,
            camera_pos: common.camera_pos,
            near_plane: common.near_plane,
            grid_size,
            far_plane: common.far_plane,
            screen_dimensions: [common.screen_width as f32, common.screen_height as f32],
            debug_mode,
            _pad: 0,
        };
        ctx.bind_pipeline_raw(pipeline);
        ctx.push_constant_data(
            ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
            0,
            &push_constants,
        );
        ctx.draw(3, 0); // Fullscreen triangle
    }
}
