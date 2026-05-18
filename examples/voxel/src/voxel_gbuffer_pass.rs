use i3_gfx::prelude::*;
use std::sync::Arc;

pub struct VoxelGBufferPass {
    pub bindless_set: DescriptorSetHandle,

    // Resolved handles (updated in declare)
    depth_buffer:           ImageHandle,
    gbuffer_albedo:         ImageHandle,
    gbuffer_normal:         ImageHandle,
    gbuffer_roughmetal:     ImageHandle,
    gbuffer_emissive:       ImageHandle,
    hiz_mip0:               ImageHandle,

    mesh_descriptor_buffer: BufferHandle,
    instance_buffer:        BufferHandle,
    draw_call_buffer:       BufferHandle,
    draw_count_buffer:      BufferHandle,
    material_buffer:        BufferHandle,
    common_buffer:          BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
}

impl VoxelGBufferPass {
    pub fn new() -> Self {
        Self {
            depth_buffer:           ImageHandle::INVALID,
            gbuffer_albedo:         ImageHandle::INVALID,
            gbuffer_normal:         ImageHandle::INVALID,
            gbuffer_roughmetal:     ImageHandle::INVALID,
            gbuffer_emissive:       ImageHandle::INVALID,
            hiz_mip0:               ImageHandle::INVALID,

            mesh_descriptor_buffer: BufferHandle::INVALID,
            instance_buffer:        BufferHandle::INVALID,
            draw_call_buffer:       BufferHandle::INVALID,
            draw_count_buffer:      BufferHandle::INVALID,
            material_buffer:        BufferHandle::INVALID,
            common_buffer:          BufferHandle::INVALID,

            bindless_set: DescriptorSetHandle(0),
            pipeline:     None,
        }
    }
}

impl RenderPass for VoxelGBufferPass {
    fn name(&self) -> &str {
        "VoxelGBufferPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("voxel_gbuffer").wait_loaded() {
            let state = asset.state.as_ref().expect("Voxel GBuffer asset missing state");
            self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                state,
                &asset.reflection_data,
                &asset.bytecode,
            ));
        } else {
            tracing::error!("VoxelGBufferPass: failed to load voxel_gbuffer pipeline");
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // Resolve image handles
        self.gbuffer_albedo     = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal     = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive   = builder.resolve_image("GBuffer_Emissive");
        self.depth_buffer       = builder.resolve_image("DepthBuffer");
        self.hiz_mip0           = builder.resolve_image("HiZFinal");

        // Resolve buffer handles
        self.mesh_descriptor_buffer = builder.resolve_buffer("MeshDescriptorBuffer");
        self.instance_buffer        = builder.resolve_buffer("InstanceBuffer");
        self.draw_call_buffer       = builder.resolve_buffer("DrawCallBuffer");
        self.draw_count_buffer      = builder.resolve_buffer("DrawCountBuffer");
        self.material_buffer        = builder.resolve_buffer("MaterialBuffer");
        self.common_buffer          = builder.resolve_buffer("CommonBuffer");

        // Declare read intents
        builder.read_buffer(self.mesh_descriptor_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.instance_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.draw_call_buffer, ResourceUsage::INDIRECT_READ);
        builder.read_buffer(self.draw_count_buffer, ResourceUsage::INDIRECT_READ);
        builder.read_buffer(self.material_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.common_buffer, ResourceUsage::SHADER_READ);

        // Resolve bindless descriptor set
        self.bindless_set = *builder.consume::<DescriptorSetHandle>("BindlessSet");

        // Declare write targets
        builder.write_image(self.gbuffer_albedo,     ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_normal,     ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_roughmetal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_emissive,   ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer,       ResourceUsage::DEPTH_STENCIL);
        builder.write_image(self.hiz_mip0,           ResourceUsage::COLOR_ATTACHMENT);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("VoxelGBufferPass::execute: pipeline not initialized!");
            return;
        };
        ctx.bind_pipeline_raw(pipeline);

        let scene_set = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::storage_buffer(0, 0, self.mesh_descriptor_buffer),
                DescriptorWrite::storage_buffer(1, 0, self.instance_buffer),
                DescriptorWrite::storage_buffer(2, 0, self.material_buffer),
            ],
        );
        ctx.bind_descriptor_set(0, scene_set);

        let common_set = ctx.create_descriptor_set(
            pipeline,
            1,
            &[DescriptorWrite::uniform_buffer(0, 0, self.common_buffer)],
        );
        ctx.bind_descriptor_set(1, common_set);

        ctx.bind_descriptor_set(2, self.bindless_set);

        ctx.draw_indirect_count(
            self.draw_call_buffer,
            0,
            self.draw_count_buffer,
            0,
            i3_renderer::constants::MAX_INSTANCES as u32,
            std::mem::size_of::<i3_gfx::graph::backend::DrawIndirectCommand>() as u32,
        );
    }
}
