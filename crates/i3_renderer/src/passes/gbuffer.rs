use std::sync::Arc;
use i3_gfx::prelude::*;


/// A single draw command extracted from the scene for the GBuffer pass.
#[derive(Clone, Copy, Debug)]
pub struct DrawCommand {
    pub mesh: crate::scene::Mesh,
    pub push_constants: GBufferPushConstants,
}

/// GBuffer vertex layout: position + normal + color.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GBufferVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 4],
}

/// Push constants for the GBuffer pass.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GBufferPushConstants {
    pub view_projection: nalgebra_glm::Mat4,
    pub model: nalgebra_glm::Mat4,
    pub material_id: u32,
    pub _pad: [u32; 3],
}

impl Default for GBufferPushConstants {
    fn default() -> Self {
        Self {
            view_projection: nalgebra_glm::identity(),
            model: nalgebra_glm::identity(),
            material_id: 0,
            _pad: [0; 3],
        }
    }
}

pub struct GBufferPass {
    pub bindless_set: u64,

    // Resolved handles (updated in record)
    depth_buffer: ImageHandle,
    gbuffer_albedo: ImageHandle,
    gbuffer_normal: ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive: ImageHandle,
    material_buffer: BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
    draw_commands: Vec<DrawCommand>,
}

impl GBufferPass {
    pub fn new() -> Self {
        Self {
            depth_buffer: ImageHandle::INVALID,
            gbuffer_albedo: ImageHandle::INVALID,
            gbuffer_normal: ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            gbuffer_emissive: ImageHandle::INVALID,
            material_buffer: BufferHandle::INVALID,
            bindless_set: 0,
            pipeline: None,
            draw_commands: Vec::new(),
        }
    }

    pub fn init_from_baked(
        &mut self,
        backend: &mut dyn RenderBackend,
        asset: &i3_io::pipeline_asset::PipelineAsset,
    ) {
        if self.pipeline.is_some() {
            return;
        }

        let state = asset.state.as_ref().expect("GBuffer asset missing state");
        self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
            state,
            &asset.reflection_data,
            &asset.bytecode,
        ));
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &str {
        "GBufferPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(handle) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("gbuffer").wait_loaded() {
            self.init_from_baked(backend, &handle);
        }
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        if builder.is_setup() {
            return;
        }
        // Resolve target handles by name
        self.gbuffer_albedo = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive = builder.resolve_image("GBuffer_Emissive");
        self.depth_buffer = builder.resolve_image("DepthBuffer");
        self.material_buffer = builder.resolve_buffer("MaterialBuffer");

        // Resolve bindless descriptor set from blackboard
        self.bindless_set = *builder.consume::<u64>("BindlessSet");

        // Consume draw commands from blackboard
        self.draw_commands = builder
            .consume::<Vec<DrawCommand>>("GBufferCommands")
            .clone();

        // Declare write targets
        builder.write_image(self.gbuffer_albedo, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_normal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_roughmetal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_emissive, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer, ResourceUsage::DEPTH_STENCIL);

        builder.read_buffer(self.material_buffer, ResourceUsage::READ);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("GBufferPass::execute: pipeline not initialized!");
            return;
        };
        ctx.bind_pipeline_raw(pipeline);

        // Bind Material SSBO at set 1
        let mat_set = ctx.create_descriptor_set(
            pipeline,
            1,
            &[DescriptorWrite::buffer(0, self.material_buffer)],
        );
        ctx.bind_descriptor_set(1, mat_set);

        // Bind Bindless Set at set 2
        ctx.bind_descriptor_set_raw(2, self.bindless_set);

        let expected_stride = std::mem::size_of::<GBufferVertex>() as u32;

        for cmd in &self.draw_commands {
            if cmd.mesh.stride != expected_stride {
                tracing::warn!(
                    "GBufferPass: Skipping mesh with incompatible stride {} (expected {})",
                    cmd.mesh.stride,
                    expected_stride
                );
                continue;
            }

            ctx.push_constant_data(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                &cmd.push_constants,
            );

            let vb = BufferHandle(SymbolId(cmd.mesh.vertex_buffer.0));
            let ib = BufferHandle(SymbolId(cmd.mesh.index_buffer.0));

            ctx.bind_vertex_buffer(0, vb);
            ctx.bind_index_buffer(ib, cmd.mesh.index_type);
            ctx.draw_indexed(cmd.mesh.index_count, 0, 0);
        }
    }
}
