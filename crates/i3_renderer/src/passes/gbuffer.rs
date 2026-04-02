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
}

impl Default for GBufferPushConstants {
    fn default() -> Self {
        Self {
            view_projection: nalgebra_glm::identity(),
        }
    }
}

pub struct GBufferPass {
    pub bindless_set: u64,

    // Resolved handles (updated in declare)
    depth_buffer:           ImageHandle,
    gbuffer_albedo:         ImageHandle,
    gbuffer_normal:         ImageHandle,
    gbuffer_roughmetal:     ImageHandle,
    gbuffer_emissive:       ImageHandle,
    
    mesh_descriptor_buffer: BufferHandle,
    instance_buffer:        BufferHandle,
    draw_call_buffer:       BufferHandle,
    draw_count_buffer:      BufferHandle,
    material_buffer:        BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
    common:   crate::render_graph::CommonData,
}

impl GBufferPass {
    pub fn new() -> Self {
        Self {
            depth_buffer:           ImageHandle::INVALID,
            gbuffer_albedo:         ImageHandle::INVALID,
            gbuffer_normal:         ImageHandle::INVALID,
            gbuffer_roughmetal:     ImageHandle::INVALID,
            gbuffer_emissive:       ImageHandle::INVALID,
            
            mesh_descriptor_buffer: BufferHandle::INVALID,
            instance_buffer:        BufferHandle::INVALID,
            draw_call_buffer:       BufferHandle::INVALID,
            draw_count_buffer:      BufferHandle::INVALID,
            material_buffer:        BufferHandle::INVALID,
            
            bindless_set: 0,
            pipeline:     None,
            common:       unsafe { std::mem::zeroed() },
        }
    }

}

impl RenderPass for GBufferPass {
    fn name(&self) -> &str {
        "GBufferPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("gbuffer").wait_loaded() {
            let state = asset.state.as_ref().expect("GBuffer asset missing state");
            self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                state,
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        if builder.is_setup() {
            return;
        }
        // Resolve target handles by name
        self.gbuffer_albedo = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive = builder.resolve_image("GBuffer_Emissive");
        self.depth_buffer = builder.resolve_image("DepthBuffer");
        
        self.mesh_descriptor_buffer = builder.resolve_buffer("MeshDescriptorBuffer");
        self.instance_buffer        = builder.resolve_buffer("InstanceBuffer");
        self.draw_call_buffer       = builder.resolve_buffer("DrawCallBuffer");
        self.draw_count_buffer      = builder.resolve_buffer("DrawCountBuffer");
        self.material_buffer        = builder.resolve_buffer("MaterialBuffer");

        // Declare reads
        builder.read_buffer(self.mesh_descriptor_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.instance_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.draw_call_buffer, ResourceUsage::INDIRECT_READ);
        builder.read_buffer(self.draw_count_buffer, ResourceUsage::INDIRECT_READ);
        builder.read_buffer(self.material_buffer, ResourceUsage::SHADER_READ);

        // Resolve bindless descriptor set from blackboard
        self.bindless_set = *builder.consume::<u64>("BindlessSet");

        // Consume CommonData to compute push constants
        self.common = *builder.consume::<crate::render_graph::CommonData>("Common");

        // Declare write targets
        builder.write_image(self.gbuffer_albedo, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_normal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_roughmetal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_emissive, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer, ResourceUsage::DEPTH_STENCIL);

        // Declare reads
        builder.read_buffer(self.mesh_descriptor_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.instance_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.draw_call_buffer, ResourceUsage::INDIRECT_READ);
        builder.read_buffer(self.draw_count_buffer, ResourceUsage::INDIRECT_READ);
        builder.read_buffer(self.material_buffer, ResourceUsage::SHADER_READ);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("GBufferPass::execute: pipeline not initialized!");
            return;
        };
        ctx.bind_pipeline_raw(pipeline);

        // Bind Global Scene Data at set 0
        // Binding 0: MeshDescriptors
        // Binding 1: Instances
        // Binding 2: Materials
        let scene_set = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::buffer(0, self.mesh_descriptor_buffer),
                DescriptorWrite::buffer(1, self.instance_buffer),
                DescriptorWrite::buffer(2, self.material_buffer),
            ],
        );
        ctx.bind_descriptor_set(0, scene_set);

        // Bind Bindless Set at set 2
        ctx.bind_descriptor_set_raw(2, self.bindless_set);

        // Push global view info
        ctx.push_constant_data(
            ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
            0,
            &GBufferPushConstants {
                view_projection: self.common.view_projection,
                ..Default::default()
            },
        );

        // Perform GPU-driven indirect drawing
        ctx.draw_indirect_count(
            self.draw_call_buffer,
            0,
            self.draw_count_buffer,
            0,
            1024 * 64, // max_draw_count
            std::mem::size_of::<i3_gfx::graph::backend::DrawIndirectCommand>() as u32,
        );
    }
}
