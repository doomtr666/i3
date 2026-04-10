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

// ─────────────────────────────────────────────────────────────────────────────
// GBufferFillPass — records the actual GPU draw commands
// ─────────────────────────────────────────────────────────────────────────────

pub struct GBufferFillPass {
    pub bindless_set: DescriptorSetHandle,

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
}

impl GBufferFillPass {
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

            bindless_set: DescriptorSetHandle(0),
            pipeline:     None,
        }
    }
}

impl RenderPass for GBufferFillPass {
    fn name(&self) -> &str {
        "GBufferFillPass"
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
        // Resolve image handles (declared as outputs by the parent GBufferPass)
        self.gbuffer_albedo     = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal     = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive   = builder.resolve_image("GBuffer_Emissive");
        self.depth_buffer       = builder.resolve_image("DepthBuffer");

        // Resolve buffer handles
        self.mesh_descriptor_buffer = builder.resolve_buffer("MeshDescriptorBuffer");
        self.instance_buffer        = builder.resolve_buffer("InstanceBuffer");
        self.draw_call_buffer       = builder.resolve_buffer("DrawCallBuffer");
        self.draw_count_buffer      = builder.resolve_buffer("DrawCountBuffer");
        self.material_buffer        = builder.resolve_buffer("MaterialBuffer");

        // Declare read intents
        builder.read_buffer(self.mesh_descriptor_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.instance_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.draw_call_buffer, ResourceUsage::INDIRECT_READ);
        builder.read_buffer(self.draw_count_buffer, ResourceUsage::INDIRECT_READ);
        builder.read_buffer(self.material_buffer, ResourceUsage::SHADER_READ);

        // Resolve bindless descriptor set
        self.bindless_set = *builder.consume::<DescriptorSetHandle>("BindlessSet");

        // Declare write targets
        builder.write_image(self.gbuffer_albedo,     ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_normal,     ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_roughmetal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_emissive,   ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer,       ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("GBufferFillPass::execute: pipeline not initialized!");
            return;
        };
        let common = frame.consume::<crate::render_graph::CommonData>("Common");
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
        ctx.bind_descriptor_set(2, self.bindless_set);

        ctx.push_constant_data(
            ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
            0,
            &GBufferPushConstants {
                view_projection: common.view_projection,
                ..Default::default()
            },
        );

        ctx.draw_indirect_count(
            self.draw_call_buffer,
            0,
            self.draw_count_buffer,
            0,
            1024 * 64,
            std::mem::size_of::<i3_gfx::graph::backend::DrawIndirectCommand>() as u32,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GBufferPass — parent group: declares GBuffer images as outputs, adds fill child
// ─────────────────────────────────────────────────────────────────────────────

pub struct GBufferPass {
    pub fill: GBufferFillPass,
}

impl GBufferPass {
    pub fn new() -> Self {
        Self { fill: GBufferFillPass::new() }
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &str {
        "GBufferPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.fill.init(backend, globals);
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        let (w, h) = (common.screen_width, common.screen_height);

        builder.declare_image_output("GBuffer_Albedo",     ImageDesc::new(w, h, Format::R8G8B8A8_SRGB));
        builder.declare_image_output("GBuffer_Normal",     ImageDesc::new(w, h, Format::R16G16_SFLOAT));
        builder.declare_image_output("GBuffer_RoughMetal", ImageDesc::new(w, h, Format::R8G8_UNORM));
        builder.declare_image_output("GBuffer_Emissive",   ImageDesc::new(w, h, Format::R11G11B10_UFLOAT));
        builder.declare_image_output("DepthBuffer", ImageDesc {
            width: w,
            height: h,
            depth: 1,
            format: Format::D32_FLOAT,
            mip_levels: 1,
            array_layers: 1,
            usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::SAMPLED,
            view_type: ImageViewType::Type2D,
            swizzle: ComponentMapping::default(),
            clear_value: None,
        });

        builder.add_pass(&mut self.fill);
    }
}
