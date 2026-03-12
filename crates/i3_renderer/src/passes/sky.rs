use i3_gfx::prelude::*;
use i3_slang::prelude::*;
use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SkyPushConstants {
    pub inv_view_proj: glm::Mat4,
    pub camera_pos: glm::Vec3,
    pub _pad0: f32,
    pub sun_direction: glm::Vec3,
    pub sun_intensity: f32,
    pub sun_color: glm::Vec3,
    pub _pad1: f32,
}

/// Sky pass struct implementing the RenderPass trait.
pub struct SkyPass {
    // Resolved handles (updated in record)
    hdr_target: ImageHandle,
    depth_buffer: ImageHandle,

    // Persistence
    shader: Option<ShaderModule>,
    pipeline: Option<BackendPipeline>,
    push_constants: Option<SkyPushConstants>,
}

impl SkyPass {
    pub fn new() -> Self {
        let dummy_image = ImageHandle(SymbolId(0));
        Self {
            hdr_target: dummy_image,
            depth_buffer: dummy_image,
            shader: None,
            pipeline: None,
            push_constants: None,
        }
    }

    pub fn create_pipeline_info(&self) -> GraphicsPipelineCreateInfo {
        GraphicsPipelineCreateInfo {
            shader_module: self.shader.clone().expect("Shader not compiled"),
            render_targets: RenderTargetsInfo {
                color_targets: vec![RenderTargetInfo {
                    format: Format::R16G16B16A16_SFLOAT,
                    ..Default::default()
                }],
                depth_stencil_format: Some(Format::D32_FLOAT),
                ..Default::default()
            },
            rasterization_state: RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            },
            depth_stencil_state: DepthStencilState {
                depth_test_enable: true,
                depth_write_enable: false, // Sky is at infinity
                depth_compare_op: CompareOp::LessOrEqual,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

impl RenderPass for SkyPass {
    fn name(&self) -> &str {
        "SkyPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend) {
        if self.pipeline.is_some() {
            return;
        }

        // 1. Compile Shader
        let slang = SlangCompiler::new().expect("Failed to create Slang compiler");
        let shader_dir = "crates/i3_renderer/shaders";

        self.shader = Some(
            slang
                .compile_file("sky", ShaderTarget::Spirv, &[shader_dir])
                .expect("Failed to compile Sky shader"),
        );

        // 2. Create Pipeline
        let info = self.create_pipeline_info();
        self.pipeline = Some(backend.create_graphics_pipeline(&info));
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        // Resolve target handles by name
        self.hdr_target = builder.resolve_image("HDR_Target");
        self.depth_buffer = builder.resolve_image("DepthBuffer");

        // Compute push constants from blackboard
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        let sun_dir = *builder.consume::<glm::Vec3>("SunDirection");
        let sun_int = *builder.consume::<f32>("SunIntensity");
        let sun_col = *builder.consume::<glm::Vec3>("SunColor");

        self.push_constants = Some(SkyPushConstants {
            inv_view_proj: common.view_projection.try_inverse().unwrap_or_default(),
            camera_pos: common.camera_pos,
            _pad0: 0.0,
            sun_direction: sun_dir,
            sun_intensity: sun_int,
            sun_color: sun_col,
            _pad1: 0.0,
        });

        // Clears the HDR target and depth buffer (to 1.0)
        builder.write_image(self.hdr_target, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer, ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let pipeline = self.pipeline.expect("SkyPass pipeline not initialized");
        ctx.bind_pipeline_raw(pipeline);

        if let Some(constants) = self.push_constants {
            ctx.push_constant_data(
                ShaderStageFlags::Vertex | ShaderStageFlags::Fragment,
                0,
                &constants,
            );
            ctx.draw(3, 0); // Fullscreen triangle
        }
    }
}
