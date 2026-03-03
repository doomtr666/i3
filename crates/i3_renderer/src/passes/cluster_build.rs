use i3_gfx::prelude::*;
use i3_slang::prelude::*;
use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ClusterBuildPushConstants {
    pub inv_projection: glm::Mat4,
    pub grid_size: [u32; 3],
    pub near_plane: f32,
    pub far_plane: f32,
    pub screen_dimensions: [f32; 2],
    pub pad: u32,
}

/// Cluster build pass struct implementing the RenderPass trait.
pub struct ClusterBuildPass {
    pub cluster_aabbs: BufferHandle,
    pub push_constants: ClusterBuildPushConstants,

    // Persistence
    shader: Option<ShaderModule>,
    pipeline: Option<BackendPipeline>,
}

impl ClusterBuildPass {
    pub fn new(cluster_aabbs: BufferHandle, push_constants: ClusterBuildPushConstants) -> Self {
        Self {
            cluster_aabbs,
            push_constants,
            shader: None,
            pipeline: None,
        }
    }
}

impl RenderPass for ClusterBuildPass {
    fn name(&self) -> &str {
        "ClusterBuild"
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
                .compile_file("cluster_build", ShaderTarget::Spirv, &[shader_dir])
                .expect("Failed to compile ClusterBuild shader"),
        );

        // 2. Create Pipeline
        self.pipeline = Some(backend.create_compute_pipeline(&ComputePipelineCreateInfo {
            shader_module: self.shader.clone().expect("Shader not compiled"),
        }));
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.write_buffer(self.cluster_aabbs, ResourceUsage::SHADER_WRITE);
        builder.bind_descriptor_set(
            0,
            vec![i3_gfx::graph::backend::DescriptorWrite {
                binding: 0,
                array_element: 0,
                descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                    buffer: self.cluster_aabbs,
                    offset: 0,
                    range: 0,
                }),
                image_info: None,
            }],
        );
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let pipeline = self
            .pipeline
            .expect("ClusterBuildPass pipeline not initialized");
        ctx.bind_pipeline_raw(pipeline);

        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &self.push_constants);

        ctx.dispatch(
            self.push_constants.grid_size[0],
            self.push_constants.grid_size[1],
            self.push_constants.grid_size[2],
        );
    }
}
