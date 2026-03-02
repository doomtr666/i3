use i3_gfx::prelude::*;
use i3_slang::prelude::*;
use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LightCullPushConstants {
    pub view_matrix: glm::Mat4, // 64 bytes
    pub grid_size: [u32; 3],    // 12 bytes
    pub light_count: u32,       // 4 bytes -> total 80 bytes
}

/// Light cull pass struct implementing the RenderPass trait.
pub struct LightCullPass {
    pub cluster_aabbs: BufferHandle,
    pub lights: BufferHandle,
    pub cluster_grid: BufferHandle,
    pub cluster_light_indices: BufferHandle,
    pub push_constants: LightCullPushConstants,

    // Persistence
    shader: Option<ShaderModule>,
    pipeline: Option<BackendPipeline>,
}

impl LightCullPass {
    pub fn new(
        cluster_aabbs: BufferHandle,
        lights: BufferHandle,
        cluster_grid: BufferHandle,
        cluster_light_indices: BufferHandle,
        push_constants: LightCullPushConstants,
    ) -> Self {
        Self {
            cluster_aabbs,
            lights,
            cluster_grid,
            cluster_light_indices,
            push_constants,
            shader: None,
            pipeline: None,
        }
    }
}

impl RenderPass for LightCullPass {
    fn name(&self) -> &str {
        "LightCull"
    }

    fn domain(&self) -> PassDomain {
        PassDomain::Compute
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
                .compile_file("light_cull", ShaderTarget::Spirv, &[shader_dir])
                .expect("Failed to compile LightCull shader"),
        );

        // 2. Create Pipeline
        self.pipeline = Some(backend.create_compute_pipeline(&ComputePipelineCreateInfo {
            shader_module: self.shader.clone().expect("Shader not compiled"),
        }));
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        builder.read_buffer(self.cluster_aabbs, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.lights, ResourceUsage::SHADER_READ);
        builder.write_buffer(self.cluster_grid, ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.cluster_light_indices, ResourceUsage::SHADER_WRITE);

        builder.bind_descriptor_set(
            0,
            vec![
                i3_gfx::graph::backend::DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                    buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                        buffer: self.cluster_aabbs,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                i3_gfx::graph::backend::DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                    buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                        buffer: self.lights,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                i3_gfx::graph::backend::DescriptorWrite {
                    binding: 2,
                    array_element: 0,
                    descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                    buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                        buffer: self.cluster_grid,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                i3_gfx::graph::backend::DescriptorWrite {
                    binding: 3,
                    array_element: 0,
                    descriptor_type: i3_gfx::graph::pipeline::BindingType::StorageBuffer,
                    buffer_info: Some(i3_gfx::graph::backend::DescriptorBufferInfo {
                        buffer: self.cluster_light_indices,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
            ],
        );
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        let pipeline = self
            .pipeline
            .expect("LightCullPass pipeline not initialized");
        ctx.bind_pipeline_raw(pipeline);

        ctx.push_constant_data(
            i3_gfx::graph::pipeline::ShaderStageFlags::Compute,
            0,
            &self.push_constants,
        );

        ctx.dispatch(
            self.push_constants.grid_size[0],
            self.push_constants.grid_size[1],
            self.push_constants.grid_size[2],
        );
    }
}
