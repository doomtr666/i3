use i3_gfx::prelude::*;

use nalgebra_glm as glm;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LightCullPushConstants {
    pub view_matrix: glm::Mat4, // 64 bytes
    pub grid_size: [u32; 3],    // 12 bytes
    pub light_count: u32,       // 4 bytes -> total 80 bytes
}

pub struct LightCullPass {
    pub push_constants: LightCullPushConstants,
    pub light_buffer_name: String,

    // Resolved handles (updated in record)
    cluster_aabbs: BufferHandle,
    lights: BufferHandle,
    cluster_grid: BufferHandle,
    cluster_light_indices: BufferHandle,

    // Persistence
    pipeline: Option<BackendPipeline>,
}

impl LightCullPass {
    pub fn new() -> Self {
        Self {
            cluster_aabbs: BufferHandle::INVALID,
            lights: BufferHandle::INVALID,
            cluster_grid: BufferHandle::INVALID,
            cluster_light_indices: BufferHandle::INVALID,
            push_constants: LightCullPushConstants {
                view_matrix: nalgebra_glm::identity(),
                grid_size: [0, 0, 0],
                light_count: 0,
            },
            light_buffer_name: "LightBuffer".to_string(),
            pipeline: None,
        }
    }

    pub fn init_from_baked(
        &mut self,
        _backend: &mut dyn RenderBackend,
        asset: &i3_io::pipeline_asset::PipelineAsset,
    ) {
        if self.pipeline.is_some() {
            return;
        }

        self.pipeline = Some(_backend.create_compute_pipeline_from_baked(
            &asset.reflection_data,
            &asset.bytecode,
        ));
    }
}

impl RenderPass for LightCullPass {
    fn name(&self) -> &str {
        "LightCull"
    }

    fn init(&mut self, _backend: &mut dyn RenderBackend) {
        // Handled by init_from_baked
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        self.cluster_aabbs = builder.resolve_buffer("ClusterAABBs");
        self.lights = builder.resolve_buffer(&self.light_buffer_name);
        self.cluster_grid = builder.resolve_buffer("ClusterGrid");
        self.cluster_light_indices = builder.resolve_buffer("ClusterLightIndices");

        // Consume CommonData to compute push constants
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        let grid_x = (common.screen_width + 63) / 64;
        let grid_y = (common.screen_height + 63) / 64;
        let grid_z: u32 = 16;

        self.push_constants = LightCullPushConstants {
            view_matrix: common.view,
            grid_size: [grid_x, grid_y, grid_z],
            light_count: common.light_count,
        };

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
