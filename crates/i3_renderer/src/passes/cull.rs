use std::sync::Arc;
use i3_gfx::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DrawCallGenPushConstants {
    pub instance_count: u32,
}

pub struct DrawCallGenPass {
    pub instance_count: u32,

    // Resolved handles
    mesh_descriptor_buffer: BufferHandle,
    instance_buffer:        BufferHandle,
    draw_call_buffer:       BufferHandle,
    draw_count_buffer:      BufferHandle,

    pipeline: Option<BackendPipeline>,
}

impl DrawCallGenPass {
    pub fn new() -> Self {
        Self {
            instance_count:         0,
            mesh_descriptor_buffer: BufferHandle::INVALID,
            instance_buffer:        BufferHandle::INVALID,
            draw_call_buffer:       BufferHandle::INVALID,
            draw_count_buffer:      BufferHandle::INVALID,
            pipeline:               None,
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

        self.pipeline = Some(backend.create_compute_pipeline_from_baked(
            &asset.reflection_data,
            &asset.bytecode,
        ));
    }
}

impl RenderPass for DrawCallGenPass {
    fn name(&self) -> &str {
        "DrawCallGen"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(handle) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("draw_call_gen").wait_loaded() {
            self.init_from_baked(backend, &handle);
        }
    }

    fn record(&mut self, builder: &mut PassBuilder) {
        if builder.is_setup() {
            return;
        }

        self.mesh_descriptor_buffer = builder.resolve_buffer("MeshDescriptorBuffer");
        self.instance_buffer        = builder.resolve_buffer("InstanceBuffer");
        self.draw_call_buffer       = builder.resolve_buffer("DrawCallBuffer");
        self.draw_count_buffer      = builder.resolve_buffer("DrawCountBuffer");

        // The number of instances is published to the graph
        self.instance_count = builder.try_consume::<Vec<crate::scene::GpuInstanceData>>("SceneInstances")
            .map(|v| v.len() as u32)
            .unwrap_or(0);

        builder.read_buffer(self.mesh_descriptor_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.instance_buffer, ResourceUsage::SHADER_READ);
        builder.write_buffer(self.draw_call_buffer, ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.draw_count_buffer, ResourceUsage::SHADER_WRITE);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.mesh_descriptor_buffer,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.instance_buffer,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 2,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.draw_call_buffer,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 3,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo {
                        buffer: self.draw_count_buffer,
                        offset: 0,
                        range: 0,
                    }),
                    image_info: None,
                },
            ],
        );
    }

    fn execute(&self, ctx: &mut dyn PassContext) {
        if self.instance_count == 0 {
            return;
        }

        let Some(pipeline) = self.pipeline else {
            tracing::error!("DrawCallGenPass::execute: pipeline not initialized!");
            return;
        };

        ctx.bind_pipeline_raw(pipeline);
        ctx.push_constant_data(
            ShaderStageFlags::Compute,
            0,
            &DrawCallGenPushConstants {
                instance_count: self.instance_count,
            },
        );

        // Dispatch 1 thread per instance (or use a sensible workgroup size, e.g. 64)
        let group_count = (self.instance_count + 63) / 64;
        ctx.dispatch(group_count, 1, 1);
    }
}
