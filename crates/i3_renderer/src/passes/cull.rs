use std::sync::Arc;
use i3_gfx::prelude::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DrawCallGenPushConstants {
    pub instance_count: u32,
    pub max_draw_calls: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// DrawCallGenComputePass — inner leaf: resolves handles, dispatches the shader
// ─────────────────────────────────────────────────────────────────────────────

struct DrawCallGenComputePass {
    instance_count: u32,

    // Resolved handles (updated in declare)
    mesh_descriptor_buffer: BufferHandle,
    instance_buffer:        BufferHandle,
    draw_call_buffer:       BufferHandle,
    draw_count_buffer:      BufferHandle,

    pipeline: Option<BackendPipeline>,
}

impl DrawCallGenComputePass {
    fn new() -> Self {
        Self {
            instance_count:         0,
            mesh_descriptor_buffer: BufferHandle::INVALID,
            instance_buffer:        BufferHandle::INVALID,
            draw_call_buffer:       BufferHandle::INVALID,
            draw_count_buffer:      BufferHandle::INVALID,
            pipeline:               None,
        }
    }
}

impl RenderPass for DrawCallGenComputePass {
    fn name(&self) -> &str {
        "DrawCallGenCompute"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("draw_call_gen").wait_loaded() {
            self.pipeline = Some(backend.create_compute_pipeline_from_baked(
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.mesh_descriptor_buffer = builder.resolve_buffer("MeshDescriptorBuffer");
        self.instance_buffer        = builder.resolve_buffer("InstanceBuffer");
        self.draw_call_buffer       = builder.resolve_buffer("DrawCallBuffer");
        self.draw_count_buffer      = builder.resolve_buffer("DrawCountBuffer");

        self.instance_count = builder.try_consume::<Vec<crate::scene::GpuInstanceData>>("SceneInstances")
            .map(|v| v.len() as u32)
            .unwrap_or(0);

        builder.read_buffer(self.mesh_descriptor_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.instance_buffer,        ResourceUsage::SHADER_READ);
        builder.write_buffer(self.draw_call_buffer,      ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.draw_count_buffer,     ResourceUsage::SHADER_WRITE);

        builder.bind_descriptor_set(
            0,
            vec![
                DescriptorWrite {
                    binding: 0,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo { buffer: self.mesh_descriptor_buffer, offset: 0, range: 0 }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 1,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo { buffer: self.instance_buffer, offset: 0, range: 0 }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 2,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo { buffer: self.draw_call_buffer, offset: 0, range: 0 }),
                    image_info: None,
                },
                DescriptorWrite {
                    binding: 3,
                    array_element: 0,
                    descriptor_type: BindingType::StorageBuffer,
                    buffer_info: Some(DescriptorBufferInfo { buffer: self.draw_count_buffer, offset: 0, range: 0 }),
                    image_info: None,
                },
            ],
        );
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        if self.instance_count == 0 {
            return;
        }
        let Some(pipeline) = self.pipeline else {
            tracing::error!("DrawCallGenComputePass::execute: pipeline not initialized!");
            return;
        };
        ctx.bind_pipeline_raw(pipeline);
        ctx.push_constant_data(
            ShaderStageFlags::Compute,
            0,
            &DrawCallGenPushConstants {
                instance_count: self.instance_count,
                max_draw_calls: 262144,
            },
        );
        let group_count = (self.instance_count + 63) / 64;
        ctx.dispatch(group_count, 1, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DrawCallGenPass — group: imports draw buffers, adds ClearDrawCount then compute
//
// Declared as a pure group (no execute, no resource intents in own storage).
// flatten_recursive skips groups with no intents, so children are flattened in
// declaration order: ClearDrawCount (index 0) → DrawCallGenCompute (index 1).
// This guarantees the draw count is zeroed before the compute shader fills it.
// ─────────────────────────────────────────────────────────────────────────────

pub struct DrawCallGenPass {
    // Owned persistent GPU buffers
    draw_call_buffer_physical:  BackendBuffer,
    draw_count_buffer_physical: BackendBuffer,

    // Handles resolved during declare (needed by children)
    draw_call_buffer:  BufferHandle,
    draw_count_buffer: BufferHandle,

    compute: DrawCallGenComputePass,
}

impl DrawCallGenPass {
    pub fn new() -> Self {
        Self {
            draw_call_buffer_physical:  BackendBuffer::INVALID,
            draw_count_buffer_physical: BackendBuffer::INVALID,
            draw_call_buffer:           BufferHandle::INVALID,
            draw_count_buffer:          BufferHandle::INVALID,
            compute:                    DrawCallGenComputePass::new(),
        }
    }
}

impl RenderPass for DrawCallGenPass {
    fn name(&self) -> &str {
        "DrawCallGen"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let max_instances: u64 = 262144;
        self.draw_call_buffer_physical = backend.create_buffer(&BufferDesc {
            size: max_instances * 16, // DrawIndirectCommand is 16 bytes
            usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::INDIRECT_BUFFER | BufferUsageFlags::TRANSFER_DST,
            memory: MemoryType::GpuOnly,
        });
        self.draw_count_buffer_physical = backend.create_buffer(&BufferDesc {
            size: 16, // 4 bytes + padding
            usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::INDIRECT_BUFFER | BufferUsageFlags::TRANSFER_DST,
            memory: MemoryType::GpuOnly,
        });
        #[cfg(debug_assertions)]
        {
            backend.set_buffer_name(self.draw_call_buffer_physical, "DrawCallBuffer");
            backend.set_buffer_name(self.draw_count_buffer_physical, "DrawCountBuffer");
        }
        self.compute.init(backend, globals);
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // Import owned buffers into graph scope; is_output: true propagates them
        // to the parent scope so GBufferFillPass and other siblings can resolve them.
        self.draw_call_buffer  = builder.import_buffer("DrawCallBuffer",  self.draw_call_buffer_physical);
        self.draw_count_buffer = builder.import_buffer("DrawCountBuffer", self.draw_count_buffer_physical);

        // Child 1: clear draw count — runs first in the flattened pass list
        builder.add_owned_pass(crate::render_graph::ClearBufferPass {
            name: "ClearDrawCount".to_string(),
            buffer: self.draw_count_buffer,
        });

        // Child 2: compute pass — runs after clear
        builder.add_pass(&mut self.compute);

        // No resource intents on the group itself → not pushed as leaf →
        // only children appear in the flat pass list, in declaration order.
    }

    // No execute() — this is a pure group.
}
