use crate::constants::{DRAW_INDIRECT_CMD_SIZE, MAX_INSTANCES};
use i3_gfx::prelude::*;
use std::sync::Arc;

// Push constants for DrawCallGenComputePass (draw_call_gen.slang).
// Layout must match the Slang struct exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DrawCallGenPushConstants {
    pub view_projection: nalgebra_glm::Mat4, // 64 bytes, offset 0
    pub instance_count:  u32,                //  4 bytes, offset 64
    pub max_draw_calls:  u32,                //  4 bytes, offset 68
    pub _pad:            [u32; 2],           //  8 bytes, offset 72
}                                            // total: 80 bytes

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
    fn name(&self) -> &str { "DrawCallGenCompute" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("draw_call_gen")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.mesh_descriptor_buffer = builder.resolve_buffer("MeshDescriptorBuffer");
        self.instance_buffer        = builder.resolve_buffer("InstanceBuffer");
        self.draw_call_buffer       = builder.resolve_buffer("DrawCallBuffer");
        self.draw_count_buffer      = builder.resolve_buffer("DrawCountBuffer");

        self.instance_count = builder
            .try_consume::<Vec<crate::scene::GpuInstanceData>>("SceneInstances")
            .map(|v| v.len() as u32)
            .unwrap_or(0);

        builder.read_buffer(self.mesh_descriptor_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.instance_buffer,        ResourceUsage::SHADER_READ);
        builder.write_buffer(self.draw_call_buffer,      ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.draw_count_buffer,     ResourceUsage::SHADER_WRITE);

        builder.descriptor_set(0, |d| {
            d.storage_buffer(self.mesh_descriptor_buffer)
                .storage_buffer(self.instance_buffer)
                .storage_buffer(self.draw_call_buffer)
                .storage_buffer(self.draw_count_buffer);
        });
    }

    fn execute(
        &self,
        ctx: &mut dyn PassContext,
        frame: &i3_gfx::graph::compiler::FrameBlackboard,
    ) {
        if self.instance_count == 0 {
            return;
        }
        let Some(pipeline) = self.pipeline else {
            tracing::error!("DrawCallGenComputePass::execute: pipeline not initialized!");
            return;
        };

        let common       = frame.consume::<crate::render_graph::CommonData>("Common");
        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");

        ctx.bind_pipeline_raw(pipeline);
        ctx.bind_descriptor_set(2, bindless_set);
        ctx.push_constant_data(
            ShaderStageFlags::Compute,
            0,
            &DrawCallGenPushConstants {
                view_projection: common.view_projection,
                instance_count:  self.instance_count,
                max_draw_calls:  MAX_INSTANCES as u32,
                _pad:            [0; 2],
            },
        );

        let group_count = (self.instance_count + 63) / 64;
        ctx.dispatch(group_count, 1, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DrawCallGenPass — group: declares transient draw buffers, clears count,
//                          then dispatches frustum cull compute.
// ─────────────────────────────────────────────────────────────────────────────

pub struct DrawCallGenPass {
    draw_call_buffer:  BufferHandle,
    draw_count_buffer: BufferHandle,
    compute:           DrawCallGenComputePass,
}

impl DrawCallGenPass {
    pub fn new() -> Self {
        Self {
            draw_call_buffer:  BufferHandle::INVALID,
            draw_count_buffer: BufferHandle::INVALID,
            compute:           DrawCallGenComputePass::new(),
        }
    }
}

impl RenderPass for DrawCallGenPass {
    fn name(&self) -> &str { "DrawCallGen" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.compute.init(backend, globals);
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.draw_call_buffer = builder.declare_buffer_output(
            "DrawCallBuffer",
            BufferDesc {
                size:   MAX_INSTANCES * DRAW_INDIRECT_CMD_SIZE,
                usage:  BufferUsageFlags::STORAGE_BUFFER
                      | BufferUsageFlags::INDIRECT_BUFFER
                      | BufferUsageFlags::TRANSFER_DST,
                memory: MemoryType::GpuOnly,
            },
        );
        self.draw_count_buffer = builder.declare_buffer_output(
            "DrawCountBuffer",
            BufferDesc {
                size:   16,
                usage:  BufferUsageFlags::STORAGE_BUFFER
                      | BufferUsageFlags::INDIRECT_BUFFER
                      | BufferUsageFlags::TRANSFER_DST,
                memory: MemoryType::GpuOnly,
            },
        );

        builder.add_owned_pass(crate::render_graph::ClearBufferPass {
            name:   "ClearDrawCount".to_string(),
            buffer: self.draw_count_buffer,
        });

        builder.add_pass(&mut self.compute);
    }

    // No execute() — pure group.
}
