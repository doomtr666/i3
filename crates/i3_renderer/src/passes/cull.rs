use crate::constants::{DRAW_INDIRECT_CMD_SIZE, MAX_INSTANCES};
use i3_gfx::graph::backend::DrawIndirectCommand as GfxDrawIndirectCommand;
use i3_gfx::prelude::*;
use std::sync::Arc;

// Push constants for DrawCallGenComputePass (draw_call_gen.slang).
// Layout must match the Slang struct exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DrawCallGenPushConstants {
    pub view_projection: nalgebra_glm::Mat4, // 64 bytes, offset 0
    pub hiz_size:        [f32; 2],            //  8 bytes, offset 64  HIZ_PREZ_WIDTH/HEIGHT
    pub hiz_mip_count:   u32,                 //  4 bytes, offset 72  (0 = skip occlusion test)
    pub instance_count:  u32,                 //  4 bytes, offset 76
    pub max_draw_calls:  u32,                 //  4 bytes, offset 80
    pub _pad:            u32,                 //  4 bytes, offset 84
}                                             // total: 88 bytes

// Separate, minimal push constants for PreZFillPass (prez.slang only reads viewProjection).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PreZPushConstants {
    view_projection: nalgebra_glm::Mat4, // 64 bytes
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
    hiz_pyramid:            ImageHandle,  // HiZPreZ or fallback — always valid when used
    hiz_mip_count:          u32,          // cached during declare(); 0 = skip occlusion test
    visible_bitset:         BufferHandle, // VisibilityBitset frame N (write)

    pipeline:          Option<BackendPipeline>,
    /// 1×1 R32_SFLOAT image filled with 0.0 (far plane in reverse-Z).
    /// Bound to binding 4 when HiZPreZ is not yet wired in the frame graph, so
    /// Vulkan always has a valid descriptor even though the shader won't sample it
    /// (hiz_mip_count == 0 → early-return in occlusionTest).
    fallback_hiz_image: Option<BackendImage>,
}

impl DrawCallGenComputePass {
    fn new() -> Self {
        Self {
            instance_count:         0,
            mesh_descriptor_buffer: BufferHandle::INVALID,
            instance_buffer:        BufferHandle::INVALID,
            draw_call_buffer:       BufferHandle::INVALID,
            draw_count_buffer:      BufferHandle::INVALID,
            hiz_pyramid:            ImageHandle::INVALID,
            hiz_mip_count:          0,
            visible_bitset:         BufferHandle::INVALID,
            pipeline:               None,
            fallback_hiz_image:     None,
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

        // 1×1 R32_SFLOAT image containing 0.0 (far plane in reverse-Z).
        // Bound to binding 4 when HiZPreZ is not yet in the frame graph so Vulkan always
        // sees a valid descriptor — even though hiz_mip_count=0 makes the shader skip sampling.
        let fallback = backend.create_image(&ImageDesc {
            width:        1,
            height:       1,
            depth:        1,
            format:       Format::R32_SFLOAT,
            mip_levels:   1,
            array_layers: 1,
            usage:        ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
            view_type:    ImageViewType::Type2D,
            swizzle:      ComponentMapping::default(),
            clear_value:  None,
        });
        let _ = backend.upload_image(fallback, bytemuck::bytes_of(&0.0f32), 0, 0, 1, 1, 0, 0);
        self.fallback_hiz_image = Some(fallback);
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.mesh_descriptor_buffer = builder.resolve_buffer("MeshDescriptorBuffer");
        self.instance_buffer        = builder.resolve_buffer("InstanceBuffer");
        self.draw_call_buffer       = builder.resolve_buffer("DrawCallBuffer");
        self.draw_count_buffer      = builder.resolve_buffer("DrawCountBuffer");
        self.visible_bitset         = builder.resolve_buffer("VisibilityBitset");

        // HiZPreZ is optional — only wired in M6.
        // Always provide a valid image handle for binding 4: use HiZPreZ when available,
        // otherwise register the 1×1 fallback so Vulkan never sees an unbound descriptor.
        self.hiz_pyramid = builder
            .try_consume::<ImageHandle>("HiZPreZ")
            .copied()
            .unwrap_or_else(|| {
                let fallback_desc = ImageDesc {
                    width: 1, height: 1, depth: 1,
                    format: Format::R32_SFLOAT,
                    mip_levels: 1, array_layers: 1,
                    usage: ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
                    view_type: ImageViewType::Type2D,
                    swizzle: ComponentMapping::default(),
                    clear_value: None,
                };
                let handle = builder.declare_image("DrawCallGenFallbackHiZ", fallback_desc);
                if let Some(physical) = self.fallback_hiz_image {
                    builder.register_external_image(handle, physical);
                }
                handle
            });

        {
            let desc = builder.get_image_desc(self.hiz_pyramid);
            let is_fallback = desc.width == 1 && desc.height == 1;
            self.hiz_mip_count = if self.fallback_hiz_image.is_some() && !is_fallback {
                desc.mip_levels
            } else {
                0
            };
        }

        self.instance_count = builder
            .try_consume::<Vec<crate::scene::GpuInstanceData>>("SceneInstances")
            .map(|v| v.len() as u32)
            .unwrap_or(0);

        builder.read_buffer(self.mesh_descriptor_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.instance_buffer,        ResourceUsage::SHADER_READ);
        builder.write_buffer(self.draw_call_buffer,      ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.draw_count_buffer,     ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.visible_bitset,        ResourceUsage::SHADER_WRITE);
        builder.read_image(self.hiz_pyramid,             ResourceUsage::SHADER_READ);

        // Descriptor set 0 built explicitly in execute() — includes the (possibly fallback) image.
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

        // Binding 4 is always valid: either HiZPreZ or the 1×1 fallback registered in declare().
        // hiz_mip_count == 0 causes the shader to skip sampling, but Vulkan requires
        // all statically-declared bindings to have a valid descriptor.
        let writes = [
            DescriptorWrite::storage_buffer(0, 0, self.mesh_descriptor_buffer),
            DescriptorWrite::storage_buffer(1, 0, self.instance_buffer),
            DescriptorWrite::storage_buffer(2, 0, self.draw_call_buffer),
            DescriptorWrite::storage_buffer(3, 0, self.draw_count_buffer),
            DescriptorWrite::sampled_image(4, 0, self.hiz_pyramid, DescriptorImageLayout::ShaderReadOnlyOptimal),
            DescriptorWrite::storage_buffer(5, 0, self.visible_bitset),
        ];

        let descriptor_set = ctx.create_descriptor_set(pipeline, 0, &writes);

        ctx.bind_pipeline_raw(pipeline);
        ctx.bind_descriptor_set(0, descriptor_set);
        ctx.bind_descriptor_set(2, bindless_set);
        ctx.push_constant_data(
            ShaderStageFlags::Compute,
            0,
            &DrawCallGenPushConstants {
                view_projection: common.view_projection,
                hiz_size:        [crate::constants::HIZ_PREZ_WIDTH as f32, crate::constants::HIZ_PREZ_HEIGHT as f32],
                hiz_mip_count:   self.hiz_mip_count,
                instance_count:  self.instance_count,
                max_draw_calls:  MAX_INSTANCES as u32,
                _pad:            0,
            },
        );

        let group_count = (self.instance_count + 63) / 64;
        ctx.dispatch(group_count, 1, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PreZCullComputePass — inner leaf for PreZCullPass
// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PreZCullPushConstants {
    view_projection: nalgebra_glm::Mat4,
    instance_count:  u32,
    max_draw_calls:  u32,
    _pad:            [u32; 2],
}

struct PreZCullComputePass {
    instance_count: u32,

    mesh_descriptor_buffer:    BufferHandle,
    instance_buffer:           BufferHandle,
    draw_call_buffer:          BufferHandle,
    draw_count_buffer:         BufferHandle,
    visibility_bitset_history: BufferHandle,

    pipeline: Option<BackendPipeline>,
}

impl PreZCullComputePass {
    fn new() -> Self {
        Self {
            instance_count:            0,
            mesh_descriptor_buffer:    BufferHandle::INVALID,
            instance_buffer:           BufferHandle::INVALID,
            draw_call_buffer:          BufferHandle::INVALID,
            draw_count_buffer:         BufferHandle::INVALID,
            visibility_bitset_history: BufferHandle::INVALID,
            pipeline:                  None,
        }
    }
}

impl RenderPass for PreZCullComputePass {
    fn name(&self) -> &str { "PreZCullCompute" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("prez_cull")
            .wait_loaded()
        {
            self.pipeline = Some(
                backend.create_compute_pipeline_from_baked(&asset.reflection_data, &asset.bytecode),
            );
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.mesh_descriptor_buffer   = builder.resolve_buffer("MeshDescriptorBuffer");
        self.instance_buffer          = builder.resolve_buffer("InstanceBuffer");
        self.draw_call_buffer         = builder.resolve_buffer("DrawCallBuffer_PreZ");
        self.draw_count_buffer        = builder.resolve_buffer("DrawCountBuffer_PreZ");
        self.visibility_bitset_history = builder.read_buffer_history("VisibilityBitset");

        self.instance_count = builder
            .try_consume::<Vec<crate::scene::GpuInstanceData>>("SceneInstances")
            .map(|v| v.len() as u32)
            .unwrap_or(0);

        builder.read_buffer(self.mesh_descriptor_buffer,    ResourceUsage::SHADER_READ);
        builder.read_buffer(self.instance_buffer,           ResourceUsage::SHADER_READ);
        builder.read_buffer(self.visibility_bitset_history, ResourceUsage::SHADER_READ);
        builder.write_buffer(self.draw_call_buffer,         ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.draw_count_buffer,        ResourceUsage::SHADER_WRITE);

        builder.descriptor_set(0, |d| {
            d.storage_buffer(self.mesh_descriptor_buffer)
                .storage_buffer(self.instance_buffer)
                .storage_buffer(self.draw_call_buffer)
                .storage_buffer(self.draw_count_buffer)
                .storage_buffer(self.visibility_bitset_history);
        });
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        if self.instance_count == 0 {
            return;
        }
        let Some(pipeline) = self.pipeline else {
            tracing::error!("PreZCullComputePass::execute: pipeline not initialized!");
            return;
        };

        let common       = frame.consume::<crate::render_graph::CommonData>("Common");
        let bindless_set = *frame.consume::<DescriptorSetHandle>("BindlessSet");

        ctx.bind_pipeline_raw(pipeline);
        ctx.bind_descriptor_set(2, bindless_set);
        ctx.push_constant_data(
            ShaderStageFlags::Compute,
            0,
            &PreZCullPushConstants {
                view_projection: common.view_projection,
                instance_count:  self.instance_count,
                max_draw_calls:  MAX_INSTANCES as u32,
                _pad:            [0; 2],
            },
        );
        ctx.dispatch((self.instance_count + 63) / 64, 1, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PreZCullPass — group: declares transient PreZ draw buffers, clears count, dispatches
//
// Emits draws for { frustum_visible ∩ visible_N1 }.
// Children: ClearDrawCount_PreZ → PreZCullCompute.
// ─────────────────────────────────────────────────────────────────────────────

pub struct PreZCullPass {
    draw_call_buffer:  BufferHandle,
    draw_count_buffer: BufferHandle,
    compute:           PreZCullComputePass,
}

impl PreZCullPass {
    pub fn new() -> Self {
        Self {
            draw_call_buffer:  BufferHandle::INVALID,
            draw_count_buffer: BufferHandle::INVALID,
            compute:           PreZCullComputePass::new(),
        }
    }
}

impl RenderPass for PreZCullPass {
    fn name(&self) -> &str { "PreZCull" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.compute.init(backend, globals);
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.draw_call_buffer = builder.declare_buffer_output(
            "DrawCallBuffer_PreZ",
            BufferDesc {
                size:   MAX_INSTANCES * DRAW_INDIRECT_CMD_SIZE,
                usage:  BufferUsageFlags::STORAGE_BUFFER
                      | BufferUsageFlags::INDIRECT_BUFFER
                      | BufferUsageFlags::TRANSFER_DST,
                memory: MemoryType::GpuOnly,
            },
        );
        self.draw_count_buffer = builder.declare_buffer_output(
            "DrawCountBuffer_PreZ",
            BufferDesc {
                size:   16,
                usage:  BufferUsageFlags::STORAGE_BUFFER
                      | BufferUsageFlags::INDIRECT_BUFFER
                      | BufferUsageFlags::TRANSFER_DST,
                memory: MemoryType::GpuOnly,
            },
        );

        builder.add_owned_pass(crate::render_graph::ClearBufferPass {
            name:   "ClearDrawCount_PreZ".to_string(),
            buffer: self.draw_count_buffer,
        });

        builder.add_pass(&mut self.compute);
    }

    // No execute() — pure group.
}

// ─────────────────────────────────────────────────────────────────────────────
// PreZFillPass — leaf: depth-only draw_indirect_count from PreZ draw buffers
// ─────────────────────────────────────────────────────────────────────────────

struct PreZFillPass {
    depth_prez:             ImageHandle,
    mesh_descriptor_buffer: BufferHandle,
    instance_buffer:        BufferHandle,
    draw_call_buffer:       BufferHandle,
    draw_count_buffer:      BufferHandle,
    bindless_set:           DescriptorSetHandle,
    pipeline:               Option<BackendPipeline>,
}

impl PreZFillPass {
    fn new() -> Self {
        Self {
            depth_prez:             ImageHandle::INVALID,
            mesh_descriptor_buffer: BufferHandle::INVALID,
            instance_buffer:        BufferHandle::INVALID,
            draw_call_buffer:       BufferHandle::INVALID,
            draw_count_buffer:      BufferHandle::INVALID,
            bindless_set:           DescriptorSetHandle(0),
            pipeline:               None,
        }
    }
}

impl RenderPass for PreZFillPass {
    fn name(&self) -> &str { "PreZFill" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("prez")
            .wait_loaded()
        {
            let state = asset.state.as_ref().expect("prez asset missing pipeline state");
            self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                state,
                &asset.reflection_data,
                &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.depth_prez             = builder.resolve_image("DepthPreZ");
        self.mesh_descriptor_buffer = builder.resolve_buffer("MeshDescriptorBuffer");
        self.instance_buffer        = builder.resolve_buffer("InstanceBuffer");
        self.draw_call_buffer       = builder.resolve_buffer("DrawCallBuffer_PreZ");
        self.draw_count_buffer      = builder.resolve_buffer("DrawCountBuffer_PreZ");
        self.bindless_set           = *builder.consume::<DescriptorSetHandle>("BindlessSet");

        builder.read_buffer(self.mesh_descriptor_buffer, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.instance_buffer,        ResourceUsage::SHADER_READ);
        builder.read_buffer(self.draw_call_buffer,       ResourceUsage::INDIRECT_READ);
        builder.read_buffer(self.draw_count_buffer,      ResourceUsage::INDIRECT_READ);
        builder.write_image(self.depth_prez,             ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            tracing::error!("PreZFillPass::execute: pipeline not initialized!");
            return;
        };
        let common = frame.consume::<crate::render_graph::CommonData>("Common");

        let scene_set = ctx.create_descriptor_set(
            pipeline,
            0,
            &[
                DescriptorWrite::storage_buffer(0, 0, self.mesh_descriptor_buffer),
                DescriptorWrite::storage_buffer(1, 0, self.instance_buffer),
            ],
        );
        ctx.bind_pipeline_raw(pipeline);
        ctx.bind_descriptor_set(0, scene_set);
        ctx.bind_descriptor_set(2, self.bindless_set);
        ctx.push_constant_data(
            ShaderStageFlags::Vertex,
            0,
            &PreZPushConstants {
                view_projection: common.view_projection,
            },
        );
        ctx.draw_indirect_count(
            self.draw_call_buffer,
            0,
            self.draw_count_buffer,
            0,
            MAX_INSTANCES as u32,
            std::mem::size_of::<GfxDrawIndirectCommand>() as u32,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PreZPass — group: declares DepthPreZ, adds PreZFill child
// ─────────────────────────────────────────────────────────────────────────────

pub struct PreZPass {
    fill: PreZFillPass,
}

impl PreZPass {
    pub fn new() -> Self {
        Self { fill: PreZFillPass::new() }
    }
}

impl RenderPass for PreZPass {
    fn name(&self) -> &str { "PreZPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.fill.init(backend, globals);
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // Fixed resolution — independent of backbuffer size.
        // Powers-of-two in both dimensions → HiZ pyramid reduces exactly at every level,
        // no odd-dimension edge cases.  The PreZ renders the same frustum as the main pass
        // so HiZ UV [0,1] maps directly to screen UV [0,1] without any scale factor.
        builder.declare_image_output("DepthPreZ", ImageDesc {
            width:        crate::constants::HIZ_PREZ_WIDTH,
            height:       crate::constants::HIZ_PREZ_HEIGHT,
            depth:        1,
            format:       Format::D32_FLOAT,
            mip_levels:   1,
            array_layers: 1,
            usage:        ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::SAMPLED,
            view_type:    ImageViewType::Type2D,
            swizzle:      ComponentMapping::default(),
            clear_value:  None,
        });

        builder.add_pass(&mut self.fill);
    }

    // No execute() — pure group.
}

// ─────────────────────────────────────────────────────────────────────────────
// DrawCallGenPass — group: declares transient draw buffers + VisibilityBitset,
//                          clears counts and bitset, then dispatches occlusion cull.
//
// Both draw buffers are fully rewritten every frame — transient, not persistent.
// The VisibilityBitset (history output, declared in render_graph.rs) is cleared
// here before DrawCallGenCompute writes to it via InterlockedOr.
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

        // Resolve the VisibilityBitset (declared as history in render_graph.rs).
        let visibility_bitset = builder.resolve_buffer("VisibilityBitset");

        builder.add_owned_pass(crate::render_graph::ClearBufferPass {
            name:   "ClearDrawCount".to_string(),
            buffer: self.draw_count_buffer,
        });
        builder.add_owned_pass(crate::render_graph::ClearBufferPass {
            name:   "ClearVisibilityBitset".to_string(),
            buffer: visibility_bitset,
        });

        builder.add_pass(&mut self.compute);
    }

    // No execute() — pure group.
}
