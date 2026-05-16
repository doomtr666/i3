use std::sync::{Arc, RwLock};

use i3_gfx::prelude::*;
use i3_io::asset::AssetLoader;

use crate::sdf_tree::GpuSdfNode;

pub const MAX_SDF_NODES: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct PushConstants {
    node_count: u32,
    _pad: [u32; 3],
}

pub struct SdfGBufferPass {
    // Shared node list written by the app each frame
    pub shared: Arc<RwLock<Vec<GpuSdfNode>>>,

    // GBuffer image handles
    gbuffer_albedo:     ImageHandle,
    gbuffer_normal:     ImageHandle,
    gbuffer_roughmetal: ImageHandle,
    gbuffer_emissive:   ImageHandle,
    hiz_mip0:           ImageHandle,
    depth_buffer:       ImageHandle,

    // Common UBO
    common_buffer: BufferHandle,
    bindless_set:  DescriptorSetHandle,

    // SDF node storage buffer
    node_buffer:     BufferHandle,   // virtual handle resolved each frame
    physical_buffer: BackendBuffer,  // persistent CpuToGpu allocation
    node_count:      u32,

    pipeline: Option<BackendPipeline>,
}

impl SdfGBufferPass {
    pub fn new(shared: Arc<RwLock<Vec<GpuSdfNode>>>) -> Self {
        Self {
            shared,
            gbuffer_albedo:     ImageHandle::INVALID,
            gbuffer_normal:     ImageHandle::INVALID,
            gbuffer_roughmetal: ImageHandle::INVALID,
            gbuffer_emissive:   ImageHandle::INVALID,
            hiz_mip0:           ImageHandle::INVALID,
            depth_buffer:       ImageHandle::INVALID,
            common_buffer:      BufferHandle::INVALID,
            bindless_set:       DescriptorSetHandle(0),
            node_buffer:        BufferHandle::INVALID,
            physical_buffer:    BackendBuffer(u64::MAX),
            node_count:         0,
            pipeline:           None,
        }
    }
}

impl RenderPass for SdfGBufferPass {
    fn name(&self) -> &str {
        "SdfGBufferPass"
    }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        // Create persistent CpuToGpu node buffer
        self.physical_buffer = backend.create_buffer(&BufferDesc {
            size:   MAX_SDF_NODES * std::mem::size_of::<GpuSdfNode>() as u64,
            usage:  BufferUsageFlags::STORAGE_BUFFER,
            memory: MemoryType::CpuToGpu,
        });

        // Load pipeline
        let loader = globals
            .try_consume::<Arc<AssetLoader>>("AssetLoader")
            .cloned();
        let Some(loader) = loader else {
            tracing::error!("SdfGBufferPass: no AssetLoader in globals");
            return;
        };
        match loader
            .load::<i3_io::pipeline_asset::PipelineAsset>("sdf_gbuffer")
            .wait_loaded()
        {
            Ok(asset) => {
                let state = asset.state.as_ref().expect("SDF asset missing pipeline state");
                self.pipeline = Some(backend.create_graphics_pipeline_from_baked(
                    state,
                    &asset.reflection_data,
                    &asset.bytecode,
                ));
            }
            Err(e) => tracing::error!("SdfGBufferPass: failed to load sdf_gbuffer: {e}"),
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // Snapshot node count for the push constant
        self.node_count = self
            .shared
            .read()
            .map(|v| v.len() as u32)
            .unwrap_or(0);

        // Import the persistent physical buffer
        self.node_buffer = builder.import_buffer("SdfNodeBuffer", self.physical_buffer);

        // GBuffer image handles
        self.gbuffer_albedo     = builder.resolve_image("GBuffer_Albedo");
        self.gbuffer_normal     = builder.resolve_image("GBuffer_Normal");
        self.gbuffer_roughmetal = builder.resolve_image("GBuffer_RoughMetal");
        self.gbuffer_emissive   = builder.resolve_image("GBuffer_Emissive");
        self.hiz_mip0           = builder.resolve_image("HiZFinal");
        self.depth_buffer       = builder.resolve_image("DepthBuffer");
        self.common_buffer      = builder.resolve_buffer("CommonBuffer");
        self.bindless_set       = *builder.consume::<DescriptorSetHandle>("BindlessSet");

        builder.read_buffer(self.node_buffer,   ResourceUsage::SHADER_READ);
        builder.read_buffer(self.common_buffer, ResourceUsage::SHADER_READ);
        builder.write_image(self.gbuffer_albedo,     ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_normal,     ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_roughmetal, ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.gbuffer_emissive,   ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.hiz_mip0,           ResourceUsage::COLOR_ATTACHMENT);
        builder.write_image(self.depth_buffer,       ResourceUsage::DEPTH_STENCIL);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else {
            return;
        };

        // Upload nodes to the CpuToGpu buffer
        let nodes_guard = self.shared.read().unwrap();
        let count = nodes_guard.len().min(MAX_SDF_NODES as usize);
        if count > 0 {
            let byte_size = count * std::mem::size_of::<GpuSdfNode>();
            let ptr = ctx.map_buffer(self.node_buffer);
            if !ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        nodes_guard.as_ptr() as *const u8,
                        ptr,
                        byte_size,
                    );
                }
                ctx.unmap_buffer(self.node_buffer);
            }
        }
        drop(nodes_guard);

        ctx.bind_pipeline_raw(pipeline);

        // Set 0: SDF node storage buffer
        let node_set = ctx.create_descriptor_set(
            pipeline,
            0,
            &[DescriptorWrite::storage_buffer(0, 0, self.node_buffer)],
        );
        ctx.bind_descriptor_set(0, node_set);

        // Set 1: common UBO
        let common_set = ctx.create_descriptor_set(
            pipeline,
            1,
            &[DescriptorWrite::uniform_buffer(0, 0, self.common_buffer)],
        );
        ctx.bind_descriptor_set(1, common_set);

        // Set 2: bindless
        ctx.bind_descriptor_set(2, self.bindless_set);

        // Push constants
        let pc = PushConstants {
            node_count: count as u32,
            _pad: [0; 3],
        };
        ctx.push_constant_data(ShaderStageFlags::Vertex | ShaderStageFlags::Fragment, 0, &pc);

        ctx.draw(3, 0);
    }
}
