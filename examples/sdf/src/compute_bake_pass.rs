use std::mem::size_of;
use std::sync::{Arc, RwLock};

use i3_brickmap::{GpuBrickJob, GpuPrimitive, BrickmapClipmapState, GRID_VOL, MAX_BRICKS_PER_LEVEL, NUM_LEVELS, BRICK_VOXELS};
use i3_gfx::prelude::*;
use i3_io::asset::AssetLoader;
use i3_voxel::SdfScene;

// ─── Shared GPU atlas buffers ─────────────────────────────────────────────────
// Owned here; shared as Arc with ClipmapGBufferPass which reads them.

pub struct ClipmapGpuBuffers {
    /// CpuToGpu — CPU writes page-table allocation info each frame.
    pub page_table_buf: BackendBuffer,
    /// GpuOnly — compute shader writes SDF values; raymarcher reads them.
    pub sdf_atlas_buf:  BackendBuffer,
    /// GpuOnly — compute shader writes material IDs; raymarcher reads them.
    pub mat_atlas_buf:  BackendBuffer,
}

impl ClipmapGpuBuffers {
    pub fn new(backend: &mut dyn RenderBackend) -> Self {
        let page_table_bytes = (NUM_LEVELS * GRID_VOL * 4) as u64;
        let atlas_bytes      = (NUM_LEVELS * MAX_BRICKS_PER_LEVEL * BRICK_VOXELS * 4) as u64;

        Self {
            page_table_buf: backend.create_buffer(&BufferDesc {
                size:   page_table_bytes,
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::CpuToGpu,
            }),
            sdf_atlas_buf: backend.create_buffer(&BufferDesc {
                size:   atlas_bytes,
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::GpuOnly,
            }),
            mat_atlas_buf: backend.create_buffer(&BufferDesc {
                size:   atlas_bytes.max(4),
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::GpuOnly,
            }),
        }
    }
}

// ─── Push constants ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
struct BakePushConstants {
    prim_count: u32,
    _pad:       [u32; 3],
}

// ─── ComputeBakePass ─────────────────────────────────────────────────────────

pub struct ComputeBakePass {
    pub clipmap_state: Arc<RwLock<BrickmapClipmapState>>,
    pub clipmap_scene: Arc<RwLock<SdfScene>>,
    pub gpu_buffers:   Arc<ClipmapGpuBuffers>,

    // Pipeline
    pipeline: Option<BackendPipeline>,

    // Per-frame CPU→GPU upload buffers
    jobs_buf: BackendBuffer,
    prims_buf: BackendBuffer,

    // Virtual handles (re-resolved each frame in declare)
    sdf_atlas_virt: BufferHandle,
    mat_atlas_virt: BufferHandle,
    jobs_virt:      BufferHandle,
    prims_virt:     BufferHandle,
}

impl ComputeBakePass {
    pub fn new(
        clipmap_state: Arc<RwLock<BrickmapClipmapState>>,
        clipmap_scene: Arc<RwLock<SdfScene>>,
        gpu_buffers:   Arc<ClipmapGpuBuffers>,
    ) -> Self {
        Self {
            clipmap_state,
            clipmap_scene,
            gpu_buffers,
            pipeline:       None,
            jobs_buf:       BackendBuffer(u64::MAX),
            prims_buf:      BackendBuffer(u64::MAX),
            sdf_atlas_virt: BufferHandle::INVALID,
            mat_atlas_virt: BufferHandle::INVALID,
            jobs_virt:      BufferHandle::INVALID,
            prims_virt:     BufferHandle::INVALID,
        }
    }
}

impl RenderPass for ComputeBakePass {
    fn name(&self) -> &str { "ComputeBakePass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        // Worst case: budget × NUM_LEVELS jobs per frame (6 × 256 = 1536 → round to 2048)
        let max_jobs  = 2048usize;
        let max_prims = 256usize;

        self.jobs_buf = backend.create_buffer(&BufferDesc {
            size:   (max_jobs  * size_of::<GpuBrickJob>()) as u64,
            usage:  BufferUsageFlags::STORAGE_BUFFER,
            memory: MemoryType::CpuToGpu,
        });
        self.prims_buf = backend.create_buffer(&BufferDesc {
            size:   (max_prims * size_of::<GpuPrimitive>()) as u64,
            usage:  BufferUsageFlags::STORAGE_BUFFER,
            memory: MemoryType::CpuToGpu,
        });

        let loader = globals.try_consume::<Arc<AssetLoader>>("AssetLoader").cloned();
        let Some(loader) = loader else {
            tracing::error!("ComputeBakePass: no AssetLoader");
            return;
        };
        match loader.load::<i3_io::pipeline_asset::PipelineAsset>("brickmap_bake").wait_loaded() {
            Ok(asset) => {
                self.pipeline = Some(backend.create_compute_pipeline_from_baked(
                    &asset.reflection_data,
                    &asset.bytecode,
                ));
            }
            Err(e) => tracing::error!("ComputeBakePass: failed to load pipeline: {e}"),
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.sdf_atlas_virt = builder.import_buffer("ClipmapSdfAtlas", self.gpu_buffers.sdf_atlas_buf);
        self.mat_atlas_virt = builder.import_buffer("ClipmapMatAtlas", self.gpu_buffers.mat_atlas_buf);
        self.jobs_virt      = builder.import_buffer("BakeJobs",        self.jobs_buf);
        self.prims_virt     = builder.import_buffer("BakePrims",       self.prims_buf);

        builder.write_buffer(self.sdf_atlas_virt, ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.mat_atlas_virt, ResourceUsage::SHADER_WRITE);
        builder.read_buffer(self.jobs_virt,  ResourceUsage::SHADER_READ);
        builder.read_buffer(self.prims_virt, ResourceUsage::SHADER_READ);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        let Some(pipeline) = self.pipeline else { return; };

        // ── CPU preparation ───────────────────────────────────────────────────
        let scene = self.clipmap_scene.read().unwrap();
        let (jobs, prims) = {
            let mut cm = self.clipmap_state.write().unwrap();
            cm.prepare_gpu_bake(&scene, 256)
        };

        if jobs.is_empty() { return; }

        let prim_count = prims.len() as u32;

        // ── Upload jobs ───────────────────────────────────────────────────────
        {
            let ptr = ctx.map_buffer(self.jobs_virt);
            if !ptr.is_null() {
                unsafe {
                    let dst = ptr as *mut GpuBrickJob;
                    let n = jobs.len().min(2048);
                    std::ptr::copy_nonoverlapping(jobs.as_ptr(), dst, n);
                }
                ctx.unmap_buffer(self.jobs_virt);
            }
        }

        // ── Upload primitives ─────────────────────────────────────────────────
        {
            let ptr = ctx.map_buffer(self.prims_virt);
            if !ptr.is_null() {
                unsafe {
                    let dst = ptr as *mut GpuPrimitive;
                    let n = prims.len().min(256);
                    std::ptr::copy_nonoverlapping(prims.as_ptr(), dst, n);
                }
                ctx.unmap_buffer(self.prims_virt);
            }
        }

        // ── Dispatch ──────────────────────────────────────────────────────────
        ctx.bind_pipeline_raw(pipeline);

        let set = ctx.create_descriptor_set(pipeline, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.jobs_virt),
            DescriptorWrite::storage_buffer(0, 1, self.prims_virt),
            DescriptorWrite::storage_buffer(0, 2, self.sdf_atlas_virt),
        ]);
        ctx.bind_descriptor_set(0, set);

        let pc = BakePushConstants { prim_count, _pad: [0; 3] };
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &pc);

        let dispatch_count = jobs.len().min(2048) as u32;
        ctx.dispatch(dispatch_count, 1, 1);

        tracing::debug!("ComputeBakePass: dispatched {}/{} bricks, {} prims", dispatch_count, jobs.len(), prim_count);
    }
}
