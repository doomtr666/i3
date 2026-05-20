use std::mem::size_of;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

use i3_brickmap::{
    BrickmapClipmapState, GpuBrickJob, GpuBvhNode, GpuPrimitive, InvalidationSphere, LevelState,
    BRICK_DWORDS, GRID_VOL, MAX_BRICKS_PER_LEVEL, NUM_LEVELS, pack_bvh, pack_scene,
};
use i3_gfx::prelude::*;
use i3_io::asset::AssetLoader;
use i3_voxel::SdfScene;

const MAX_SPHERES: usize = 64;
const MAX_PRIMS:   usize = 256;

// ─── Shared GPU buffers ───────────────────────────────────────────────────────

pub struct ClipmapGpuBuffers {
    /// GpuOnly — dirty/cull passes write; G-buffer shader reads.
    pub page_table_buf:     BackendBuffer,
    /// GpuOnly — bake writes SDF values; raymarcher reads.
    pub sdf_atlas_buf:      BackendBuffer,
    /// GpuOnly — bake writes material IDs; raymarcher reads.
    pub mat_atlas_buf:      BackendBuffer,
    /// GpuOnly — dirty pass writes uint (lev_flat) per dirty brick.
    pub dirty_list_buf:     BackendBuffer,
    /// GpuOnly — per-level alloc counters + free stack.
    pub alloc_buf:          BackendBuffer,
    /// GpuOnly — single u32 atomic dirty-brick counter.
    pub dirty_count_buf:    BackendBuffer,
    /// GpuOnly — single u32 atomic job counter.
    pub job_count_buf:      BackendBuffer,
    /// GpuOnly + INDIRECT — VkDispatchIndirectCommand for cull pass.
    pub dispatch_dirty_buf: BackendBuffer,
    /// GpuOnly + INDIRECT — VkDispatchIndirectCommand for bake pass.
    pub dispatch_jobs_buf:  BackendBuffer,
    /// GpuOnly — GpuBrickJob array written by cull, read by bake.
    pub jobs_buf:           BackendBuffer,
    /// CpuToGpu — GpuPrimitive array, uploaded each frame.
    pub prims_buf:          BackendBuffer,
    /// CpuToGpu — GpuBvhNode array, uploaded when scene changes.
    pub bvh_buf:            BackendBuffer,
    /// CpuToGpu — LevelState array, uploaded each frame.
    pub level_state_buf:    BackendBuffer,
    /// CpuToGpu — InvalidationSphere array, uploaded each frame.
    pub invalidation_buf:   BackendBuffer,
}

impl ClipmapGpuBuffers {
    pub fn new(backend: &mut dyn RenderBackend) -> Self {
        let atlas_bytes      = (NUM_LEVELS * MAX_BRICKS_PER_LEVEL * BRICK_DWORDS * 4) as u64;
        let max_jobs         = NUM_LEVELS * MAX_BRICKS_PER_LEVEL;
        let alloc_stride     = 8 + MAX_BRICKS_PER_LEVEL * 4;
        let alloc_bytes      = (NUM_LEVELS * alloc_stride) as u64;
        let dirty_list_bytes = (NUM_LEVELS * GRID_VOL * 4) as u64;
        let bvh_max_bytes    = (65536 * size_of::<GpuBvhNode>()) as u64;
        let level_state_bytes = (NUM_LEVELS * size_of::<LevelState>()) as u64;
        let invalidation_bytes = (MAX_SPHERES * size_of::<InvalidationSphere>()) as u64;

        Self {
            page_table_buf: backend.create_buffer(&BufferDesc {
                size:   (NUM_LEVELS * GRID_VOL * 4) as u64,
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::GpuOnly,
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
            dirty_list_buf: backend.create_buffer(&BufferDesc {
                size:   dirty_list_bytes,
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::GpuOnly,
            }),
            alloc_buf: backend.create_buffer(&BufferDesc {
                size:   alloc_bytes,
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::GpuOnly,
            }),
            dirty_count_buf: backend.create_buffer(&BufferDesc {
                size:   4,
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::GpuOnly,
            }),
            job_count_buf: backend.create_buffer(&BufferDesc {
                size:   4,
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::GpuOnly,
            }),
            dispatch_dirty_buf: backend.create_buffer(&BufferDesc {
                size:   12,
                usage:  BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::INDIRECT_BUFFER,
                memory: MemoryType::GpuOnly,
            }),
            dispatch_jobs_buf: backend.create_buffer(&BufferDesc {
                size:   12,
                usage:  BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::INDIRECT_BUFFER,
                memory: MemoryType::GpuOnly,
            }),
            jobs_buf: backend.create_buffer(&BufferDesc {
                size:   (max_jobs * size_of::<GpuBrickJob>()) as u64,
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::GpuOnly,
            }),
            prims_buf: backend.create_buffer(&BufferDesc {
                size:   (MAX_PRIMS * size_of::<GpuPrimitive>()) as u64,
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::CpuToGpu,
            }),
            bvh_buf: backend.create_buffer(&BufferDesc {
                size:   bvh_max_bytes,
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::CpuToGpu,
            }),
            level_state_buf: backend.create_buffer(&BufferDesc {
                size:   level_state_bytes,
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::CpuToGpu,
            }),
            invalidation_buf: backend.create_buffer(&BufferDesc {
                size:   invalidation_bytes.max(16),
                usage:  BufferUsageFlags::STORAGE_BUFFER,
                memory: MemoryType::CpuToGpu,
            }),
        }
    }
}

// ─── Push constants ───────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
struct DirtyPC { sphere_count: u32, _pad: [u32; 3] }

#[repr(C)]
#[derive(Clone, Copy)]
struct CullPC { prim_count: u32, bvh_root: u32, _pad: [u32; 2] }

#[repr(C)]
#[derive(Clone, Copy)]
struct BakePC { prim_count: u32, _pad: [u32; 3] }

// ─── Shared per-frame data (set by SetupPass, read by later passes) ───────────

struct BrickmapShared {
    gpu_buffers:   Arc<ClipmapGpuBuffers>,
    clipmap_state: Arc<RwLock<BrickmapClipmapState>>,
    clipmap_scene: Arc<RwLock<SdfScene>>,
    prim_count:   AtomicU32,
    bvh_root:     AtomicU32,
    sphere_count: AtomicU32,
}

// ─── Pipeline loader helper ───────────────────────────────────────────────────

fn load_pipeline(
    globals: &mut PassBuilder,
    backend: &mut dyn RenderBackend,
    name: &str,
) -> Option<BackendPipeline> {
    let loader = globals.try_consume::<Arc<AssetLoader>>("AssetLoader").cloned()?;
    match loader.load::<i3_io::pipeline_asset::PipelineAsset>(name).wait_loaded() {
        Ok(asset) => Some(backend.create_compute_pipeline_from_baked(
            &asset.reflection_data,
            &asset.bytecode,
        )),
        Err(e) => {
            tracing::error!("brickmap: failed to load pipeline '{name}': {e}");
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 0: BrickmapSetupPass — master importer + GPU reset + first-frame init
//
// This is the ONLY pass that calls import_buffer for ALL shared buffers.
// Its declare() publishes every handle into the parent scope (is_output: true).
// All subsequent passes call resolve_buffer(name) to obtain the SAME SymbolId,
// allowing the sync planner to detect cross-pass buffer dependencies and emit
// the correct Vulkan barriers automatically.
//
// CPU: uploads BVH, primitives, level states, invalidation spheres each frame.
// GPU: reset dispatch — zeroes dirty_count / job_count / dispatch_{dirty,jobs}.
// Frame 1 only: clears GpuOnly persistent buffers via TRANSFER_WRITE.
// ─────────────────────────────────────────────────────────────────────────────

pub struct BrickmapSetupPass {
    shared:              Arc<BrickmapShared>,
    pipeline:            Option<BackendPipeline>,
    first_frame:         AtomicBool,
    // All virtual handles — resolved by later passes via resolve_buffer.
    page_table_virt:     BufferHandle,
    sdf_atlas_virt:      BufferHandle,
    mat_atlas_virt:      BufferHandle,
    dirty_list_virt:     BufferHandle,
    alloc_virt:          BufferHandle,
    dirty_count_virt:    BufferHandle,
    job_count_virt:      BufferHandle,
    dispatch_dirty_virt: BufferHandle,
    dispatch_jobs_virt:  BufferHandle,
    jobs_virt:           BufferHandle,
    bvh_virt:            BufferHandle,
    level_state_virt:    BufferHandle,
    invalidation_virt:   BufferHandle,
    prims_virt:          BufferHandle,
}

impl RenderPass for BrickmapSetupPass {
    fn name(&self) -> &str { "BrickmapSetupPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.pipeline = load_pipeline(globals, backend, "brickmap_reset");
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let gb = &self.shared.gpu_buffers;

        // ── Master importer: all handles published to parent scope ────────────
        // Later passes call resolve_buffer(same_name) → same SymbolId →
        // sync planner detects RAW/WAW/WAR dependencies → correct barriers.
        self.dirty_count_virt    = builder.import_buffer("BakeDirtyCount",    gb.dirty_count_buf);
        self.job_count_virt      = builder.import_buffer("BakeJobCount",      gb.job_count_buf);
        self.dispatch_dirty_virt = builder.import_buffer("BakeDispatchDirty", gb.dispatch_dirty_buf);
        self.dispatch_jobs_virt  = builder.import_buffer("BakeDispatchJobs",  gb.dispatch_jobs_buf);
        self.bvh_virt            = builder.import_buffer("BakeBvh",           gb.bvh_buf);
        self.level_state_virt    = builder.import_buffer("BakeLevelState",    gb.level_state_buf);
        self.invalidation_virt   = builder.import_buffer("BakeInvalidation",  gb.invalidation_buf);
        self.prims_virt          = builder.import_buffer("BakePrims",         gb.prims_buf);
        self.page_table_virt     = builder.import_buffer("ClipmapPageTable",  gb.page_table_buf);
        self.alloc_virt          = builder.import_buffer("BakeAlloc",         gb.alloc_buf);
        self.sdf_atlas_virt      = builder.import_buffer("ClipmapSdfAtlas",   gb.sdf_atlas_buf);
        self.mat_atlas_virt      = builder.import_buffer("ClipmapMatAtlas",   gb.mat_atlas_buf);
        self.dirty_list_virt     = builder.import_buffer("BakeDirtyList",     gb.dirty_list_buf);
        self.jobs_virt           = builder.import_buffer("BakeJobs",          gb.jobs_buf);

        // GPU reset: zeroes dirty_count / job_count / dispatch_{dirty,jobs}.
        builder.write_buffer(self.dirty_count_virt,    ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.job_count_virt,      ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.dispatch_dirty_virt, ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.dispatch_jobs_virt,  ResourceUsage::SHADER_WRITE);

        // CPU uploads to CpuToGpu buffers — declare SHADER_READ so the sync
        // planner tracks the correct state for later passes that read them.
        builder.read_buffer(self.bvh_virt,          ResourceUsage::SHADER_READ);
        builder.read_buffer(self.level_state_virt,  ResourceUsage::SHADER_READ);
        builder.read_buffer(self.invalidation_virt, ResourceUsage::SHADER_READ);
        builder.read_buffer(self.prims_virt,        ResourceUsage::SHADER_READ);

        // First frame only: clear GpuOnly persistent buffers (TRANSFER_WRITE).
        // These are DIFFERENT resources from the SHADER_WRITE targets above,
        // so no mid-pass barrier is required.
        if self.first_frame.load(Ordering::Relaxed) {
            builder.write_buffer(self.page_table_virt, ResourceUsage::TRANSFER_WRITE);
            builder.write_buffer(self.alloc_virt,       ResourceUsage::TRANSFER_WRITE);
            builder.write_buffer(self.sdf_atlas_virt,   ResourceUsage::TRANSFER_WRITE);
            builder.write_buffer(self.mat_atlas_virt,   ResourceUsage::TRANSFER_WRITE);
        }
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        // ── CPU: collect and upload per-frame data ────────────────────────────
        let scene     = self.shared.clipmap_scene.read().unwrap();
        let bvh_nodes = pack_bvh(&scene);
        let prims     = pack_scene(&scene);
        let (level_states, spheres) = {
            let mut cm = self.shared.clipmap_state.write().unwrap();
            cm.collect_frame_upload()
        };

        let prim_count   = prims.len() as u32;
        let sphere_count = spheres.len() as u32;
        let bvh_root     = if bvh_nodes.is_empty() { u32::MAX } else { 0u32 };

        self.shared.prim_count.store(prim_count,     Ordering::Release);
        self.shared.bvh_root.store(bvh_root,         Ordering::Release);
        self.shared.sphere_count.store(sphere_count, Ordering::Release);

        if !bvh_nodes.is_empty() {
            let ptr = ctx.map_buffer(self.bvh_virt);
            if !ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bvh_nodes.as_ptr(),
                        ptr as *mut GpuBvhNode,
                        bvh_nodes.len().min(65536),
                    );
                }
                ctx.unmap_buffer(self.bvh_virt);
            }
        }
        {
            let ptr = ctx.map_buffer(self.prims_virt);
            if !ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        prims.as_ptr(),
                        ptr as *mut GpuPrimitive,
                        prims.len().min(MAX_PRIMS),
                    );
                }
                ctx.unmap_buffer(self.prims_virt);
            }
        }
        {
            let ptr = ctx.map_buffer(self.level_state_virt);
            if !ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        level_states.as_ptr(),
                        ptr as *mut LevelState,
                        NUM_LEVELS,
                    );
                }
                ctx.unmap_buffer(self.level_state_virt);
            }
        }
        {
            let ptr = ctx.map_buffer(self.invalidation_virt);
            if !ptr.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        spheres.as_ptr(),
                        ptr as *mut InvalidationSphere,
                        spheres.len().min(MAX_SPHERES),
                    );
                }
                ctx.unmap_buffer(self.invalidation_virt);
            }
        }
        drop(scene);

        // ── Frame 1: clear GpuOnly buffers to initial state ───────────────────
        if self.first_frame.swap(false, Ordering::AcqRel) {
            // page_table: PAGE_EMPTY = 0xFFFFFFFF (no brick allocated).
            ctx.clear_buffer(self.page_table_virt, 0xFFFF_FFFFu32);
            // alloc_buf: brick_count = 0, free_top = 0, stack = empty per level.
            ctx.clear_buffer(self.alloc_virt, 0u32);
            // sdf_atlas: 0xFF per byte → SDF = +max (fully outside, safe default).
            ctx.clear_buffer(self.sdf_atlas_virt, 0xFF_FF_FF_FFu32);
            // mat_atlas: material 0 — safe default.
            ctx.clear_buffer(self.mat_atlas_virt, 0u32);
        }

        // ── GPU: reset dispatch ───────────────────────────────────────────────
        let Some(pl) = self.pipeline else { return; };
        ctx.bind_pipeline_raw(pl);
        let set = ctx.create_descriptor_set(pl, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.dirty_count_virt),
            DescriptorWrite::storage_buffer(0, 1, self.job_count_virt),
            DescriptorWrite::storage_buffer(0, 2, self.dispatch_dirty_virt),
            DescriptorWrite::storage_buffer(0, 3, self.dispatch_jobs_virt),
        ]);
        ctx.bind_descriptor_set(0, set);
        ctx.dispatch(1, 1, 1);

        tracing::debug!(
            "BrickmapSetupPass: {} prims, {} BVH nodes, {} spheres",
            prim_count,
            bvh_nodes.len(),
            sphere_count,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1: BrickmapDirtyPass — detect dirty bricks (camera shift + spheres)
// ─────────────────────────────────────────────────────────────────────────────

pub struct BrickmapDirtyPass {
    shared:           Arc<BrickmapShared>,
    pipeline:         Option<BackendPipeline>,
    level_state_virt: BufferHandle,
    invalidation_virt: BufferHandle,
    page_table_virt:  BufferHandle,
    dirty_list_virt:  BufferHandle,
    dirty_count_virt: BufferHandle,
    alloc_virt:       BufferHandle,
}

impl RenderPass for BrickmapDirtyPass {
    fn name(&self) -> &str { "BrickmapDirtyPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.pipeline = load_pipeline(globals, backend, "brickmap_dirty");
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // Resolve handles from SetupPass (same SymbolId → correct barriers).
        self.level_state_virt  = builder.resolve_buffer("BakeLevelState");
        self.invalidation_virt = builder.resolve_buffer("BakeInvalidation");
        self.page_table_virt   = builder.resolve_buffer("ClipmapPageTable");
        self.dirty_list_virt   = builder.resolve_buffer("BakeDirtyList");
        self.dirty_count_virt  = builder.resolve_buffer("BakeDirtyCount");
        self.alloc_virt        = builder.resolve_buffer("BakeAlloc");

        builder.read_buffer(self.level_state_virt,  ResourceUsage::SHADER_READ);
        builder.read_buffer(self.invalidation_virt, ResourceUsage::SHADER_READ);
        // page_table: read old_slot before writing PAGE_EMPTY.
        builder.read_buffer(self.page_table_virt,   ResourceUsage::SHADER_READ);
        builder.write_buffer(self.page_table_virt,  ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.dirty_list_virt,  ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.dirty_count_virt, ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.alloc_virt,       ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        let Some(pl) = self.pipeline else { return; };
        ctx.bind_pipeline_raw(pl);
        let set = ctx.create_descriptor_set(pl, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.level_state_virt),
            DescriptorWrite::storage_buffer(0, 1, self.invalidation_virt),
            DescriptorWrite::storage_buffer(0, 2, self.page_table_virt),
            DescriptorWrite::storage_buffer(0, 3, self.dirty_list_virt),
            DescriptorWrite::storage_buffer(0, 4, self.dirty_count_virt),
            DescriptorWrite::storage_buffer(0, 5, self.alloc_virt),
        ]);
        ctx.bind_descriptor_set(0, set);
        let sphere_count = self.shared.sphere_count.load(Ordering::Acquire);
        let pc = DirtyPC { sphere_count, _pad: [0; 3] };
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &pc);
        let total = (NUM_LEVELS * GRID_VOL) as u32;
        ctx.dispatch((total + 63) / 64, 1, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2: BrickmapIndirectDirtyPass — copy dirty_count → dispatch_dirty.x
// ─────────────────────────────────────────────────────────────────────────────

pub struct BrickmapIndirectDirtyPass {
    pipeline:            Option<BackendPipeline>,
    dirty_count_virt:    BufferHandle,
    dispatch_dirty_virt: BufferHandle,
}

impl RenderPass for BrickmapIndirectDirtyPass {
    fn name(&self) -> &str { "BrickmapIndirectDirtyPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.pipeline = load_pipeline(globals, backend, "brickmap_indirect");
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.dirty_count_virt    = builder.resolve_buffer("BakeDirtyCount");
        self.dispatch_dirty_virt = builder.resolve_buffer("BakeDispatchDirty");

        builder.read_buffer(self.dirty_count_virt,     ResourceUsage::SHADER_READ);
        builder.write_buffer(self.dispatch_dirty_virt, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        let Some(pl) = self.pipeline else { return; };
        ctx.bind_pipeline_raw(pl);
        let set = ctx.create_descriptor_set(pl, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.dirty_count_virt),
            DescriptorWrite::storage_buffer(0, 1, self.dispatch_dirty_virt),
        ]);
        ctx.bind_descriptor_set(0, set);
        ctx.dispatch(1, 1, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3: BrickmapCullPass — BVH traversal + center cull + slot alloc + emit jobs
// Dispatched indirectly from dispatch_dirty_buf.
// ─────────────────────────────────────────────────────────────────────────────

pub struct BrickmapCullPass {
    shared:              Arc<BrickmapShared>,
    pipeline:            Option<BackendPipeline>,
    bvh_virt:            BufferHandle,
    prims_virt:          BufferHandle,
    level_state_virt:    BufferHandle,
    dirty_list_virt:     BufferHandle,
    dispatch_dirty_virt: BufferHandle,
    page_table_virt:     BufferHandle,
    alloc_virt:          BufferHandle,
    jobs_virt:           BufferHandle,
    job_count_virt:      BufferHandle,
}

impl RenderPass for BrickmapCullPass {
    fn name(&self) -> &str { "BrickmapCullPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.pipeline = load_pipeline(globals, backend, "brickmap_cull");
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        // All handles resolved from SetupPass's published scope.
        self.bvh_virt            = builder.resolve_buffer("BakeBvh");
        self.prims_virt          = builder.resolve_buffer("BakePrims");
        self.level_state_virt    = builder.resolve_buffer("BakeLevelState");
        self.dirty_list_virt     = builder.resolve_buffer("BakeDirtyList");
        self.dispatch_dirty_virt = builder.resolve_buffer("BakeDispatchDirty");
        self.page_table_virt     = builder.resolve_buffer("ClipmapPageTable");
        self.alloc_virt          = builder.resolve_buffer("BakeAlloc");
        self.jobs_virt           = builder.resolve_buffer("BakeJobs");
        self.job_count_virt      = builder.resolve_buffer("BakeJobCount");

        builder.read_buffer(self.bvh_virt,            ResourceUsage::SHADER_READ);
        builder.read_buffer(self.prims_virt,          ResourceUsage::SHADER_READ);
        builder.read_buffer(self.level_state_virt,    ResourceUsage::SHADER_READ);
        builder.read_buffer(self.dirty_list_virt,     ResourceUsage::SHADER_READ);
        builder.read_buffer(self.dispatch_dirty_virt, ResourceUsage::INDIRECT_READ);
        builder.write_buffer(self.page_table_virt,    ResourceUsage::SHADER_WRITE);
        // alloc_buf: CAS reads (free_top, stack entries) then writes (pop/push/alloc).
        builder.read_buffer(self.alloc_virt,          ResourceUsage::SHADER_READ);
        builder.write_buffer(self.alloc_virt,         ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.jobs_virt,          ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.job_count_virt,     ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        let Some(pl) = self.pipeline else { return; };
        ctx.bind_pipeline_raw(pl);
        let set = ctx.create_descriptor_set(pl, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.bvh_virt),
            DescriptorWrite::storage_buffer(0, 1, self.prims_virt),
            DescriptorWrite::storage_buffer(0, 2, self.level_state_virt),
            DescriptorWrite::storage_buffer(0, 3, self.dirty_list_virt),
            DescriptorWrite::storage_buffer(0, 4, self.page_table_virt),
            DescriptorWrite::storage_buffer(0, 5, self.alloc_virt),
            DescriptorWrite::storage_buffer(0, 6, self.jobs_virt),
            DescriptorWrite::storage_buffer(0, 7, self.job_count_virt),
        ]);
        ctx.bind_descriptor_set(0, set);
        let prim_count = self.shared.prim_count.load(Ordering::Acquire);
        let bvh_root   = self.shared.bvh_root.load(Ordering::Acquire);
        let pc = CullPC { prim_count, bvh_root, _pad: [0; 2] };
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &pc);
        ctx.dispatch_indirect(self.dispatch_dirty_virt, 0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 4: BrickmapIndirectJobsPass — copy job_count → dispatch_jobs.x
// ─────────────────────────────────────────────────────────────────────────────

pub struct BrickmapIndirectJobsPass {
    pipeline:           Option<BackendPipeline>,
    job_count_virt:     BufferHandle,
    dispatch_jobs_virt: BufferHandle,
}

impl RenderPass for BrickmapIndirectJobsPass {
    fn name(&self) -> &str { "BrickmapIndirectJobsPass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.pipeline = load_pipeline(globals, backend, "brickmap_indirect");
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.job_count_virt     = builder.resolve_buffer("BakeJobCount");
        self.dispatch_jobs_virt = builder.resolve_buffer("BakeDispatchJobs");

        builder.read_buffer(self.job_count_virt,      ResourceUsage::SHADER_READ);
        builder.write_buffer(self.dispatch_jobs_virt, ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        let Some(pl) = self.pipeline else { return; };
        ctx.bind_pipeline_raw(pl);
        let set = ctx.create_descriptor_set(pl, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.job_count_virt),
            DescriptorWrite::storage_buffer(0, 1, self.dispatch_jobs_virt),
        ]);
        ctx.bind_descriptor_set(0, set);
        ctx.dispatch(1, 1, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 5: BrickmapBakePass — evaluate SDF + material at 9³ voxels per brick
// Dispatched indirectly from dispatch_jobs_buf.
// ─────────────────────────────────────────────────────────────────────────────

pub struct BrickmapBakePass {
    shared:             Arc<BrickmapShared>,
    pipeline:           Option<BackendPipeline>,
    jobs_virt:          BufferHandle,
    prims_virt:         BufferHandle,
    dispatch_jobs_virt: BufferHandle,
    sdf_atlas_virt:     BufferHandle,
    mat_atlas_virt:     BufferHandle,
}

impl RenderPass for BrickmapBakePass {
    fn name(&self) -> &str { "BrickmapBakePass" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.pipeline = load_pipeline(globals, backend, "brickmap_bake");
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.jobs_virt          = builder.resolve_buffer("BakeJobs");
        self.prims_virt         = builder.resolve_buffer("BakePrims");
        self.dispatch_jobs_virt = builder.resolve_buffer("BakeDispatchJobs");
        self.sdf_atlas_virt     = builder.resolve_buffer("ClipmapSdfAtlas");
        self.mat_atlas_virt     = builder.resolve_buffer("ClipmapMatAtlas");

        builder.read_buffer(self.jobs_virt,           ResourceUsage::SHADER_READ);
        builder.read_buffer(self.prims_virt,          ResourceUsage::SHADER_READ);
        builder.read_buffer(self.dispatch_jobs_virt,  ResourceUsage::INDIRECT_READ);
        builder.write_buffer(self.sdf_atlas_virt,     ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.mat_atlas_virt,     ResourceUsage::SHADER_WRITE);
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
        let Some(pl) = self.pipeline else { return; };
        ctx.bind_pipeline_raw(pl);
        let set = ctx.create_descriptor_set(pl, 0, &[
            DescriptorWrite::storage_buffer(0, 0, self.jobs_virt),
            DescriptorWrite::storage_buffer(0, 1, self.prims_virt),
            DescriptorWrite::storage_buffer(0, 2, self.sdf_atlas_virt),
            DescriptorWrite::storage_buffer(0, 3, self.mat_atlas_virt),
        ]);
        ctx.bind_descriptor_set(0, set);
        let prim_count = self.shared.prim_count.load(Ordering::Acquire);
        let pc = BakePC { prim_count, _pad: [0; 3] };
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &pc);
        ctx.dispatch_indirect(self.dispatch_jobs_virt, 0);
    }
}

// ─── Factory ──────────────────────────────────────────────────────────────────

pub fn create_brickmap_passes(
    clipmap_state: Arc<RwLock<BrickmapClipmapState>>,
    clipmap_scene: Arc<RwLock<SdfScene>>,
    gpu_buffers:   Arc<ClipmapGpuBuffers>,
) -> Vec<Box<dyn RenderPass>> {
    let shared = Arc::new(BrickmapShared {
        gpu_buffers:   gpu_buffers.clone(),
        clipmap_state: clipmap_state.clone(),
        clipmap_scene: clipmap_scene.clone(),
        prim_count:   AtomicU32::new(0),
        bvh_root:     AtomicU32::new(u32::MAX),
        sphere_count: AtomicU32::new(0),
    });

    let inv = BufferHandle::INVALID;

    vec![
        Box::new(BrickmapSetupPass {
            shared:              shared.clone(),
            pipeline:            None,
            first_frame:         AtomicBool::new(true),
            page_table_virt:     inv,
            sdf_atlas_virt:      inv,
            mat_atlas_virt:      inv,
            dirty_list_virt:     inv,
            alloc_virt:          inv,
            dirty_count_virt:    inv,
            job_count_virt:      inv,
            dispatch_dirty_virt: inv,
            dispatch_jobs_virt:  inv,
            jobs_virt:           inv,
            bvh_virt:            inv,
            level_state_virt:    inv,
            invalidation_virt:   inv,
            prims_virt:          inv,
        }),
        Box::new(BrickmapDirtyPass {
            shared:            shared.clone(),
            pipeline:          None,
            level_state_virt:  inv,
            invalidation_virt: inv,
            page_table_virt:   inv,
            dirty_list_virt:   inv,
            dirty_count_virt:  inv,
            alloc_virt:        inv,
        }),
        Box::new(BrickmapIndirectDirtyPass {
            pipeline:            None,
            dirty_count_virt:    inv,
            dispatch_dirty_virt: inv,
        }),
        Box::new(BrickmapCullPass {
            shared:              shared.clone(),
            pipeline:            None,
            bvh_virt:            inv,
            prims_virt:          inv,
            level_state_virt:    inv,
            dirty_list_virt:     inv,
            dispatch_dirty_virt: inv,
            page_table_virt:     inv,
            alloc_virt:          inv,
            jobs_virt:           inv,
            job_count_virt:      inv,
        }),
        Box::new(BrickmapIndirectJobsPass {
            pipeline:           None,
            job_count_virt:     inv,
            dispatch_jobs_virt: inv,
        }),
        Box::new(BrickmapBakePass {
            shared:             shared.clone(),
            pipeline:           None,
            jobs_virt:          inv,
            prims_virt:         inv,
            dispatch_jobs_virt: inv,
            sdf_atlas_virt:     inv,
            mat_atlas_virt:     inv,
        }),
    ]
}
