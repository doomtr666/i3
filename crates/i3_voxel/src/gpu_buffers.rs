use std::mem::size_of;

use i3_gfx::prelude::*;

use crate::{
    clipmap::{ClipLevelGpu, GRID_VOL, NUM_LEVELS},
    gpu_scene::{GpuBrickJob, GpuBvhNode, GpuPrimitive},
    noise_graph::VmOp,
    BRICK_BYTES, MAX_PRIMS, MAX_SVO_BRICKS, WEIGHT_BRICK_BYTES,
};

/// Max ops in a compiled noise graph (terrain VM bytecode). Matches evalGraph's cap.
pub const MAX_VM_OPS: usize = 256;

/// Frames the renderer keeps in flight. CPU-written buffers are ring-buffered
/// this many times so frame N's map-write never lands on a buffer that the GPU
/// is still reading for frame N-1 / N-2. The frame fence guarantees frame N-3 is
/// complete before frame N begins, so `buf[N % RING]` is always free to write.
pub const RING: usize = 3;

pub struct ClipmapBuffers {
    /// CpuToGpu, ring-buffered — STAGING for the clipmap page table (`[NUM_LEVELS·GRID_VOL]`
    /// u32 slots). The CPU map-writes here, then ClipmapSetupPass copies it into a frame-graph-
    /// declared device-local buffer ("ClipPageTableDevice") that the render reads. Host-
    /// visible memory is read over PCIe — bad for the marcher, which reads the page table at
    /// every cell crossing; the device copy keeps it in VRAM. Hence TRANSFER_SRC here.
    pub page_table: [BackendBuffer; RING],
    /// CpuToGpu, ring-buffered — per-level origin + voxel size (`[ClipLevelGpu; NUM_LEVELS]`).
    /// Tiny; read directly by the render (no device copy needed).
    pub clip_levels: [BackendBuffer; RING],
    /// CpuToGpu, ring-buffered — GpuBrickJob array (new bricks to bake this frame).
    pub jobs:      [BackendBuffer; RING],
    /// CpuToGpu, ring-buffered — GpuPrimitive array.
    pub prims:     [BackendBuffer; RING],
    /// CpuToGpu, ring-buffered — GpuBvhNode array.
    pub bvh:       [BackendBuffer; RING],
    /// CpuToGpu, ring-buffered — compiled noise-graph bytecode (terrain VM).
    pub vm_ops:    [BackendBuffer; RING],
    /// GpuOnly, single — geometry atlas: 8 bytes/voxel (f16 dist + oct normal +
    /// material + flags). Queue-ordered, no ring needed.
    pub geom_atlas_buf: BackendBuffer,
    /// GpuOnly, single — pre-mix weight atlas: 4 bytes/voxel (the baked terrain layer
    /// blend weights). Written by the bake (once/brick), read + weighted-sampled by the
    /// render. Byte address mirrors geom at half scale: `geom_addr >> 1`.
    pub weight_atlas_buf: BackendBuffer,
}

impl ClipmapBuffers {
    pub fn new(backend: &mut dyn RenderBackend) -> Self {
        let atlas_bytes      = (MAX_SVO_BRICKS as u64) * (BRICK_BYTES as u64);
        let weight_bytes     = (MAX_SVO_BRICKS as u64) * (WEIGHT_BRICK_BYTES as u64);
        let page_table_bytes = (NUM_LEVELS * GRID_VOL * size_of::<u32>()) as u64;
        let clip_levels_bytes= (NUM_LEVELS * size_of::<ClipLevelGpu>()) as u64;
        let jobs_bytes       = (MAX_SVO_BRICKS * size_of::<GpuBrickJob>()  as u32) as u64;
        let prims_bytes      = (MAX_PRIMS      * size_of::<GpuPrimitive>() as u32) as u64;
        let bvh_bytes        = (65536          * size_of::<GpuBvhNode>()         ) as u64;
        let vm_ops_bytes     = (MAX_VM_OPS     * size_of::<VmOp>()               ) as u64;

        fn storage(backend: &mut dyn RenderBackend, size: u64, mem: MemoryType, usage: BufferUsageFlags) -> BackendBuffer {
            backend.create_buffer(&BufferDesc { size, usage, memory: mem })
        }
        fn ring(backend: &mut dyn RenderBackend, size: u64, usage: BufferUsageFlags) -> [BackendBuffer; RING] {
            std::array::from_fn(|_| storage(backend, size, MemoryType::CpuToGpu, usage))
        }

        let store = BufferUsageFlags::STORAGE_BUFFER;
        Self {
            // Staging → copied to the device-local "ClipPageTableDevice" (declared by the
            // frame graph in ClipmapSetupPass), so it's a transfer source.
            page_table:  ring(backend, page_table_bytes, store | BufferUsageFlags::TRANSFER_SRC),
            clip_levels: ring(backend, clip_levels_bytes, store),
            jobs:        ring(backend, jobs_bytes,  store),
            prims:       ring(backend, prims_bytes, store),
            bvh:         ring(backend, bvh_bytes,   store),
            vm_ops:      ring(backend, vm_ops_bytes, store),
            geom_atlas_buf:   storage(backend, atlas_bytes,  MemoryType::GpuOnly, store),
            weight_atlas_buf: storage(backend, weight_bytes, MemoryType::GpuOnly, store),
        }
    }
}
