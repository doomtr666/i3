use std::mem::size_of;

use i3_gfx::prelude::*;

use crate::{BRICK_DWORDS, MAX_PRIMS, MAX_SVO_BRICKS, MAX_SVO_NODES, gpu_scene::{GpuBrickJob, GpuBvhNode, GpuPrimitive, GpuSvoNode}};

/// Frames the renderer keeps in flight. CPU-written buffers are ring-buffered
/// this many times so frame N's map-write never lands on a buffer that the GPU
/// is still reading for frame N-1 / N-2. The frame fence guarantees frame N-3 is
/// complete before frame N begins, so `buf[N % RING]` is always free to write.
pub const RING: usize = 3;

pub struct SvoGpuBuffers {
    /// CpuToGpu, ring-buffered — full SVO node pool (read every pixel by render).
    pub node_pool: [BackendBuffer; RING],
    /// CpuToGpu, ring-buffered — GpuBrickJob array (new bricks to bake this frame).
    pub jobs:      [BackendBuffer; RING],
    /// CpuToGpu, ring-buffered — GpuPrimitive array.
    pub prims:     [BackendBuffer; RING],
    /// CpuToGpu, ring-buffered — GpuBvhNode array.
    pub bvh:       [BackendBuffer; RING],
    /// GpuOnly, single — u8-packed SDF values; queue-ordered, no ring needed.
    pub sdf_atlas_buf: BackendBuffer,
    /// GpuOnly, single — u8-packed material IDs.
    pub mat_atlas_buf: BackendBuffer,
}

impl SvoGpuBuffers {
    pub fn new(backend: &mut dyn RenderBackend) -> Self {
        let atlas_bytes     = (MAX_SVO_BRICKS * BRICK_DWORDS as u32 * 4) as u64;
        let node_pool_bytes = (MAX_SVO_NODES  * size_of::<GpuSvoNode>()  as u32) as u64;
        let jobs_bytes      = (MAX_SVO_BRICKS * size_of::<GpuBrickJob>()  as u32) as u64;
        let prims_bytes     = (MAX_PRIMS      * size_of::<GpuPrimitive>() as u32) as u64;
        let bvh_bytes       = (65536          * size_of::<GpuBvhNode>()         ) as u64;

        fn storage(backend: &mut dyn RenderBackend, size: u64, mem: MemoryType) -> BackendBuffer {
            backend.create_buffer(&BufferDesc { size, usage: BufferUsageFlags::STORAGE_BUFFER, memory: mem })
        }
        fn ring(backend: &mut dyn RenderBackend, size: u64) -> [BackendBuffer; RING] {
            std::array::from_fn(|_| storage(backend, size, MemoryType::CpuToGpu))
        }

        Self {
            node_pool: ring(backend, node_pool_bytes),
            jobs:      ring(backend, jobs_bytes),
            prims:     ring(backend, prims_bytes),
            bvh:       ring(backend, bvh_bytes),
            sdf_atlas_buf: storage(backend, atlas_bytes,        MemoryType::GpuOnly),
            mat_atlas_buf: storage(backend, atlas_bytes.max(4), MemoryType::GpuOnly),
        }
    }
}
