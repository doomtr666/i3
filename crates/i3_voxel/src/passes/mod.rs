mod setup_pass;
mod bake_pass;
mod render_pass;

pub use setup_pass::ClipmapSetupPass;
pub use bake_pass::ClipmapBakePass;
pub use render_pass::ClipmapRenderPass;

use std::sync::{Arc, RwLock, atomic::{AtomicBool, AtomicU32, AtomicUsize}};
use i3_gfx::prelude::*;
use crate::SdfScene;

use crate::{clipmap::ClipmapState, gpu_buffers::ClipmapBuffers};

// ─── Shared per-frame state ───────────────────────────────────────────────────

pub(crate) struct ClipmapShared {
    pub gpu_buffers: Arc<ClipmapBuffers>,
    pub clipmap:     Arc<RwLock<ClipmapState>>,
    pub sdf_scene:   Arc<RwLock<SdfScene>>,
    pub job_count:   AtomicU32,
    pub prim_count:  AtomicU32,
    pub terrain_on:  AtomicU32,   // 1 if the scene has a VolumeTerrain primitive
    pub vm_ops:      Arc<Vec<crate::VmOp>>,  // compiled terrain noise graph (static)
    pub terrain_mat: crate::TerrainMatParams, // terrain texture indices + blend params (static)
    pub bvh_root:    AtomicU32,
    pub debug_flags: Arc<AtomicU32>,
    pub enabled:     Arc<AtomicBool>,
    /// Active ring-buffer slot for this frame's CpuToGpu buffers.
    /// Advanced once per frame by ClipmapSetupPass::declare; read by all SVO passes.
    pub cur_ring:    AtomicUsize,
}

// ─── Factory ─────────────────────────────────────────────────────────────────

/// Returns `(compute_passes, render_pass)`.
///
/// `compute_passes` (Setup + Bake) go into `extra_pre_gbuffer_passes`.
/// `render_pass` (GBuffer sphere-trace) goes into `extra_gbuffer_passes`.
pub fn create_clipmap_passes(
    clipmap:     Arc<RwLock<ClipmapState>>,
    sdf_scene:   Arc<RwLock<SdfScene>>,
    gpu_buffers: Arc<ClipmapBuffers>,
    debug_flags: Arc<AtomicU32>,
    enabled:     Arc<AtomicBool>,
    vm_ops:      Vec<crate::VmOp>,
    terrain_mat: crate::TerrainMatParams,
) -> (Vec<Box<dyn RenderPass>>, Box<dyn RenderPass>) {
    let shared = Arc::new(ClipmapShared {
        gpu_buffers,
        clipmap,
        sdf_scene,
        job_count:   AtomicU32::new(0),
        prim_count:  AtomicU32::new(0),
        terrain_on:  AtomicU32::new(0),
        vm_ops:      Arc::new(vm_ops),
        terrain_mat,
        bvh_root:    AtomicU32::new(u32::MAX),
        debug_flags,
        enabled,
        cur_ring:    AtomicUsize::new(0),
    });

    let inv_buf = BufferHandle::INVALID;
    let inv_img = ImageHandle::INVALID;

    let compute: Vec<Box<dyn RenderPass>> = vec![
        Box::new(ClipmapSetupPass::new(shared.clone(), inv_buf)),
        Box::new(ClipmapBakePass::new(shared.clone(), inv_buf)),
    ];
    let render: Box<dyn RenderPass> =
        Box::new(ClipmapRenderPass::new(shared.clone(), inv_buf, inv_img));

    (compute, render)
}
