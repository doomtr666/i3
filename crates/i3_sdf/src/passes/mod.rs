mod setup_pass;
mod bake_pass;
mod render_pass;

pub use setup_pass::SvoSetupPass;
pub use bake_pass::SvoBakePass;
pub use render_pass::SvoRenderPass;

use std::sync::{Arc, RwLock, atomic::{AtomicBool, AtomicU32, AtomicUsize}};
use i3_gfx::prelude::*;
use i3_voxel::SdfScene;

use crate::{gpu_buffers::SvoGpuBuffers, svo::SvoTree};

// ─── Shared per-frame state ───────────────────────────────────────────────────

pub(crate) struct SvoShared {
    pub gpu_buffers: Arc<SvoGpuBuffers>,
    pub svo_tree:    Arc<RwLock<SvoTree>>,
    pub sdf_scene:   Arc<RwLock<SdfScene>>,
    pub job_count:   AtomicU32,
    pub prim_count:  AtomicU32,
    pub bvh_root:    AtomicU32,
    pub debug_flags: Arc<AtomicU32>,
    pub enabled:     Arc<AtomicBool>,
    /// Active ring-buffer slot for this frame's CpuToGpu buffers.
    /// Advanced once per frame by SvoSetupPass::declare; read by all SVO passes.
    pub cur_ring:    AtomicUsize,
}

// ─── Factory ─────────────────────────────────────────────────────────────────

/// Returns `(compute_passes, render_pass)`.
///
/// `compute_passes` (Setup + Bake) go into `extra_pre_gbuffer_passes`.
/// `render_pass` (GBuffer sphere-trace) goes into `extra_gbuffer_passes`.
pub fn create_svo_passes(
    svo_tree:    Arc<RwLock<SvoTree>>,
    sdf_scene:   Arc<RwLock<SdfScene>>,
    gpu_buffers: Arc<SvoGpuBuffers>,
    debug_flags: Arc<AtomicU32>,
    enabled:     Arc<AtomicBool>,
) -> (Vec<Box<dyn RenderPass>>, Box<dyn RenderPass>) {
    let shared = Arc::new(SvoShared {
        gpu_buffers,
        svo_tree,
        sdf_scene,
        job_count:   AtomicU32::new(0),
        prim_count:  AtomicU32::new(0),
        bvh_root:    AtomicU32::new(u32::MAX),
        debug_flags,
        enabled,
        cur_ring:    AtomicUsize::new(0),
    });

    let inv_buf = BufferHandle::INVALID;
    let inv_img = ImageHandle::INVALID;

    let compute: Vec<Box<dyn RenderPass>> = vec![
        Box::new(SvoSetupPass::new(shared.clone(), inv_buf)),
        Box::new(SvoBakePass::new(shared.clone(), inv_buf)),
    ];
    let render: Box<dyn RenderPass> =
        Box::new(SvoRenderPass::new(shared.clone(), inv_buf, inv_img));

    (compute, render)
}
