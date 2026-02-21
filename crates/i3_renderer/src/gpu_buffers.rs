use i3_gfx::prelude::*;

/// Manages persistent GPU buffers for the renderer's scene data.
///
/// These buffers live across frames. The sync passes stream delta
/// updates into them each frame via the `SceneProvider`.
pub struct GpuBuffers {
    pub object_buffer: BufferHandle,
    pub material_buffer: BufferHandle,
    pub light_buffer: BufferHandle,
    pub camera_ubo: BufferHandle,
}

/// Per-frame camera data uploaded to the CameraUBO.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CameraData {
    pub view: nalgebra_glm::Mat4,
    pub projection: nalgebra_glm::Mat4,
    pub view_projection: nalgebra_glm::Mat4,
    pub inv_view: nalgebra_glm::Mat4,
    pub inv_projection: nalgebra_glm::Mat4,
    pub camera_position: nalgebra_glm::Vec4,
}

impl GpuBuffers {
    /// Creates the persistent GPU buffers.
    ///
    /// Buffer allocation is deferred — handles are invalid until
    /// the first frame's sync pass allocates them via the backend.
    pub fn new() -> Self {
        Self {
            object_buffer: BufferHandle::INVALID,
            material_buffer: BufferHandle::INVALID,
            light_buffer: BufferHandle::INVALID,
            camera_ubo: BufferHandle::INVALID,
        }
    }
}
