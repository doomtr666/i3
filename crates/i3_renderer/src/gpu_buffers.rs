use i3_gfx::prelude::*;

/// Manages persistent GPU buffers for the renderer's scene data.
///
/// These buffers live across frames. The sync passes stream delta
/// updates into them each frame via the `SceneProvider`.
pub struct GpuBuffers {
    pub object_buffer: BackendBuffer,
    pub material_buffer: BackendBuffer,
    pub light_buffer: BackendBuffer,
    pub camera_ubo: BackendBuffer,
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
    pub fn allocate(backend: &mut dyn RenderBackend) -> Self {
        // Compute standard sizes
        let object_buffer_size = 1024 * 64; // Stub
        let material_buffer_size = 1024 * 64; // Stub
        let camera_ubo_size = std::mem::size_of::<CameraData>() as u64;

        // Light Buffer (Header + 1024 lights) -> Struct size is 48 bytes approx
        let max_lights = 1024;
        let light_data_size = std::mem::size_of::<crate::scene::LightData>() as u64;
        // SSBO Layout: struct { uint count; vec3 pad; LightData lights[]; }
        let light_buffer_size = 16 + (max_lights * light_data_size);

        let create_storage = |backend: &mut dyn RenderBackend, size: u64| {
            backend.create_buffer(&BufferDesc {
                size,
                usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
                memory: MemoryType::CpuToGpu,
            })
        };

        let create_ubo = |backend: &mut dyn RenderBackend, size: u64| {
            backend.create_buffer(&BufferDesc {
                size,
                usage: BufferUsageFlags::UNIFORM_BUFFER | BufferUsageFlags::TRANSFER_DST,
                memory: MemoryType::CpuToGpu,
            })
        };

        Self {
            object_buffer: create_storage(backend, object_buffer_size),
            material_buffer: create_storage(backend, material_buffer_size),
            light_buffer: create_storage(backend, light_buffer_size),
            camera_ubo: create_ubo(backend, camera_ubo_size),
        }
    }
}
