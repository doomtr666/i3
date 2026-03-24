use i3_gfx::prelude::*;
use crate::scene::{GpuMeshDescriptor, GpuInstanceData};

/// Manages persistent GPU buffers for the renderer's scene data.
///
/// These buffers live across frames. The sync passes stream delta
/// updates into them each frame via the `SceneProvider`.
pub struct GpuBuffers {
    pub object_buffer: BackendBuffer,
    pub material_buffer: BackendBuffer,
    pub light_buffer: BackendBuffer,
    pub camera_ubo: BackendBuffer,

    // GPU-Driven Rendering Buffers
    pub mesh_descriptor_buffer: BackendBuffer,
    pub instance_buffer:        BackendBuffer,
    pub draw_call_buffer:       BackendBuffer,
    pub draw_count_buffer:      BackendBuffer,
    pub visible_instance_buffer: BackendBuffer,
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
        let object_buffer_size = 1024 * 1024 * 2; // 2MB for ~14k objects
        let material_buffer_size = 1024 * 1024; // 1MB for ~16k materials
        let camera_ubo_size = std::mem::size_of::<CameraData>() as u64;

        // Light Buffer (1024 lights) -> Struct size is 48 bytes approx
        let max_lights = 1024;
        let light_data_size = std::mem::size_of::<crate::scene::LightData>() as u64;
        let light_buffer_size = max_lights * light_data_size;

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

        // GPU-Driven Buffers
        let max_meshes = 4096;
        let max_instances = 65536;

        let mesh_descriptor_buffer_size = max_meshes * std::mem::size_of::<GpuMeshDescriptor>() as u64;
        let instance_buffer_size = max_instances * std::mem::size_of::<GpuInstanceData>() as u64;
        let draw_call_buffer_size = max_instances * 16; // DrawIndirectCommand is 16 bytes
        let draw_count_buffer_size = 16; // 4 bytes + padding
        let visible_instance_buffer_size = max_instances * 4;

        let buffers = Self {
            object_buffer: create_storage(backend, object_buffer_size),
            material_buffer: create_storage(backend, material_buffer_size),
            light_buffer: create_storage(backend, light_buffer_size),
            camera_ubo: create_ubo(backend, camera_ubo_size),

            mesh_descriptor_buffer: create_storage(backend, mesh_descriptor_buffer_size),
            instance_buffer:        create_storage(backend, instance_buffer_size),
            draw_call_buffer: backend.create_buffer(&BufferDesc {
                size: draw_call_buffer_size,
                usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::INDIRECT_BUFFER | BufferUsageFlags::TRANSFER_DST,
                memory: MemoryType::GpuOnly,
            }),
            draw_count_buffer: backend.create_buffer(&BufferDesc {
                size: draw_count_buffer_size,
                usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::INDIRECT_BUFFER | BufferUsageFlags::TRANSFER_DST,
                memory: MemoryType::GpuOnly,
            }),
            visible_instance_buffer: backend.create_buffer(&BufferDesc {
                size: visible_instance_buffer_size,
                usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
                memory: MemoryType::GpuOnly,
            }),
        };

        #[cfg(debug_assertions)]
        {
            backend.set_buffer_name(buffers.object_buffer, "ObjectBuffer");
            backend.set_buffer_name(buffers.material_buffer, "MaterialBuffer");
            backend.set_buffer_name(buffers.light_buffer, "LightBuffer");
            backend.set_buffer_name(buffers.camera_ubo, "CameraUBO");
            backend.set_buffer_name(buffers.mesh_descriptor_buffer, "MeshDescriptorBuffer");
            backend.set_buffer_name(buffers.instance_buffer, "InstanceBuffer");
            backend.set_buffer_name(buffers.draw_call_buffer, "DrawCallBuffer");
            backend.set_buffer_name(buffers.draw_count_buffer, "DrawCountBuffer");
            backend.set_buffer_name(buffers.visible_instance_buffer, "VisibleInstanceBuffer");
        }

        buffers
    }
}
