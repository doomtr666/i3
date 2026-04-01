use crate::backend::VulkanBackend;
use crate::commands::VulkanPassContext;
use ash::vk;
use i3_gfx::graph::backend::*;
use vk_mem::Alloc;

pub struct PhysicalAccelerationStructure {
    pub handle: vk::AccelerationStructureKHR,
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub address: u64,
    pub build_info: Option<BlasCreateInfo>, // Store for BLAS rebuilds
}

pub fn create_blas(
    backend: &mut VulkanBackend,
    info: &BlasCreateInfo,
) -> BackendAccelerationStructure {
    let device = backend.get_device().clone();

    // 1. Convert geometries to Vulkan
    let mut geometries = Vec::with_capacity(info.geometries.len());
    let mut build_ranges = Vec::with_capacity(info.geometries.len());
    let mut max_primitive_counts = Vec::with_capacity(info.geometries.len());

    for geo in &info.geometries {
        let vertex_address = backend.get_buffer_address(geo.vertex_buffer) + geo.vertex_offset;
        let index_address = backend.get_buffer_address(geo.index_buffer) + geo.index_offset;

        let tri_data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(crate::convert::convert_format(geo.vertex_format))
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vertex_address,
            })
            .vertex_stride(geo.vertex_stride as u64)
            .max_vertex(geo.vertex_count)
            .index_type(crate::convert::convert_index_type(geo.index_type))
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: index_address,
            });

        geometries.push(
            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: tri_data,
                })
                .flags(vk::GeometryFlagsKHR::OPAQUE),
        );

        max_primitive_counts.push(geo.index_count / 3);
        build_ranges.push(
            vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(geo.index_count / 3)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0),
        );
    }

    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(convert_build_flags(info.flags))
        .geometries(&geometries);

    // 2. Query sizes
    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        device
            .accel_struct
            .as_ref()
            .unwrap()
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &max_primitive_counts,
                &mut size_info,
            )
    };

    // 3. Create backing buffer
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size_info.acceleration_structure_size)
        .usage(
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let alloc_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::AutoPreferDevice,
        ..Default::default()
    };

    let (buffer, allocation) = unsafe {
        device
            .allocator
            .lock()
            .unwrap()
            .create_buffer(&buffer_info, &alloc_info)
            .unwrap()
    };

    // 4. Create Acceleration Structure
    let create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(buffer)
        .offset(0)
        .size(size_info.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

    let handle = unsafe {
        device
            .accel_struct
            .as_ref()
            .unwrap()
            .create_acceleration_structure(&create_info, None)
            .unwrap()
    };

    let address_info =
        vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle);
    let address = unsafe {
        device
            .accel_struct
            .as_ref()
            .unwrap()
            .get_acceleration_structure_device_address(&address_info)
    };

    let physical = PhysicalAccelerationStructure {
        handle,
        buffer,
        allocation,
        address,
        build_info: Some(BlasCreateInfo {
            geometries: info
                .geometries
                .iter()
                .map(|g| BlasGeometryDesc {
                    vertex_buffer: g.vertex_buffer,
                    vertex_offset: g.vertex_offset,
                    vertex_count: g.vertex_count,
                    vertex_stride: g.vertex_stride,
                    vertex_format: g.vertex_format,
                    index_buffer: g.index_buffer,
                    index_offset: g.index_offset,
                    index_count: g.index_count,
                    index_type: g.index_type,
                })
                .collect(),
            flags: info.flags,
        }),
    };

    let id = backend.accel_structs.insert(physical);
    BackendAccelerationStructure(id)
}

pub fn destroy_blas(backend: &mut VulkanBackend, handle: BackendAccelerationStructure) {
    if let Some(pas) = backend.accel_structs.remove(handle.0) {
        backend.dead_accel_structs.push((
            backend.frame_count,
            pas.handle,
            pas.buffer,
            pas.allocation,
        ));
    }
}

pub fn create_tlas(
    backend: &mut VulkanBackend,
    info: &TlasCreateInfo,
) -> BackendAccelerationStructure {
    let device = backend.get_device().clone();

    // Top-level AS requires a geometry entry of type INSTANCES for size querying
    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false),
        });
    let geometries = [geometry];

    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(convert_build_flags(info.flags))
        .geometries(&geometries);

    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        device
            .accel_struct
            .as_ref()
            .unwrap()
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[info.max_instances],
                &mut size_info,
            )
    };

    // Create backing buffer
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size_info.acceleration_structure_size)
        .usage(
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let alloc_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::AutoPreferDevice,
        ..Default::default()
    };

    let (buffer, allocation) = unsafe {
        device
            .allocator
            .lock()
            .unwrap()
            .create_buffer(&buffer_info, &alloc_info)
            .unwrap()
    };

    // Create Acceleration Structure
    let create_info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(buffer)
        .offset(0)
        .size(size_info.acceleration_structure_size)
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

    let handle = unsafe {
        device
            .accel_struct
            .as_ref()
            .unwrap()
            .create_acceleration_structure(&create_info, None)
            .unwrap()
    };

    let address_info =
        vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle);
    let address = unsafe {
        device
            .accel_struct
            .as_ref()
            .unwrap()
            .get_acceleration_structure_device_address(&address_info)
    };

    let physical = PhysicalAccelerationStructure {
        handle,
        buffer,
        allocation,
        address,
        build_info: None,
    };

    let id = backend.accel_structs.insert(physical);
    BackendAccelerationStructure(id)
}

pub fn destroy_tlas(backend: &mut VulkanBackend, handle: BackendAccelerationStructure) {
    destroy_blas(backend, handle)
}

pub fn build_blas(ctx: &mut VulkanPassContext, handle: BackendAccelerationStructure, update: bool) {
    tracing::debug!("Building BLAS {:?}", handle);
    let device = ctx.backend().get_device().clone();
    let (blas_handle, build_info_copy) = {
        let pas = ctx
            .backend()
            .accel_structs
            .get(handle.0)
            .expect("BLAS not found");
        (
            pas.handle,
            pas.build_info.clone().expect("BLAS build info missing"),
        )
    };
    let info = &build_info_copy;

    // 1. Convert geometries to Vulkan
    let mut geometries = Vec::with_capacity(info.geometries.len());
    let mut build_ranges = Vec::with_capacity(info.geometries.len());

    for geo in &info.geometries {
        let vertex_address =
            ctx.backend().get_buffer_address(geo.vertex_buffer) + geo.vertex_offset;
        let index_address = ctx.backend().get_buffer_address(geo.index_buffer) + geo.index_offset;

        let tri_data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(crate::convert::convert_format(geo.vertex_format))
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vertex_address,
            })
            .vertex_stride(geo.vertex_stride as u64)
            .max_vertex(geo.vertex_count)
            .index_type(crate::convert::convert_index_type(geo.index_type))
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: index_address,
            });

        geometries.push(
            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: tri_data,
                })
                .flags(vk::GeometryFlagsKHR::OPAQUE),
        );

        build_ranges.push(
            vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(geo.index_count / 3)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0),
        );
    }

    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(convert_build_flags(info.flags))
        .mode(if update {
            vk::BuildAccelerationStructureModeKHR::UPDATE
        } else {
            vk::BuildAccelerationStructureModeKHR::BUILD
        })
        .dst_acceleration_structure(blas_handle)
        .geometries(&geometries);

    // Get scratch size
    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    let max_primitive_counts: Vec<u32> =
        info.geometries.iter().map(|g| g.index_count / 3).collect();
    unsafe {
        device
            .accel_struct
            .as_ref()
            .unwrap()
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &max_primitive_counts,
                &mut size_info,
            )
    };

    // Create scratch buffer
    let scratch_desc = BufferDesc {
        size: if update {
            size_info.update_scratch_size
        } else {
            size_info.build_scratch_size
        },
        usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        memory: MemoryType::GpuOnly,
    };
    let scratch_handle = ctx.backend_mut().create_buffer(&scratch_desc);
    let scratch_address = ctx.backend_mut().get_buffer_address(scratch_handle);
    build_info.scratch_data = vk::DeviceOrHostAddressKHR {
        device_address: scratch_address,
    };

    let build_range_infos = [build_ranges.as_slice()];

    // Barrier: serialize against any prior AS writes on this queue (WAW, BLAS→TLAS RAW).
    // vkCmdPipelineBarrier2 within a CB also covers prior submissions on the same queue.
    let as_barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR);
    let dep_info = vk::DependencyInfo::default()
        .memory_barriers(std::slice::from_ref(&as_barrier));
    unsafe {
        device.handle.cmd_pipeline_barrier2(ctx.cmd, &dep_info);
    }

    unsafe {
        device
            .accel_struct
            .as_ref()
            .unwrap()
            .cmd_build_acceleration_structures(ctx.cmd, &[build_info], &build_range_infos);
    }

    // Destroy scratch buffer (will be deferred for 2 frames by backend)
    ctx.backend_mut().destroy_buffer(scratch_handle);
}

pub fn build_tlas(
    ctx: &mut VulkanPassContext,
    handle: BackendAccelerationStructure,
    instances: &[TlasInstanceDesc],
    update: bool,
) {
    if !instances.is_empty() {
        tracing::debug!("Building TLAS {:?} with {} instances", handle, instances.len());
    }
    let device = ctx.backend().get_device().clone();
    let tlas_handle = {
        let pas = ctx
            .backend()
            .accel_structs
            .get(handle.0)
            .expect("TLAS not found");
        pas.handle
    };

    // 1. Convert instances to Vulkan layout
    let vk_instances: Vec<vk::AccelerationStructureInstanceKHR> = instances
        .iter()
        .map(|inst| {
            let blas_pas = ctx
                .backend()
                .accel_structs
                .get(inst.blas.0)
                .expect("BLAS not found");
            let vk_inst = vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: inst.transform,
                },
                instance_custom_index_and_mask: vk::Packed24_8::new(inst.instance_id, inst.mask),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    inst.sbt_offset,
                    inst.flags,
                ),
                // Safety: AccelerationStructureReferenceKHR is a union.
                // We use transmute to move the device address into the union.
                acceleration_structure_reference: unsafe { std::mem::transmute(blas_pas.address) },
            };
            vk_inst
        })
        .collect();

    // 2. Upload instances to a temporary buffer
    let inst_data = unsafe {
        std::slice::from_raw_parts(
            vk_instances.as_ptr() as *const u8,
            vk_instances.len() * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>(),
        )
    };

    let inst_buffer_desc = BufferDesc {
        size: inst_data.len() as u64,
        usage: BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT
            | BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        memory: MemoryType::CpuToGpu,
    };
    let inst_buffer_handle = {
        let backend = ctx.backend_mut();
        let handle = backend.create_buffer(&inst_buffer_desc);
        backend.upload_buffer(handle, inst_data, 0).unwrap();
        handle
    };
    let inst_buffer_address = ctx.backend_mut().get_buffer_address(inst_buffer_handle);

    // 3. Build command
    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: inst_buffer_address,
                }),
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    let geometries = [geometry];
    let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(if update {
            vk::BuildAccelerationStructureModeKHR::UPDATE
        } else {
            vk::BuildAccelerationStructureModeKHR::BUILD
        })
        .dst_acceleration_structure(tlas_handle)
        .geometries(&geometries);

    // Get scratch size
    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        device
            .accel_struct
            .as_ref()
            .unwrap()
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[instances.len() as u32],
                &mut size_info,
            )
    };

    // Create scratch buffer
    let scratch_desc = BufferDesc {
        size: if update {
            size_info.update_scratch_size
        } else {
            size_info.build_scratch_size
        },
        usage: BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        memory: MemoryType::GpuOnly,
    };
    let scratch_handle = ctx.backend_mut().create_buffer(&scratch_desc);
    let scratch_address = ctx.backend_mut().get_buffer_address(scratch_handle);
    build_info.scratch_data = vk::DeviceOrHostAddressKHR {
        device_address: scratch_address,
    };

    let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
        .primitive_count(instances.len() as u32)
        .primitive_offset(0)
        .first_vertex(0)
        .transform_offset(0);

    let build_ranges = [build_range];
    let build_range_infos = [build_ranges.as_slice()];

    // Barrier: serialize against prior AS writes (WAW inter-frame, BLAS→TLAS RAW).
    let as_barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR);
    let dep_info = vk::DependencyInfo::default()
        .memory_barriers(std::slice::from_ref(&as_barrier));
    unsafe {
        device.handle.cmd_pipeline_barrier2(ctx.cmd, &dep_info);
    }

    unsafe {
        device
            .accel_struct
            .as_ref()
            .unwrap()
            .cmd_build_acceleration_structures(ctx.cmd, &[build_info], &build_range_infos);
    }

    // Destroy transient buffers
    ctx.backend_mut().destroy_buffer(inst_buffer_handle);
    ctx.backend_mut().destroy_buffer(scratch_handle);
}

fn convert_build_flags(flags: AccelStructBuildFlags) -> vk::BuildAccelerationStructureFlagsKHR {
    let mut vk_flags = vk::BuildAccelerationStructureFlagsKHR::empty();
    if flags.contains(AccelStructBuildFlags::PREFER_FAST_TRACE) {
        vk_flags |= vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;
    }
    if flags.contains(AccelStructBuildFlags::PREFER_FAST_BUILD) {
        vk_flags |= vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD;
    }
    if flags.contains(AccelStructBuildFlags::ALLOW_UPDATE) {
        vk_flags |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE;
    }
    if flags.contains(AccelStructBuildFlags::ALLOW_COMPACTION) {
        vk_flags |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION;
    }
    vk_flags
}
