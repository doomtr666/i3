//! # Descriptors - Bindless System
//!
//! This module manages Vulkan descriptor sets, with a focus on the **bindless** approach.
//!
//! ## Bindless Architecture
//!
//! Instead of binding descriptor sets per-draw, the bindless approach uses a single
//! large descriptor set containing all textures, buffers, and samplers. Shaders
//! access resources via indices passed as push constants or vertex attributes.
//!
//! ## Benefits
//!
//! - **Reduced CPU overhead**: No need to bind descriptor sets per draw call
//! - **Flexible resource access**: Shaders can dynamically index into arrays
//! - **Simplified state management**: Fewer pipeline layouts and descriptor sets
//!
//! ## Descriptor Set Layout
//!
//! The bindless set typically contains:
//! - Binding 0: Array of sampled images (textures)
//! - Binding 1: Array of storage images
//! - Binding 2: Array of samplers
//! - Binding 3: Array of uniform buffers
//! - Binding 4: Array of storage buffers
//!
//! ## Thread Safety
//!
//! Descriptor set allocation uses a mutex-protected arena to allow safe
//! concurrent access from multiple threads.

use ash::vk;
use i3_gfx::graph::backend::*;
use i3_gfx::graph::pipeline::*;
use i3_gfx::graph::types::*;
use tracing::error;

use crate::backend::VulkanBackend;

/// Update a bindless texture in the global bindless descriptor set.
///
/// This function updates a single texture entry in the bindless descriptor set.
/// The texture can then be accessed in shaders via its index.
///
/// # Arguments
///
/// * `backend` - Mutable reference to the backend
/// * `texture` - Handle to the image to bind
/// * `sampler` - Handle to the sampler to use (currently unused)
/// * `index` - Index in the descriptor array
/// * `set` - Handle to the descriptor set
/// * `binding` - Binding point in the descriptor set
pub fn update_bindless_texture(
    backend: &mut VulkanBackend,
    texture: ImageHandle,
    sampler: SamplerHandle,
    index: u32,
    set: u64,
    binding: u32,
) {
    let physical = backend.resolve_image(texture);
    update_bindless_texture_raw(backend, physical, sampler, index, set, binding);
}

/// Update a bindless texture using a raw backend image handle.
pub fn update_bindless_texture_raw(
    backend: &mut VulkanBackend,
    texture: BackendImage,
    _sampler: SamplerHandle,
    index: u32,
    set: u64,
    binding: u32,
) {
    let vk_set = if let Some(s) = backend.descriptor_sets.lock().unwrap().get(set as u64) {
        *s
    } else {
        error!("Descriptor set (bindless) not found: {}", set);
        return;
    };

    if let Some(img) = backend.images.get(texture.0) {
        let image_info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: img.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let write = vk::WriteDescriptorSet::default()
            .dst_set(vk_set)
            .dst_binding(binding)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&image_info));

        unsafe {
            backend
                .get_device()
                .handle
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
    } else {
        error!(
            "Physical image not found for bindless update: {:?}",
            texture.0
        );
    }
}

/// Update a bindless sampler in the global bindless descriptor set.
pub fn update_bindless_sampler(
    backend: &mut VulkanBackend,
    sampler: SamplerHandle,
    set: u64,
    binding: u32,
) {
    let vk_set = if let Some(s) = backend.descriptor_sets.lock().unwrap().get(set as u64) {
        *s
    } else {
        error!("Descriptor set (bindless) not found: {}", set);
        return;
    };

    if let Some(&vk_sampler) = backend.samplers.get(sampler.0) {
        let image_info = vk::DescriptorImageInfo {
            sampler: vk_sampler,
            image_view: vk::ImageView::null(),
            image_layout: vk::ImageLayout::UNDEFINED,
        };

        let write = vk::WriteDescriptorSet::default()
            .dst_set(vk_set)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(std::slice::from_ref(&image_info));

        unsafe {
            backend
                .get_device()
                .handle
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
    } else {
        error!("Sampler not found for bindless update: {:?}", sampler.0);
    }
}

/// Allocate a descriptor set from the static pool.
///
/// This function allocates a descriptor set from the static descriptor pool.
/// The pool is pre-allocated at initialization time and reused across frames.
///
/// # Arguments
///
/// * `backend` - Mutable reference to the backend
/// * `pipeline` - Handle to the pipeline (used to get the descriptor set layout)
/// * `set_index` - Index of the descriptor set to allocate
///
/// # Returns
///
/// Handle to the allocated descriptor set, or an error if allocation fails
pub fn allocate_descriptor_set(
    backend: &mut VulkanBackend,
    pipeline: PipelineHandle,
    set_index: u32,
) -> Result<DescriptorSetHandle, String> {
    let pipeline_id = pipeline.0.0;
    let layout = {
        let p = backend
            .pipeline_resources
            .get(pipeline_id)
            .ok_or_else(|| format!("Pipeline layout not found for {:?}", pipeline))?;

        if set_index as usize >= p.set_layouts.len() {
            return Err(format!(
                "Set index {} out of bounds for pipeline {:?}",
                set_index, pipeline
            ));
        }
        p.set_layouts[set_index as usize]
    };

    let layouts_to_alloc = [layout];
    let pool = backend.static_descriptor_pool;
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts_to_alloc);

    let mut arena = backend.descriptor_sets.lock().unwrap();

    let sets = unsafe {
        backend
            .get_device()
            .handle
            .allocate_descriptor_sets(&alloc_info)
            .map_err(|e| format!("Failed to allocate descriptor set: {}", e))?
    };

    let set = sets[0];
    let handle_id = arena.insert(set);

    Ok(DescriptorSetHandle(handle_id))
}

/// Update a descriptor set with the given writes.
///
/// This function updates a descriptor set with new buffer and image bindings.
/// It resolves virtual resource handles to physical Vulkan resources.
///
/// # Arguments
///
/// * `backend` - Mutable reference to the backend
/// * `set` - Handle to the descriptor set to update
/// * `writes` - Array of descriptor writes to apply
///
/// # Resource Resolution
///
/// The function resolves virtual handles (ImageHandle, BufferHandle) to physical
/// Vulkan resources (VkImage, VkBuffer) before updating the descriptor set.
pub fn update_descriptor_set(
    backend: &mut VulkanBackend,
    set: DescriptorSetHandle,
    writes: &[DescriptorWrite],
) {
    let vk_set = if let Some(s) = backend.descriptor_sets.lock().unwrap().get(set.0) {
        *s
    } else {
        error!("Descriptor set not found: {:?}", set);
        return;
    };

    // We need to keep the structures alive until the call to update_descriptor_sets
    // But `vk::WriteDescriptorSet` holds references.
    // We iterate and build vectors.

    let mut descriptor_writes = Vec::new();
    let mut buffer_infos = Vec::new(); // Store infos to keep alive
    let mut image_infos = Vec::new();

    // Pass 1: Create Info structures
    for write in writes {
        match write.descriptor_type {
            BindingType::UniformBuffer | BindingType::StorageBuffer => {
                if let Some(info) = &write.buffer_info {
                    let physical_id = backend.resolve_buffer(info.buffer).0;
                    if let Some(buf) = backend.buffers.get(physical_id) {
                        buffer_infos.push(vk::DescriptorBufferInfo {
                            buffer: buf.buffer,
                            offset: info.offset,
                            range: if info.range == 0 {
                                vk::WHOLE_SIZE
                            } else {
                                info.range
                            },
                        });
                    }
                }
            }
            BindingType::CombinedImageSampler
            | BindingType::Texture
            | BindingType::StorageTexture
            | BindingType::Sampler => {
                if let Some(info) = &write.image_info {
                    // Resolve Image View
                    // We need `image_views` map, but it's keyed by physical ID.
                    // `info.image` is a logical handle.
                    // We first convert logical -> physical
                    let physical_id =
                        if let Some(&phy) = backend.external_to_physical.get(&info.image.0.0) {
                            phy
                        } else {
                            info.image.0.0
                        };

                    if let Some(img) = backend.images.get(physical_id) {
                        let layout = match info.image_layout {
                            DescriptorImageLayout::General => vk::ImageLayout::GENERAL,
                            DescriptorImageLayout::ShaderReadOnlyOptimal => {
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                            }
                        };

                        let vk_sampler = if let Some(sampler_handle) = info.sampler {
                            backend
                                .samplers
                                .get(sampler_handle.0)
                                .cloned()
                                .unwrap_or(vk::Sampler::null())
                        } else {
                            vk::Sampler::null()
                        };

                        image_infos.push(vk::DescriptorImageInfo {
                            sampler: vk_sampler,
                            image_view: img.view,
                            image_layout: layout,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    // Pass 2: Create WriteDescriptorSet
    let mut buf_idx = 0;
    let mut img_idx = 0;

    for write in writes {
        let mut vk_write = vk::WriteDescriptorSet::default()
            .dst_set(vk_set)
            .dst_binding(write.binding)
            .dst_array_element(write.array_element);

        match write.descriptor_type {
            BindingType::UniformBuffer => {
                vk_write = vk_write
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1);
                if buf_idx < buffer_infos.len() {
                    vk_write = vk_write.buffer_info(&buffer_infos[buf_idx..=buf_idx]);
                    buf_idx += 1;
                    descriptor_writes.push(vk_write);
                }
            }
            BindingType::StorageBuffer => {
                vk_write = vk_write
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1);
                if buf_idx < buffer_infos.len() {
                    vk_write = vk_write.buffer_info(&buffer_infos[buf_idx..=buf_idx]);
                    buf_idx += 1;
                    descriptor_writes.push(vk_write);
                }
            }
            BindingType::CombinedImageSampler => {
                vk_write = vk_write
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1);
                if img_idx < image_infos.len() {
                    vk_write = vk_write.image_info(&image_infos[img_idx..=img_idx]);
                    img_idx += 1;
                    descriptor_writes.push(vk_write);
                }
            }
            BindingType::Texture => {
                // Sampled Image
                vk_write = vk_write
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1);
                if img_idx < image_infos.len() {
                    vk_write = vk_write.image_info(&image_infos[img_idx..=img_idx]);
                    img_idx += 1;
                    descriptor_writes.push(vk_write);
                }
            }
            BindingType::StorageTexture => {
                vk_write = vk_write
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1);
                if img_idx < image_infos.len() {
                    vk_write = vk_write.image_info(&image_infos[img_idx..=img_idx]);
                    img_idx += 1;
                    descriptor_writes.push(vk_write);
                }
            }
            BindingType::Sampler => {
                vk_write = vk_write
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .descriptor_count(1);
                if img_idx < image_infos.len() {
                    vk_write = vk_write.image_info(&image_infos[img_idx..=img_idx]);
                    img_idx += 1;
                    descriptor_writes.push(vk_write);
                }
            }
            _ => {}
        }
    }

    unsafe {
        backend
            .get_device()
            .handle
            .update_descriptor_sets(&descriptor_writes, &[]);
    }

    // Pass 3: AccelerationStructure writes — require p_next chain, handled separately.
    for write in writes {
        if write.descriptor_type != BindingType::AccelerationStructure {
            continue;
        }

        if !backend.rt_supported {
            error!("Attempted to update AccelerationStructure descriptor but Ray Tracing is not supported or enabled.");
            continue;
        }

        let Some(as_handle) = write.accel_struct_info else {
            continue;
        };
        let physical = backend.resolve_accel_struct(as_handle);
        let vk_as = match backend.accel_structs.get(physical.0) {
            Some(pas) => pas.handle,
            None => {
                error!(
                    "AccelerationStructure not found for descriptor write: {:?}",
                    as_handle
                );
                continue;
            }
        };
        let as_handles = [vk_as];
        let mut as_ext = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&as_handles);
        let vk_write = vk::WriteDescriptorSet::default()
            .dst_set(vk_set)
            .dst_binding(write.binding)
            .dst_array_element(write.array_element)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .push_next(&mut as_ext);
        unsafe {
            backend
                .get_device()
                .handle
                .update_descriptor_sets(std::slice::from_ref(&vk_write), &[]);
        }
    }
}
