use ash::vk;
use i3_gfx::graph::backend::*;
use i3_gfx::graph::pipeline::*;
use i3_gfx::graph::types::*;
use tracing::error;

use crate::backend::VulkanBackend;

/// Update a bindless texture in the global bindless descriptor set.
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

/// Allocate a descriptor set from the static pool.
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
}
