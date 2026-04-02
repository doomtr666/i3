use ash::vk;
use i3_gfx::graph::sync as abstract_sync;
use crate::sync::*;

pub fn translate_layout_to_abstract(layout: vk::ImageLayout) -> abstract_sync::ImageLayout {
    match layout {
        vk::ImageLayout::UNDEFINED => abstract_sync::ImageLayout::Undefined,
        vk::ImageLayout::GENERAL => abstract_sync::ImageLayout::General,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => abstract_sync::ImageLayout::ColorAttachmentOptimal,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => abstract_sync::ImageLayout::DepthStencilAttachmentOptimal,
        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL => abstract_sync::ImageLayout::DepthStencilReadOnlyOptimal,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => abstract_sync::ImageLayout::ShaderReadOnlyOptimal,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => abstract_sync::ImageLayout::TransferSrcOptimal,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => abstract_sync::ImageLayout::TransferDstOptimal,
        vk::ImageLayout::PREINITIALIZED => abstract_sync::ImageLayout::Preinitialized,
        vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL => abstract_sync::ImageLayout::DepthReadOnlyStencilAttachmentOptimal,
        vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL => abstract_sync::ImageLayout::DepthAttachmentStencilReadOnlyOptimal,
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => abstract_sync::ImageLayout::DepthAttachmentOptimal,
        vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL => abstract_sync::ImageLayout::DepthReadOnlyOptimal,
        vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL => abstract_sync::ImageLayout::StencilAttachmentOptimal,
        vk::ImageLayout::STENCIL_READ_ONLY_OPTIMAL => abstract_sync::ImageLayout::StencilReadOnlyOptimal,
        vk::ImageLayout::PRESENT_SRC_KHR => abstract_sync::ImageLayout::PresentSrc,
        _ => abstract_sync::ImageLayout::Undefined,
    }
}

pub fn translate_layout_from_abstract(layout: abstract_sync::ImageLayout) -> vk::ImageLayout {
    match layout {
        abstract_sync::ImageLayout::Undefined => vk::ImageLayout::UNDEFINED,
        abstract_sync::ImageLayout::General => vk::ImageLayout::GENERAL,
        abstract_sync::ImageLayout::ColorAttachmentOptimal => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        abstract_sync::ImageLayout::DepthStencilAttachmentOptimal => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        abstract_sync::ImageLayout::DepthStencilReadOnlyOptimal => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        abstract_sync::ImageLayout::ShaderReadOnlyOptimal => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        abstract_sync::ImageLayout::TransferSrcOptimal => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        abstract_sync::ImageLayout::TransferDstOptimal => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        abstract_sync::ImageLayout::Preinitialized => vk::ImageLayout::PREINITIALIZED,
        abstract_sync::ImageLayout::DepthReadOnlyStencilAttachmentOptimal => vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
        abstract_sync::ImageLayout::DepthAttachmentStencilReadOnlyOptimal => vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
        abstract_sync::ImageLayout::DepthAttachmentOptimal => vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        abstract_sync::ImageLayout::DepthReadOnlyOptimal => vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
        abstract_sync::ImageLayout::StencilAttachmentOptimal => vk::ImageLayout::STENCIL_ATTACHMENT_OPTIMAL,
        abstract_sync::ImageLayout::StencilReadOnlyOptimal => vk::ImageLayout::STENCIL_READ_ONLY_OPTIMAL,
        abstract_sync::ImageLayout::PresentSrc => vk::ImageLayout::PRESENT_SRC_KHR,
    }
}

pub fn translate_access_to_abstract(access: vk::AccessFlags2) -> abstract_sync::AccessFlags {
    let mut result = abstract_sync::AccessFlags::NONE;
    if access.contains(vk::AccessFlags2::INDIRECT_COMMAND_READ) { result |= abstract_sync::AccessFlags::INDIRECT_COMMAND_READ; }
    if access.contains(vk::AccessFlags2::INDEX_READ) { result |= abstract_sync::AccessFlags::INDEX_READ; }
    if access.contains(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ) { result |= abstract_sync::AccessFlags::VERTEX_ATTRIBUTE_READ; }
    if access.contains(vk::AccessFlags2::UNIFORM_READ) { result |= abstract_sync::AccessFlags::UNIFORM_READ; }
    if access.contains(vk::AccessFlags2::INPUT_ATTACHMENT_READ) { result |= abstract_sync::AccessFlags::INPUT_ATTACHMENT_READ; }
    if access.contains(vk::AccessFlags2::SHADER_READ) { result |= abstract_sync::AccessFlags::SHADER_READ; }
    if access.contains(vk::AccessFlags2::SHADER_WRITE) { result |= abstract_sync::AccessFlags::SHADER_WRITE; }
    if access.contains(vk::AccessFlags2::COLOR_ATTACHMENT_READ) { result |= abstract_sync::AccessFlags::COLOR_ATTACHMENT_READ; }
    if access.contains(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE) { result |= abstract_sync::AccessFlags::COLOR_ATTACHMENT_WRITE; }
    if access.contains(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ) { result |= abstract_sync::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ; }
    if access.contains(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE) { result |= abstract_sync::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE; }
    if access.contains(vk::AccessFlags2::TRANSFER_READ) { result |= abstract_sync::AccessFlags::TRANSFER_READ; }
    if access.contains(vk::AccessFlags2::TRANSFER_WRITE) { result |= abstract_sync::AccessFlags::TRANSFER_WRITE; }
    if access.contains(vk::AccessFlags2::HOST_READ) { result |= abstract_sync::AccessFlags::HOST_READ; }
    if access.contains(vk::AccessFlags2::HOST_WRITE) { result |= abstract_sync::AccessFlags::HOST_WRITE; }
    if access.contains(vk::AccessFlags2::MEMORY_READ) { result |= abstract_sync::AccessFlags::MEMORY_READ; }
    if access.contains(vk::AccessFlags2::MEMORY_WRITE) { result |= abstract_sync::AccessFlags::MEMORY_WRITE; }
    if access.contains(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR) { result |= abstract_sync::AccessFlags::ACCELERATION_STRUCTURE_READ; }
    if access.contains(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR) { result |= abstract_sync::AccessFlags::ACCELERATION_STRUCTURE_WRITE; }
    result
}

pub fn translate_access_from_abstract(access: abstract_sync::AccessFlags) -> vk::AccessFlags2 {
    let mut result = vk::AccessFlags2::empty();
    if access.contains(abstract_sync::AccessFlags::INDIRECT_COMMAND_READ) { result |= vk::AccessFlags2::INDIRECT_COMMAND_READ; }
    if access.contains(abstract_sync::AccessFlags::INDEX_READ) { result |= vk::AccessFlags2::INDEX_READ; }
    if access.contains(abstract_sync::AccessFlags::VERTEX_ATTRIBUTE_READ) { result |= vk::AccessFlags2::VERTEX_ATTRIBUTE_READ; }
    if access.contains(abstract_sync::AccessFlags::UNIFORM_READ) { result |= vk::AccessFlags2::UNIFORM_READ; }
    if access.contains(abstract_sync::AccessFlags::INPUT_ATTACHMENT_READ) { result |= vk::AccessFlags2::INPUT_ATTACHMENT_READ; }
    if access.contains(abstract_sync::AccessFlags::SHADER_READ) { result |= vk::AccessFlags2::SHADER_READ; }
    if access.contains(abstract_sync::AccessFlags::SHADER_WRITE) { result |= vk::AccessFlags2::SHADER_WRITE; }
    if access.contains(abstract_sync::AccessFlags::COLOR_ATTACHMENT_READ) { result |= vk::AccessFlags2::COLOR_ATTACHMENT_READ; }
    if access.contains(abstract_sync::AccessFlags::COLOR_ATTACHMENT_WRITE) { result |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE; }
    if access.contains(abstract_sync::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ) { result |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ; }
    if access.contains(abstract_sync::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE) { result |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE; }
    if access.contains(abstract_sync::AccessFlags::TRANSFER_READ) { result |= vk::AccessFlags2::TRANSFER_READ; }
    if access.contains(abstract_sync::AccessFlags::TRANSFER_WRITE) { result |= vk::AccessFlags2::TRANSFER_WRITE; }
    if access.contains(abstract_sync::AccessFlags::HOST_READ) { result |= vk::AccessFlags2::HOST_READ; }
    if access.contains(abstract_sync::AccessFlags::HOST_WRITE) { result |= vk::AccessFlags2::HOST_WRITE; }
    if access.contains(abstract_sync::AccessFlags::MEMORY_READ) { result |= vk::AccessFlags2::MEMORY_READ; }
    if access.contains(abstract_sync::AccessFlags::MEMORY_WRITE) { result |= vk::AccessFlags2::MEMORY_WRITE; }
    if access.contains(abstract_sync::AccessFlags::ACCELERATION_STRUCTURE_READ) { result |= vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR; }
    if access.contains(abstract_sync::AccessFlags::ACCELERATION_STRUCTURE_WRITE) { result |= vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR; }
    result
}

pub fn translate_stages_to_abstract(stages: vk::PipelineStageFlags2) -> abstract_sync::StageFlags {
    let mut result = abstract_sync::StageFlags::empty();
    if stages.contains(vk::PipelineStageFlags2::TOP_OF_PIPE) { result |= abstract_sync::StageFlags::TOP_OF_PIPE; }
    if stages.contains(vk::PipelineStageFlags2::DRAW_INDIRECT) { result |= abstract_sync::StageFlags::DRAW_INDIRECT; }
    if stages.contains(vk::PipelineStageFlags2::VERTEX_INPUT) { result |= abstract_sync::StageFlags::VERTEX_INPUT; }
    if stages.contains(vk::PipelineStageFlags2::VERTEX_SHADER) { result |= abstract_sync::StageFlags::VERTEX_SHADER; }
    if stages.contains(vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER) { result |= abstract_sync::StageFlags::TESSELLATION_CONTROL_SHADER; }
    if stages.contains(vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER) { result |= abstract_sync::StageFlags::TESSELLATION_EVALUATION_SHADER; }
    if stages.contains(vk::PipelineStageFlags2::GEOMETRY_SHADER) { result |= abstract_sync::StageFlags::GEOMETRY_SHADER; }
    if stages.contains(vk::PipelineStageFlags2::FRAGMENT_SHADER) { result |= abstract_sync::StageFlags::FRAGMENT_SHADER; }
    if stages.contains(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS) { result |= abstract_sync::StageFlags::EARLY_FRAGMENT_TESTS; }
    if stages.contains(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS) { result |= abstract_sync::StageFlags::LATE_FRAGMENT_TESTS; }
    if stages.contains(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT) { result |= abstract_sync::StageFlags::COLOR_ATTACHMENT_OUTPUT; }
    if stages.contains(vk::PipelineStageFlags2::COMPUTE_SHADER) { result |= abstract_sync::StageFlags::COMPUTE_SHADER; }
    if stages.contains(vk::PipelineStageFlags2::TRANSFER) { result |= abstract_sync::StageFlags::TRANSFER; }
    if stages.contains(vk::PipelineStageFlags2::BOTTOM_OF_PIPE) { result |= abstract_sync::StageFlags::BOTTOM_OF_PIPE; }
    if stages.contains(vk::PipelineStageFlags2::HOST) { result |= abstract_sync::StageFlags::HOST; }
    if stages.contains(vk::PipelineStageFlags2::ALL_GRAPHICS) { result |= abstract_sync::StageFlags::ALL_GRAPHICS; }
    if stages.contains(vk::PipelineStageFlags2::ALL_COMMANDS) { result |= abstract_sync::StageFlags::ALL_COMMANDS; }
    if stages.contains(vk::PipelineStageFlags2::COPY) { result |= abstract_sync::StageFlags::COPY; }
    if stages.contains(vk::PipelineStageFlags2::RESOLVE) { result |= abstract_sync::StageFlags::RESOLVE; }
    if stages.contains(vk::PipelineStageFlags2::BLIT) { result |= abstract_sync::StageFlags::BLIT; }
    if stages.contains(vk::PipelineStageFlags2::CLEAR) { result |= abstract_sync::StageFlags::CLEAR; }
    if stages.contains(vk::PipelineStageFlags2::INDEX_INPUT) { result |= abstract_sync::StageFlags::INDEX_INPUT; }
    if stages.contains(vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT) { result |= abstract_sync::StageFlags::VERTEX_ATTRIBUTE_INPUT; }
    if stages.contains(vk::PipelineStageFlags2::PRE_RASTERIZATION_SHADERS) { result |= abstract_sync::StageFlags::PRE_RASTERIZATION_SHADERS; }
    if stages.contains(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR) { result |= abstract_sync::StageFlags::RAY_TRACING_SHADER; }
    if stages.contains(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR) { result |= abstract_sync::StageFlags::ACCELERATION_STRUCTURE_BUILD; }
    result
}

pub fn translate_stages_from_abstract(stages: abstract_sync::StageFlags) -> vk::PipelineStageFlags2 {
    let mut result = vk::PipelineStageFlags2::empty();
    if stages.contains(abstract_sync::StageFlags::TOP_OF_PIPE) { result |= vk::PipelineStageFlags2::TOP_OF_PIPE; }
    if stages.contains(abstract_sync::StageFlags::DRAW_INDIRECT) { result |= vk::PipelineStageFlags2::DRAW_INDIRECT; }
    if stages.contains(abstract_sync::StageFlags::VERTEX_INPUT) { result |= vk::PipelineStageFlags2::VERTEX_INPUT; }
    if stages.contains(abstract_sync::StageFlags::VERTEX_SHADER) { result |= vk::PipelineStageFlags2::VERTEX_SHADER; }
    if stages.contains(abstract_sync::StageFlags::TESSELLATION_CONTROL_SHADER) { result |= vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER; }
    if stages.contains(abstract_sync::StageFlags::TESSELLATION_EVALUATION_SHADER) { result |= vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER; }
    if stages.contains(abstract_sync::StageFlags::GEOMETRY_SHADER) { result |= vk::PipelineStageFlags2::GEOMETRY_SHADER; }
    if stages.contains(abstract_sync::StageFlags::FRAGMENT_SHADER) { result |= vk::PipelineStageFlags2::FRAGMENT_SHADER; }
    if stages.contains(abstract_sync::StageFlags::EARLY_FRAGMENT_TESTS) { result |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS; }
    if stages.contains(abstract_sync::StageFlags::LATE_FRAGMENT_TESTS) { result |= vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS; }
    if stages.contains(abstract_sync::StageFlags::COLOR_ATTACHMENT_OUTPUT) { result |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT; }
    if stages.contains(abstract_sync::StageFlags::COMPUTE_SHADER) { result |= vk::PipelineStageFlags2::COMPUTE_SHADER; }
    if stages.contains(abstract_sync::StageFlags::TRANSFER) { result |= vk::PipelineStageFlags2::TRANSFER; }
    if stages.contains(abstract_sync::StageFlags::BOTTOM_OF_PIPE) { result |= vk::PipelineStageFlags2::BOTTOM_OF_PIPE; }
    if stages.contains(abstract_sync::StageFlags::HOST) { result |= vk::PipelineStageFlags2::HOST; }
    if stages.contains(abstract_sync::StageFlags::ALL_GRAPHICS) { result |= vk::PipelineStageFlags2::ALL_GRAPHICS; }
    if stages.contains(abstract_sync::StageFlags::ALL_COMMANDS) { result |= vk::PipelineStageFlags2::ALL_COMMANDS; }
    if stages.contains(abstract_sync::StageFlags::COPY) { result |= vk::PipelineStageFlags2::COPY; }
    if stages.contains(abstract_sync::StageFlags::RESOLVE) { result |= vk::PipelineStageFlags2::RESOLVE; }
    if stages.contains(abstract_sync::StageFlags::BLIT) { result |= vk::PipelineStageFlags2::BLIT; }
    if stages.contains(abstract_sync::StageFlags::CLEAR) { result |= vk::PipelineStageFlags2::CLEAR; }
    if stages.contains(abstract_sync::StageFlags::INDEX_INPUT) { result |= vk::PipelineStageFlags2::INDEX_INPUT; }
    if stages.contains(abstract_sync::StageFlags::VERTEX_ATTRIBUTE_INPUT) { result |= vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT; }
    if stages.contains(abstract_sync::StageFlags::PRE_RASTERIZATION_SHADERS) { result |= vk::PipelineStageFlags2::PRE_RASTERIZATION_SHADERS; }
    if stages.contains(abstract_sync::StageFlags::RAY_TRACING_SHADER) { result |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR; }
    if stages.contains(abstract_sync::StageFlags::ACCELERATION_STRUCTURE_BUILD) { result |= vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR; }
    result
}

// Translate abstract plan to Vulkan barriers.
// Cross-queue stage/access normalization is handled upstream in the abstract
// sync planner (SYNC-01): old_state for cross-queue transitions is already
// normalized to (ALL_COMMANDS, MEMORY_READ|MEMORY_WRITE), so no clamping is
// needed here. new_state is always queue-compatible by construction in
// get_image_state / get_buffer_state.
pub fn translate_plan(
    backend: &crate::backend::VulkanBackend,
    plan: &abstract_sync::SyncPlan,
) -> crate::sync::SyncPlan {
    let mut vk_plan = crate::sync::SyncPlan::default();
    vk_plan.pass_sync = vec![PassSyncData::default(); plan.passes.len()];

    for (i, abstract_pass) in plan.passes.iter().enumerate() {
        for transition in &abstract_pass.pre_transitions {
            match transition.resource_kind {
                abstract_sync::ResourceKind::Image => {
                    let physical_id = backend.external_to_physical.get(&transition.resource_id).copied();
                    if let Some(physical_id) = physical_id {
                        if let Some(img) = backend.images.get(physical_id) {
                            let barrier = vk::ImageMemoryBarrier2::default()
                                .image(img.image)
                                .old_layout(translate_layout_from_abstract(transition.old_state.layout))
                                .new_layout(translate_layout_from_abstract(transition.new_state.layout))
                                .src_access_mask(translate_access_from_abstract(transition.old_state.access))
                                .dst_access_mask(translate_access_from_abstract(transition.new_state.access))
                                .src_stage_mask(translate_stages_from_abstract(transition.old_state.stage))
                                .dst_stage_mask(translate_stages_from_abstract(transition.new_state.stage))
                                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: if img.format == vk::Format::D32_SFLOAT { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR },
                                    base_mip_level: 0,
                                    level_count: img.desc.mip_levels.max(1),
                                    base_array_layer: 0,
                                    layer_count: img.desc.array_layers.max(1),
                                });
                            vk_plan.pass_sync[i].pre_barriers.push(Barrier::Image(barrier));
                        }
                    }
                }
                abstract_sync::ResourceKind::Buffer => {
                    let physical_id = backend.external_buffer_to_physical.get(&transition.resource_id).copied();
                    if let Some(physical_id) = physical_id {
                        if let Some(buf) = backend.buffers.get(physical_id) {
                            let barrier = vk::BufferMemoryBarrier2::default()
                                .buffer(buf.buffer)
                                .src_access_mask(translate_access_from_abstract(transition.old_state.access))
                                .dst_access_mask(translate_access_from_abstract(transition.new_state.access))
                                .src_stage_mask(translate_stages_from_abstract(transition.old_state.stage))
                                .dst_stage_mask(translate_stages_from_abstract(transition.new_state.stage))
                                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                .offset(0)
                                .size(vk::WHOLE_SIZE);
                            vk_plan.pass_sync[i].pre_barriers.push(Barrier::Buffer(barrier));
                        }
                    }
                }
                abstract_sync::ResourceKind::AccelStruct => {
                    // TODO: Handle AS transitions (memory barriers, no layout)
                }
            }
        }

        for transition in &abstract_pass.post_transitions {
            if let abstract_sync::ResourceKind::Image = transition.resource_kind {
                let physical_id = backend.external_to_physical.get(&transition.resource_id).copied();
                if let Some(physical_id) = physical_id {
                    if let Some(img) = backend.images.get(physical_id) {
                        let barrier = vk::ImageMemoryBarrier2::default()
                            .image(img.image)
                            .old_layout(translate_layout_from_abstract(transition.old_state.layout))
                            .new_layout(translate_layout_from_abstract(transition.new_state.layout))
                            .src_access_mask(translate_access_from_abstract(transition.old_state.access))
                            .dst_access_mask(translate_access_from_abstract(transition.new_state.access))
                            .src_stage_mask(translate_stages_from_abstract(transition.old_state.stage))
                            .dst_stage_mask(translate_stages_from_abstract(transition.new_state.stage))
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: if img.format == vk::Format::D32_SFLOAT { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR },
                                base_mip_level: 0,
                                level_count: img.desc.mip_levels.max(1),
                                base_array_layer: 0,
                                layer_count: img.desc.array_layers.max(1),
                            });
                        vk_plan.pass_sync[i].post_barriers.push(Barrier::Image(barrier));
                    }
                }
            }
        }

        for (virtual_id, op) in &abstract_pass.load_ops {
            if let Some(&physical_id) = backend.external_to_physical.get(virtual_id) {
                if let Some(img) = backend.images.get(physical_id) {
                    vk_plan.pass_sync[i].load_ops.insert(img.image, match op {
                        abstract_sync::LoadOp::Load => vk::AttachmentLoadOp::LOAD,
                        abstract_sync::LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
                        abstract_sync::LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
                    });
                }
            }
        }
    }

    vk_plan
}
