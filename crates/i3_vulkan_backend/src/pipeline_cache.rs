//! # Pipeline Cache - Pipeline Creation and Caching
//!
//! This module handles the creation of Vulkan graphics and compute pipelines.
//! Pipelines are expensive to create, so they are cached and reused.
//!
//! ## Pipeline Creation
//!
//! Pipeline creation involves:
//! 1. **Shader module creation**: Compiling SPIR-V bytecode into GPU-executable code
//! 2. **Vertex input setup**: Defining vertex attributes and bindings
//! 3. **Rasterization state**: Configuring culling, depth testing, etc.
//! 4. **Color blend state**: Configuring color attachment blending
//! 5. **Descriptor set layout**: Creating layouts for shader resource binding
//! 6. **Pipeline layout**: Combining descriptor layouts and push constant ranges
//!
//! ## Caching Strategy
//!
//! Pipelines are stored in the [`ResourceArena`] and can be looked up by handle.
//! The backend does not currently implement disk-based pipeline caching (VkPipelineCache),
//! but this could be added for faster startup times.
//!
//! ## Dynamic State
//!
//! The backend uses dynamic state for viewport and scissor to avoid recreating
//! pipelines when the window is resized.

use ash::vk;
use i3_gfx::graph::backend::BackendPipeline;
use i3_gfx::graph::pipeline::*;
use std::collections::HashMap;
use tracing::debug;

use crate::backend::VulkanBackend;
use crate::convert::*;
use crate::resource_arena::PhysicalPipeline;
use i3_io::pipeline_asset::BakeableGraphicsPipeline;

/// Create a graphics pipeline from the given description.
///
/// This function creates a complete graphics pipeline including:
/// - Shader modules (vertex, fragment, etc.)
/// - Vertex input state
/// - Input assembly state
/// - Rasterization state
/// - Multisample state
/// - Depth stencil state
/// - Color blend state
/// - Dynamic state (viewport, scissor)
/// - Descriptor set layouts
/// - Pipeline layout
///
/// # Arguments
///
/// * `backend` - Mutable reference to the backend
/// * `desc` - Pipeline creation description
///
/// # Returns
///
/// Handle to the created pipeline in the ResourceArena
pub fn create_graphics_pipeline(
    backend: &mut VulkanBackend,
    desc: &GraphicsPipelineCreateInfo,
) -> BackendPipeline {
    let device = backend.get_device().clone();
    let id = backend.next_id();
    debug!("Creating Graphics Pipeline");

    // 1. Create Shader Module (once per pipeline setup)
    let create_info = vk::ShaderModuleCreateInfo::default().code(unsafe {
        std::slice::from_raw_parts(
            desc.shader_module.bytecode.as_ptr() as *const u32,
            desc.shader_module.bytecode.len() / 4,
        )
    });

    let module = unsafe { device.handle.create_shader_module(&create_info, None) }
        .expect("Shader module creation failed");
    backend.shader_modules.push(module);

    let mut stages = Vec::new();

    // Create CStrings first to ensure stable pointers
    let entry_points: Vec<std::ffi::CString> = desc
        .shader_module
        .stages
        .iter()
        .map(|s| std::ffi::CString::new(s.entry_point.as_str()).unwrap())
        .collect();

    for (stage_info, entry_point_cstr) in desc.shader_module.stages.iter().zip(&entry_points) {
        let stage_flag = if stage_info.stage.contains(ShaderStageFlags::Vertex) {
            vk::ShaderStageFlags::VERTEX
        } else if stage_info.stage.contains(ShaderStageFlags::Fragment) {
            vk::ShaderStageFlags::FRAGMENT
        } else if stage_info.stage.contains(ShaderStageFlags::Compute) {
            vk::ShaderStageFlags::COMPUTE
        } else {
            vk::ShaderStageFlags::empty()
        };

        stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .module(module)
                .stage(stage_flag)
                .name(entry_point_cstr.as_c_str()),
        );
    }

    // 2. Vertex Input
    let vk_vertex_bindings: Vec<vk::VertexInputBindingDescription> = desc
        .vertex_input
        .bindings
        .iter()
        .map(|b| vk::VertexInputBindingDescription {
            binding: b.binding,
            stride: b.stride,
            input_rate: convert_vertex_input_rate(b.input_rate),
        })
        .collect();

    let vk_vertex_attributes: Vec<vk::VertexInputAttributeDescription> = desc
        .vertex_input
        .attributes
        .iter()
        .map(|a| vk::VertexInputAttributeDescription {
            location: a.location,
            binding: a.binding,
            format: convert_vertex_format(a.format),
            offset: a.offset,
        })
        .collect();

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vk_vertex_bindings)
        .vertex_attribute_descriptions(&vk_vertex_attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(convert_primitive_topology(desc.input_assembly.topology))
        .primitive_restart_enable(desc.input_assembly.primitive_restart_enable);

    let tessellation = vk::PipelineTessellationStateCreateInfo::default()
        .patch_control_points(desc.tessellation_state.patch_control_points);

    // 3. Dynamic States
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(desc.rasterization_state.depth_clamp_enable)
        .rasterizer_discard_enable(desc.rasterization_state.rasterizer_discard_enable)
        .polygon_mode(convert_polygon_mode(desc.rasterization_state.polygon_mode))
        .cull_mode(convert_cull_mode(desc.rasterization_state.cull_mode))
        // Engine Convention: Vulkan uses Clockwise Front Face to compensate for Negative Viewport
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(desc.rasterization_state.depth_bias_enable)
        .depth_bias_constant_factor(desc.rasterization_state.depth_bias_constant_factor)
        .depth_bias_clamp(desc.rasterization_state.depth_bias_clamp)
        .depth_bias_slope_factor(desc.rasterization_state.depth_bias_slope_factor)
        .line_width(desc.rasterization_state.line_width);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(convert_sample_count(desc.multisample_state.sample_count))
        .sample_shading_enable(desc.multisample_state.sample_shading_enable)
        .alpha_to_coverage_enable(desc.multisample_state.alpha_to_coverage_enable);

    // 4. Depth Stencil

    // 5. Color Blend
    let attachments: Vec<vk::PipelineColorBlendAttachmentState> = desc
        .render_targets
        .color_targets
        .iter()
        .map(|target| {
            let mut attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(convert_color_component_flags(target.write_mask));

            if let Some(blend) = target.blend {
                attachment = attachment
                    .blend_enable(true)
                    .src_color_blend_factor(convert_blend_factor(blend.src_color_factor))
                    .dst_color_blend_factor(convert_blend_factor(blend.dst_color_factor))
                    .color_blend_op(convert_blend_op(blend.color_op))
                    .src_alpha_blend_factor(convert_blend_factor(blend.src_alpha_factor))
                    .dst_alpha_blend_factor(convert_blend_factor(blend.dst_alpha_factor))
                    .alpha_blend_op(convert_blend_op(blend.alpha_op));
            } else {
                attachment = attachment.blend_enable(false);
            }
            attachment
        })
        .collect();

    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&attachments)
        .logic_op_enable(desc.render_targets.logic_op.is_some())
        .logic_op(convert_logic_op(
            desc.render_targets.logic_op.unwrap_or(LogicOp::NoOp),
        ));

    // 5. Layout (Push Constants + Descriptor Sets)

    // Group bindings by set index
    let mut set_bindings: HashMap<u32, Vec<vk::DescriptorSetLayoutBinding>> = HashMap::new();
    for binding in &desc.shader_module.reflection.bindings {
        let descriptor_type = convert_binding_type_to_descriptor(binding.binding_type.clone());

        let stage_flags = convert_shader_stage_flags(ShaderStageFlags::All); // Simplified for MVP

        let vk_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding.binding)
            .descriptor_type(descriptor_type)
            .descriptor_count(binding.count)
            .stage_flags(stage_flags);

        set_bindings
            .entry(binding.set)
            .or_default()
            .push(vk_binding);
    }

    // Create Descriptor Set Layouts (filling gaps)
    let mut descriptor_set_layouts = Vec::new();
    if !set_bindings.is_empty() || backend.bindless_set_layout != vk::DescriptorSetLayout::null() {
        // Force at least 3 sets (0, 1, 2) to ensure Bindless is always at Set 2
        let max_set = (*set_bindings.keys().max().unwrap_or(&0)).max(2);
        for i in 0..=max_set {
            let bindings = set_bindings.get(&i).map(|v| v.as_slice()).unwrap_or(&[]);
            let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default();
            let mut binding_flags = Vec::with_capacity(bindings.len());
            for b in bindings {
                if b.descriptor_count >= 1000 || b.descriptor_count == 0 {
                    binding_flags.push(
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND
                            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                    );
                } else {
                    binding_flags.push(vk::DescriptorBindingFlags::empty());
                }
            }

            binding_flags_info = binding_flags_info.binding_flags(&binding_flags);

            let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(bindings)
                .push_next(&mut binding_flags_info);

            // If any binding has UPDATE_AFTER_BIND, the set layout itself needs the flag
            if binding_flags
                .iter()
                .any(|f| f.contains(vk::DescriptorBindingFlags::UPDATE_AFTER_BIND))
            {
                layout_info.flags |= vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
            }

            let layout = if i == 2 && backend.bindless_set_layout != vk::DescriptorSetLayout::null()
            {
                tracing::debug!(
                    "Pipeline reusing global bindless layout for Set 2: {:?}",
                    backend.bindless_set_layout
                );
                backend.bindless_set_layout
            } else {
                let layout = unsafe {
                    device
                        .handle
                        .create_descriptor_set_layout(&layout_info, None)
                        .expect("Failed to create descriptor set layout")
                };
                backend.descriptor_set_layouts.push(layout); // Track for cleanup
                tracing::debug!("Pipeline created new layout for Set {}: {:?}", i, layout);
                layout
            };

            descriptor_set_layouts.push(layout);
        }
    }

    // Push Constants from reflection
    let pc_ranges: Vec<vk::PushConstantRange> = desc
        .shader_module
        .reflection
        .push_constants
        .iter()
        .map(|pc| vk::PushConstantRange {
            stage_flags: convert_shader_stage_flags(ShaderStageFlags::from_bits_truncate(
                pc.stage_flags.bits(),
            )),
            offset: pc.offset,
            size: pc.size,
        })
        .collect();

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&descriptor_set_layouts)
        .push_constant_ranges(&pc_ranges);

    let pipeline_layout =
        unsafe { device.handle.create_pipeline_layout(&layout_info, None) }.unwrap();

    // 6. Dynamic Rendering Info
    let mut color_formats = Vec::new();
    for rt in &desc.render_targets.color_targets {
        color_formats.push(convert_format(rt.format));
    }

    let depth_format = desc
        .render_targets
        .depth_stencil_format
        .map(|f| convert_format(f))
        .unwrap_or(vk::Format::UNDEFINED);

    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(depth_format);

    debug!(
        "Pipeline {:?} formats: color={:?}, depth={:?}",
        id, color_formats, depth_format
    );

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(desc.depth_stencil_state.depth_test_enable)
        .depth_write_enable(desc.depth_stencil_state.depth_write_enable)
        .depth_compare_op(convert_compare_op(
            desc.depth_stencil_state.depth_compare_op,
        ))
        .depth_bounds_test_enable(desc.depth_stencil_state.depth_bounds_test_enable)
        .stencil_test_enable(desc.depth_stencil_state.stencil_test_enable)
        .front(convert_stencil_op_state(&desc.depth_stencil_state.front))
        .back(convert_stencil_op_state(&desc.depth_stencil_state.back))
        .min_depth_bounds(desc.depth_stencil_state.min_depth_bounds)
        .max_depth_bounds(desc.depth_stencil_state.max_depth_bounds);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .tessellation_state(&tessellation)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .push_next(&mut rendering_info);

    let pipeline = unsafe {
        device
            .handle
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .expect("Pipeline creation failed")[0];

    // Create PhysicalPipeline struct
    let physical = PhysicalPipeline {
        handle: pipeline,
        layout: pipeline_layout,
        bind_point: vk::PipelineBindPoint::GRAPHICS,
        set_layouts: descriptor_set_layouts,
        physical_id: 0,
    };

    let physical_handle = backend.pipeline_resources.insert(physical);
    backend
        .pipeline_resources
        .get_mut(physical_handle)
        .unwrap()
        .physical_id = physical_handle;
    BackendPipeline(physical_handle)
}

/// Create a compute pipeline from the given description.
///
/// Compute pipelines are simpler than graphics pipelines because they only have
/// a single shader stage and no fixed-function state (rasterization, blending, etc.).
///
/// # Arguments
///
/// * `backend` - Mutable reference to the backend
/// * `desc` - Compute pipeline creation description
///
/// # Returns
///
/// Handle to the created pipeline in the ResourceArena
pub fn create_compute_pipeline(
    backend: &mut VulkanBackend,
    desc: &ComputePipelineCreateInfo,
) -> BackendPipeline {
    let device = backend.get_device().clone();
    let _id = backend.next_id();
    debug!("Creating Compute Pipeline");

    // 1. Create Shader Module
    let create_info = vk::ShaderModuleCreateInfo::default().code(unsafe {
        std::slice::from_raw_parts(
            desc.shader_module.bytecode.as_ptr() as *const u32,
            desc.shader_module.bytecode.len() / 4,
        )
    });

    let module = unsafe { device.handle.create_shader_module(&create_info, None) }
        .expect("Shader module creation failed");
    backend.shader_modules.push(module);

    // Compute has exactly one stage
    let entry_point =
        std::ffi::CString::new(desc.shader_module.stages[0].entry_point.as_str()).unwrap();
    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry_point);

    // 2. Layout
    let mut set_bindings: HashMap<u32, Vec<vk::DescriptorSetLayoutBinding>> = HashMap::new();
    for binding in &desc.shader_module.reflection.bindings {
        let descriptor_type = convert_binding_type_to_descriptor(binding.binding_type.clone());
        let stage_flags = vk::ShaderStageFlags::COMPUTE;

        let vk_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding.binding)
            .descriptor_type(descriptor_type)
            .descriptor_count(binding.count)
            .stage_flags(stage_flags);

        set_bindings
            .entry(binding.set)
            .or_default()
            .push(vk_binding);
    }

    let mut descriptor_set_layouts = Vec::new();
    if !set_bindings.is_empty() {
        let max_set = *set_bindings.keys().max().unwrap();
        for i in 0..=max_set {
            let bindings = set_bindings.get(&i).map(|v| v.as_slice()).unwrap_or(&[]);
            let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default();
            let mut binding_flags = Vec::with_capacity(bindings.len());
            for b in bindings {
                if b.descriptor_count >= 1000 || b.descriptor_count == 0 {
                    binding_flags.push(
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND
                            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                    );
                } else {
                    binding_flags.push(vk::DescriptorBindingFlags::empty());
                }
            }

            binding_flags_info = binding_flags_info.binding_flags(&binding_flags);

            let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(bindings)
                .push_next(&mut binding_flags_info);

            // If any binding has UPDATE_AFTER_BIND, the set layout itself needs the flag
            if binding_flags
                .iter()
                .any(|f| f.contains(vk::DescriptorBindingFlags::UPDATE_AFTER_BIND))
            {
                layout_info.flags |= vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
            }

            let layout = if i == 2 && backend.bindless_set_layout != vk::DescriptorSetLayout::null()
            {
                backend.bindless_set_layout
            } else {
                let layout = unsafe {
                    device
                        .handle
                        .create_descriptor_set_layout(&layout_info, None)
                        .expect("Failed to create descriptor set layout")
                };
                backend.descriptor_set_layouts.push(layout); // Track for cleanup
                layout
            };

            descriptor_set_layouts.push(layout);
        }
    }

    let pc_ranges: Vec<vk::PushConstantRange> = desc
        .shader_module
        .reflection
        .push_constants
        .iter()
        .map(|pc| vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: pc.offset,
            size: pc.size,
        })
        .collect();

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&descriptor_set_layouts)
        .push_constant_ranges(&pc_ranges);

    let pipeline_layout =
        unsafe { device.handle.create_pipeline_layout(&layout_info, None) }.unwrap();

    // 3. Pipeline
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(pipeline_layout);

    let pipeline = unsafe {
        device
            .handle
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .expect("Compute pipeline creation failed")[0];

    // Create PhysicalPipeline struct
    let physical = PhysicalPipeline {
        handle: pipeline,
        layout: pipeline_layout,
        bind_point: vk::PipelineBindPoint::COMPUTE,
        set_layouts: descriptor_set_layouts,
        physical_id: 0,
    };

    let handle = backend.pipeline_resources.insert(physical);
    backend
        .pipeline_resources
        .get_mut(handle)
        .unwrap()
        .physical_id = handle;
    BackendPipeline(handle)
}

pub fn create_graphics_pipeline_from_baked(
    backend: &mut VulkanBackend,
    baked: &BakeableGraphicsPipeline,
    reflection_bytes: &[u8],
    bytecode: &[u8],
) -> BackendPipeline {
    use i3_gfx::graph::pipeline::ShaderReflection;
    let reflection: ShaderReflection =
        postcard::from_bytes(reflection_bytes).expect("Failed to deserialize reflection");
    tracing::debug!("Creating Baked Graphics Pipeline with {} bindings", reflection.bindings.len());
    let device = backend.get_device().clone();
    let _id = backend.next_id();

    // 1. Shader Module
    let create_info = vk::ShaderModuleCreateInfo::default().code(unsafe {
        std::slice::from_raw_parts(bytecode.as_ptr() as *const u32, bytecode.len() / 4)
    });

    let module = unsafe { device.handle.create_shader_module(&create_info, None) }
        .expect("Shader module creation failed");
    backend.shader_modules.push(module);

    let mut stages = Vec::new();
    let entry_points: Vec<std::ffi::CString> = reflection
        .entry_points
        .iter()
        .map(|e| std::ffi::CString::new(e.name.as_str()).unwrap())
        .collect();

    for (entry_info, entry_point_cstr) in reflection.entry_points.iter().zip(&entry_points) {
        let stage_flag = match entry_info.stage.as_str() {
            "vertex" => vk::ShaderStageFlags::VERTEX,
            "fragment" => vk::ShaderStageFlags::FRAGMENT,
            "compute" => vk::ShaderStageFlags::COMPUTE,
            _ => vk::ShaderStageFlags::empty(),
        };

        if stage_flag != vk::ShaderStageFlags::empty() {
            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .module(module)
                    .stage(stage_flag)
                    .name(entry_point_cstr.as_c_str()),
            );
        }
    }

    // 2. Vertex Input
    let vk_vertex_bindings: Vec<vk::VertexInputBindingDescription> = baked.vertex_bindings
        [..baked.vertex_binding_count as usize]
        .iter()
        .map(|b| vk::VertexInputBindingDescription {
            binding: b.binding,
            stride: b.stride,
            input_rate: if b.input_rate == 1 {
                vk::VertexInputRate::INSTANCE
            } else {
                vk::VertexInputRate::VERTEX
            },
        })
        .collect();

    let vk_vertex_attributes: Vec<vk::VertexInputAttributeDescription> = baked.vertex_attributes
        [..baked.vertex_attribute_count as usize]
        .iter()
        .map(|a| vk::VertexInputAttributeDescription {
            location: a.location,
            binding: a.binding,
            format: convert_u32_vertex_format(a.format),
            offset: a.offset,
        })
        .collect();

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vk_vertex_bindings)
        .vertex_attribute_descriptions(&vk_vertex_attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(convert_u32_topology(baked.topology))
        .primitive_restart_enable(baked.primitive_restart_enable != 0);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(baked.rasterization.depth_clamp_enable != 0)
        .rasterizer_discard_enable(baked.rasterization.rasterizer_discard_enable != 0)
        .polygon_mode(convert_u32_polygon_mode(baked.rasterization.polygon_mode))
        .cull_mode(convert_u32_cull_mode(baked.rasterization.cull_mode))
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(baked.rasterization.depth_bias_enable != 0)
        .depth_bias_constant_factor(baked.rasterization.depth_bias_constant_factor)
        .depth_bias_clamp(baked.rasterization.depth_bias_clamp)
        .depth_bias_slope_factor(baked.rasterization.depth_bias_slope_factor)
        .line_width(baked.rasterization.line_width);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(baked.depth_stencil.depth_test_enable != 0)
        .depth_write_enable(baked.depth_stencil.depth_write_enable != 0)
        .depth_compare_op(convert_u32_compare_op(baked.depth_stencil.depth_compare_op))
        .stencil_test_enable(baked.depth_stencil.stencil_test_enable != 0)
        .front(vk::StencilOpState {
            fail_op: convert_u32_stencil_op(baked.depth_stencil.front.fail_op),
            pass_op: convert_u32_stencil_op(baked.depth_stencil.front.pass_op),
            depth_fail_op: convert_u32_stencil_op(baked.depth_stencil.front.depth_fail_op),
            compare_op: convert_u32_compare_op(baked.depth_stencil.front.compare_op),
            compare_mask: baked.depth_stencil.front.compare_mask,
            write_mask: baked.depth_stencil.front.write_mask,
            reference: baked.depth_stencil.front.reference,
        })
        .back(vk::StencilOpState {
            fail_op: convert_u32_stencil_op(baked.depth_stencil.back.fail_op),
            pass_op: convert_u32_stencil_op(baked.depth_stencil.back.pass_op),
            depth_fail_op: convert_u32_stencil_op(baked.depth_stencil.back.depth_fail_op),
            compare_op: convert_u32_compare_op(baked.depth_stencil.back.compare_op),
            compare_mask: baked.depth_stencil.back.compare_mask,
            write_mask: baked.depth_stencil.back.write_mask,
            reference: baked.depth_stencil.back.reference,
        })
        .depth_bounds_test_enable(baked.depth_stencil.depth_bounds_test_enable != 0)
        .min_depth_bounds(baked.depth_stencil.min_depth_bounds)
        .max_depth_bounds(baked.depth_stencil.max_depth_bounds);

    let color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = baked.color_targets
        [..baked.color_target_count as usize]
        .iter()
        .map(|target| {
            let mut attachment = vk::PipelineColorBlendAttachmentState::default().color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            );
            if target.blend_enable != 0 {
                attachment = attachment
                    .blend_enable(true)
                    .src_color_blend_factor(convert_u32_blend_factor(target.src_color_factor))
                    .dst_color_blend_factor(convert_u32_blend_factor(target.dst_color_factor))
                    .color_blend_op(convert_u32_blend_op(target.color_op))
                    .src_alpha_blend_factor(convert_u32_blend_factor(target.src_alpha_factor))
                    .dst_alpha_blend_factor(convert_u32_blend_factor(target.dst_alpha_factor))
                    .alpha_blend_op(convert_u32_blend_op(target.alpha_op));
            }
            attachment
        })
        .collect();

    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

    // 3. Layout
    let mut set_bindings: HashMap<u32, Vec<vk::DescriptorSetLayoutBinding>> = HashMap::new();
    for binding in &reflection.bindings {
        let vk_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding.binding)
            .descriptor_type(convert_binding_type_to_descriptor(
                binding.binding_type.clone(),
            ))
            .descriptor_count(binding.count)
            .stage_flags(convert_shader_stage_flags(ShaderStageFlags::All));
        set_bindings
            .entry(binding.set)
            .or_default()
            .push(vk_binding);
    }

    let mut descriptor_set_layouts = Vec::new();

    // Ensure Set 2 is available for bindless
    let max_set = (*set_bindings.keys().max().unwrap_or(&0)).max(2);
    for i in 0..=max_set {
        let bindings = set_bindings.get(&i).map(|v| v.as_slice()).unwrap_or(&[]);

        let layout = if i == 2 && backend.bindless_set_layout != vk::DescriptorSetLayout::null() {
            backend.bindless_set_layout
        } else {
            let mut binding_flags = Vec::with_capacity(bindings.len());
            for b in bindings {
                if b.descriptor_count >= 1000 || b.descriptor_count == 0 {
                    binding_flags.push(
                        vk::DescriptorBindingFlags::PARTIALLY_BOUND
                            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                    );
                } else {
                    binding_flags.push(vk::DescriptorBindingFlags::empty());
                }
            }
            let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&binding_flags);
            let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(bindings)
                .push_next(&mut flags_info);

            if binding_flags
                .iter()
                .any(|f| f.contains(vk::DescriptorBindingFlags::UPDATE_AFTER_BIND))
            {
                layout_info.flags |= vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
            }

            let layout = unsafe {
                device
                    .handle
                    .create_descriptor_set_layout(&layout_info, None)
                    .unwrap()
            };
            backend.descriptor_set_layouts.push(layout);
            layout
        };
        descriptor_set_layouts.push(layout);
    }

    let pc_ranges: Vec<vk::PushConstantRange> = reflection
        .push_constants
        .iter()
        .map(|pc| vk::PushConstantRange {
            stage_flags: convert_shader_stage_flags(ShaderStageFlags::from_bits_truncate(
                pc.stage_flags.bits(),
            )),
            offset: pc.offset,
            size: pc.size,
        })
        .collect();

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&descriptor_set_layouts)
        .push_constant_ranges(&pc_ranges);
    let pipeline_layout = unsafe {
        device
            .handle
            .create_pipeline_layout(&layout_info, None)
            .expect("Failed to create pipeline layout")
    };

    // 4. Rendering Info
    let color_formats: Vec<vk::Format> = baked.color_targets[..baked.color_target_count as usize]
        .iter()
        .map(|t| convert_u32_format(t.format))
        .collect();
    let depth_format = convert_u32_format(baked.depth_stencil_format);
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(depth_format);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .push_next(&mut rendering_info);

    let pipeline = unsafe {
        device
            .handle
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .unwrap()[0]
    };

    let physical_handle = backend.pipeline_resources.insert(PhysicalPipeline {
        handle: pipeline,
        layout: pipeline_layout,
        bind_point: vk::PipelineBindPoint::GRAPHICS,
        set_layouts: descriptor_set_layouts,
        physical_id: 0,
    });
    backend
        .pipeline_resources
        .get_mut(physical_handle)
        .unwrap()
        .physical_id = physical_handle;
    BackendPipeline(physical_handle)
}

pub fn create_compute_pipeline_from_baked(
    backend: &mut VulkanBackend,
    reflection_bytes: &[u8],
    bytecode: &[u8],
) -> BackendPipeline {
    use i3_gfx::graph::pipeline::ShaderReflection;
    let reflection: ShaderReflection =
        postcard::from_bytes(reflection_bytes).expect("Failed to deserialize reflection");
    tracing::debug!("Creating Baked Compute Pipeline with {} bindings", reflection.bindings.len());
    let device = backend.get_device().clone();

    // 1. Shader Module
    let create_info = vk::ShaderModuleCreateInfo::default().code(unsafe {
        std::slice::from_raw_parts(bytecode.as_ptr() as *const u32, bytecode.len() / 4)
    });
    let module = unsafe { device.handle.create_shader_module(&create_info, None) }
        .expect("Shader module creation failed");
    backend.shader_modules.push(module);

    let entry_point = std::ffi::CString::new("main").unwrap();
    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry_point);

    // 2. Layout
    let mut set_bindings: HashMap<u32, Vec<vk::DescriptorSetLayoutBinding>> = HashMap::new();
    for binding in &reflection.bindings {
        let vk_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding.binding)
            .descriptor_type(convert_binding_type_to_descriptor(
                binding.binding_type.clone(),
            ))
            .descriptor_count(binding.count)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        set_bindings
            .entry(binding.set)
            .or_default()
            .push(vk_binding);
    }

    let mut descriptor_set_layouts = Vec::new();

    if !set_bindings.is_empty() {
        let max_set = (*set_bindings.keys().max().unwrap_or(&0)).max(2);
        for i in 0..=max_set {
            let bindings = set_bindings.get(&i).map(|v| v.as_slice()).unwrap_or(&[]);

            let layout = if i == 2 && backend.bindless_set_layout != vk::DescriptorSetLayout::null()
            {
                backend.bindless_set_layout
            } else {
                let mut binding_flags = Vec::with_capacity(bindings.len());
                for b in bindings {
                    if b.descriptor_count >= 1000 || b.descriptor_count == 0 {
                        binding_flags.push(
                            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                        );
                    } else {
                        binding_flags.push(vk::DescriptorBindingFlags::empty());
                    }
                }
                let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                    .binding_flags(&binding_flags);
                let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(bindings)
                    .push_next(&mut flags_info);

                if binding_flags
                    .iter()
                    .any(|f| f.contains(vk::DescriptorBindingFlags::UPDATE_AFTER_BIND))
                {
                    layout_info.flags |= vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL;
                }

                let layout = unsafe {
                    device
                        .handle
                        .create_descriptor_set_layout(&layout_info, None)
                        .unwrap()
                };
                backend.descriptor_set_layouts.push(layout);
                layout
            };
            descriptor_set_layouts.push(layout);
        }
    }

    let pc_ranges: Vec<vk::PushConstantRange> = reflection
        .push_constants
        .iter()
        .map(|pc| vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: pc.offset,
            size: pc.size,
        })
        .collect();

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&descriptor_set_layouts)
        .push_constant_ranges(&pc_ranges);
    let pipeline_layout = unsafe {
        device
            .handle
            .create_pipeline_layout(&layout_info, None)
            .expect("Failed to create pipeline layout")
    };

    // 3. Pipeline
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(pipeline_layout);
    let pipeline = unsafe {
        device
            .handle
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .expect("Compute pipeline creation failed")[0]
    };

    let physical_handle = backend.pipeline_resources.insert(PhysicalPipeline {
        handle: pipeline,
        layout: pipeline_layout,
        bind_point: vk::PipelineBindPoint::COMPUTE,
        set_layouts: descriptor_set_layouts,
        physical_id: 0,
    });
    backend
        .pipeline_resources
        .get_mut(physical_handle)
        .unwrap()
        .physical_id = physical_handle;
    BackendPipeline(physical_handle)
}
