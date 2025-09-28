#pragma once

#include "common.h"

// filter
VkFilter i3_vk_convert_filter(i3_rbk_filter_t filter);

// sampler mipmap mode
VkSamplerMipmapMode i3_vk_convert_sampler_mipmap_mode(i3_rbk_sampler_mipmap_mode_t mode);

// sampler address mode
VkSamplerAddressMode i3_vk_convert_sampler_address_mode(i3_rbk_sampler_address_mode_t mode);

// border color
VkBorderColor i3_vk_convert_border_color(i3_rbk_border_color_t color);

// compare op
VkCompareOp i3_vk_convert_compare_op(i3_rbk_compare_op_t op);

// format
VkFormat i3_vk_convert_format(i3_rbk_format_t format);
bool i3_vk_is_depth_format(VkFormat format);
bool i3_vk_is_srgb_format(VkFormat format);

// image type
VkImageType i3_vk_convert_image_type(i3_rbk_image_type_t type);

// sample count
VkSampleCountFlagBits i3_vk_convert_sample_count(uint32_t count);

// view type
VkImageViewType i3_vk_convert_image_view_type(i3_rbk_image_view_type_t type);

// component swizzle
VkComponentSwizzle i3_vk_convert_component_swizzle(i3_rbk_component_swizzle_t swizzle);

// image aspect flags
VkImageAspectFlags i3_vk_convert_image_aspect_flags(i3_rbk_image_aspect_flags_t flags);

// shader stage
VkShaderStageFlagBits i3_vk_convert_shader_stage(i3_rbk_shader_stage_flags_t stage);
VkShaderStageFlags i3_vk_convert_shader_stage_flags(i3_rbk_shader_stage_flags_t stage);

// vertex input rate
VkVertexInputRate i3_vk_convert_vertex_input_rate(i3_rbk_vertex_input_rate_t rate);

// primitive topology
VkPrimitiveTopology i3_vk_convert_primitive_topology(i3_rbk_primitive_topology_t topology);

// polygon mode
VkPolygonMode i3_vk_convert_polygon_mode(i3_rbk_polygon_mode_t mode);

// cull mode flags
VkCullModeFlags i3_vk_convert_cull_mode_flags(i3_rbk_cull_mode_flags_t flags);

// front face
VkFrontFace i3_vk_convert_front_face(i3_rbk_front_face_t face);

// stencil op
VkStencilOp i3_vk_convert_stencil_op(i3_rbk_stencil_op_t op);

// logic op
VkLogicOp i3_vk_convert_logic_op(i3_rbk_logic_op_t op);

// blend factor
VkBlendFactor i3_vk_convert_blend_factor(i3_rbk_blend_factor_t factor);

// blend op
VkBlendOp i3_vk_convert_blend_op(i3_rbk_blend_op_t op);

// color component flags
VkColorComponentFlags i3_vk_convert_color_component_flags(i3_rbk_color_component_flags_t flags);

// dynamic state
VkDynamicState i3_vk_convert_dynamic_state(i3_rbk_dynamic_state_t state);

// descriptor type
VkDescriptorType i3_vk_convert_descriptor_type(i3_rbk_descriptor_type_t type);

// index type
VkIndexType i3_vk_convert_index_type(i3_rbk_index_type_t type);

// attachment load op
VkAttachmentLoadOp i3_vk_convert_attachment_load_op(i3_rbk_attachment_load_op_t op);
// attachment store op
VkAttachmentStoreOp i3_vk_convert_attachment_store_op(i3_rbk_attachment_store_op_t op);