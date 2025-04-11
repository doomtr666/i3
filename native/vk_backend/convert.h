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
VkShaderStageFlagBits i3_vk_convert_shader_stage(i3_rbk_shader_stage_flag_bits_t stage);

// vertex input rate
VkVertexInputRate i3_vk_convert_vertex_input_rate(i3_rbk_vertex_input_rate_t rate);

// primitive topology
VkPrimitiveTopology i3_vk_convert_primitive_topology(i3_rbk_primitive_topology_t topology);
