#include "convert.h"

// filter
VkFilter i3_vk_convert_filter(i3_rbk_filter_t filter)
{
    switch (filter)
    {
    case I3_RBK_FILTER_NEAREST:
        return VK_FILTER_NEAREST;
    case I3_RBK_FILTER_LINEAR:
        return VK_FILTER_LINEAR;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported filter: %d", filter);
        return VK_FILTER_NEAREST;
    }
    }
}

// sampler mipmap mode
VkSamplerMipmapMode i3_vk_convert_sampler_mipmap_mode(i3_rbk_sampler_mipmap_mode_t mode)
{
    switch (mode)
    {
    case I3_RBK_SAMPLER_MIPMAP_MODE_NEAREST:
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    case I3_RBK_SAMPLER_MIPMAP_MODE_LINEAR:
        return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported sampler mipmap mode: %d", mode);
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    }
    }
}

// sampler address mode
VkSamplerAddressMode i3_vk_convert_sampler_address_mode(i3_rbk_sampler_address_mode_t mode)
{
    switch (mode)
    {
    case I3_RBK_SAMPLER_ADDRESS_MODE_REPEAT:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case I3_RBK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case I3_RBK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case I3_RBK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported sampler address mode: %d", mode);
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
    }
}

// border color
VkBorderColor i3_vk_convert_border_color(i3_rbk_border_color_t color)
{
    switch (color)
    {
    case I3_RBK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK:
        return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    case I3_RBK_BORDER_COLOR_INT_TRANSPARENT_BLACK:
        return VK_BORDER_COLOR_INT_TRANSPARENT_BLACK;
    case I3_RBK_BORDER_COLOR_FLOAT_OPAQUE_BLACK:
        return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    case I3_RBK_BORDER_COLOR_INT_OPAQUE_BLACK:
        return VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    case I3_RBK_BORDER_COLOR_FLOAT_OPAQUE_WHITE:
        return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    case I3_RBK_BORDER_COLOR_INT_OPAQUE_WHITE:
        return VK_BORDER_COLOR_INT_OPAQUE_WHITE;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported border color: %d", color);
        return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    }
    }
}

// compare op
VkCompareOp i3_vk_convert_compare_op(i3_rbk_compare_op_t op)
{
    switch (op)
    {
    case I3_RBK_COMPARE_OP_NEVER:
        return VK_COMPARE_OP_NEVER;
    case I3_RBK_COMPARE_OP_LESS:
        return VK_COMPARE_OP_LESS;
    case I3_RBK_COMPARE_OP_EQUAL:
        return VK_COMPARE_OP_EQUAL;
    case I3_RBK_COMPARE_OP_LESS_OR_EQUAL:
        return VK_COMPARE_OP_LESS_OR_EQUAL;
    case I3_RBK_COMPARE_OP_GREATER:
        return VK_COMPARE_OP_GREATER;
    case I3_RBK_COMPARE_OP_NOT_EQUAL:
        return VK_COMPARE_OP_NOT_EQUAL;
    case I3_RBK_COMPARE_OP_GREATER_OR_EQUAL:
        return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case I3_RBK_COMPARE_OP_ALWAYS:
        return VK_COMPARE_OP_ALWAYS;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported compare op: %d", op);
        return VK_COMPARE_OP_NEVER;
    }
    }
}

// format
VkFormat i3_vk_convert_format(i3_rbk_format_t format)
{
    switch (format)
    {

    case I3_RBK_FORMAT_R8_UNORM:
        return VK_FORMAT_R8_UNORM;
    case I3_RBK_FORMAT_R16_UNORM:
        return VK_FORMAT_R16_UNORM;
    case I3_RBK_FORMAT_R32_SFLOAT:
        return VK_FORMAT_R32_SFLOAT;
    case I3_RBK_FORMAT_R8G8B8A8_UNORM:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case I3_RBK_FORMAT_A2R10G10B10_UNORM:
        return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
    case I3_RBK_FORMAT_R16G16_FLOAT:
        return VK_FORMAT_R16G16_SFLOAT;
    case I3_RBK_FORMAT_R16G16B16A16_FLOAT:
        return VK_FORMAT_R16G16B16A16_SFLOAT;

    case I3_RBK_FORMAT_R32G32B32_SFLOAT:
        return VK_FORMAT_R32G32B32_SFLOAT;
    case I3_RBK_FORMAT_R32G32B32A32_SFLOAT:
        return VK_FORMAT_R32G32B32A32_SFLOAT;

    case I3_RBK_FORMAT_D16_UNORM:
        return VK_FORMAT_D16_UNORM;
    case I3_RBK_FORMAT_D32_FLOAT:
        return VK_FORMAT_D32_SFLOAT;
    case I3_RBK_FORMAT_D24_UNORM_S8_UINT:
        return VK_FORMAT_D24_UNORM_S8_UINT;

    case I3_RBK_FORMAT_UNDEFINED:
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported format: %d", format);
        return VK_FORMAT_UNDEFINED;
    }
    }
}

bool i3_vk_is_depth_format(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
        return true;
    default:
        return false;
    }
}

bool i3_vk_is_srgb_format(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_R8_SRGB:
    case VK_FORMAT_R8G8_SRGB:
    case VK_FORMAT_R8G8B8_SRGB:
    case VK_FORMAT_B8G8R8_SRGB:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
        return true;
    default:
        return false;
    }
}


// image type
VkImageType i3_vk_convert_image_type(i3_rbk_image_type_t type)
{
    switch (type)
    {
    case I3_RBK_IMAGE_TYPE_1D:
        return VK_IMAGE_TYPE_1D;
    case I3_RBK_IMAGE_TYPE_2D:
        return VK_IMAGE_TYPE_2D;
    case I3_RBK_IMAGE_TYPE_3D:
        return VK_IMAGE_TYPE_3D;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported image type: %d", type);
        return VK_IMAGE_TYPE_2D;
    }
    }
}

// sample count
VkSampleCountFlagBits i3_vk_convert_sample_count(uint32_t count)
{
    switch (count)
    {
    case 1:
        return VK_SAMPLE_COUNT_1_BIT;
    case 2:
        return VK_SAMPLE_COUNT_2_BIT;
    case 4:
        return VK_SAMPLE_COUNT_4_BIT;
    case 8:
        return VK_SAMPLE_COUNT_8_BIT;
    case 16:
        return VK_SAMPLE_COUNT_16_BIT;
    case 32:
        return VK_SAMPLE_COUNT_32_BIT;
    case 64:
        return VK_SAMPLE_COUNT_64_BIT;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported sample count: %d", count);
        return VK_SAMPLE_COUNT_1_BIT;
    }
    }
}

// view type
VkImageViewType i3_vk_convert_image_view_type(i3_rbk_image_view_type_t type)
{
    switch (type)
    {
    case I3_RBK_IMAGE_VIEW_TYPE_1D:
        return VK_IMAGE_VIEW_TYPE_1D;
    case I3_RBK_IMAGE_VIEW_TYPE_2D:
        return VK_IMAGE_VIEW_TYPE_2D;
    case I3_RBK_IMAGE_VIEW_TYPE_3D:
        return VK_IMAGE_VIEW_TYPE_3D;
    case I3_RBK_IMAGE_VIEW_TYPE_CUBE:
        return VK_IMAGE_VIEW_TYPE_CUBE;
    case I3_RBK_IMAGE_VIEW_TYPE_1D_ARRAY:
        return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case I3_RBK_IMAGE_VIEW_TYPE_2D_ARRAY:
        return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case I3_RBK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
        return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported image view type: %d", type);
        return VK_IMAGE_VIEW_TYPE_2D;
    }
    }
}

// component swizzle
VkComponentSwizzle i3_vk_convert_component_swizzle(i3_rbk_component_swizzle_t swizzle)
{
    switch (swizzle)
    {
    case I3_RBK_COMPONENT_SWIZZLE_IDENTITY:
        return VK_COMPONENT_SWIZZLE_IDENTITY;
    case I3_RBK_COMPONENT_SWIZZLE_ZERO:
        return VK_COMPONENT_SWIZZLE_ZERO;
    case I3_RBK_COMPONENT_SWIZZLE_ONE:
        return VK_COMPONENT_SWIZZLE_ONE;
    case I3_RBK_COMPONENT_SWIZZLE_R:
        return VK_COMPONENT_SWIZZLE_R;
    case I3_RBK_COMPONENT_SWIZZLE_G:
        return VK_COMPONENT_SWIZZLE_G;
    case I3_RBK_COMPONENT_SWIZZLE_B:
        return VK_COMPONENT_SWIZZLE_B;
    case I3_RBK_COMPONENT_SWIZZLE_A:
        return VK_COMPONENT_SWIZZLE_A;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported component swizzle: %d", swizzle);
        return VK_COMPONENT_SWIZZLE_IDENTITY;
    }
    }
}

// image aspect flags
VkImageAspectFlags i3_vk_convert_image_aspect_flags(i3_rbk_image_aspect_flags_t flags)
{
    VkImageAspectFlags res = 0;
    if (flags & I3_RBK_IMAGE_ASPECT_COLOR)
        res |= VK_IMAGE_ASPECT_COLOR_BIT;
    if (flags & I3_RBK_IMAGE_ASPECT_DEPTH)
        res |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (flags & I3_RBK_IMAGE_ASPECT_STENCIL)
        res |= VK_IMAGE_ASPECT_STENCIL_BIT;

    return res;
}

// shader stage
VkShaderStageFlagBits i3_vk_convert_shader_stage(i3_rbk_shader_stage_flag_bits_t stage)
{
    switch (stage)
    {
    case I3_RBK_SHADER_STAGE_VERTEX:
        return VK_SHADER_STAGE_VERTEX_BIT;
    case I3_RBK_SHADER_STAGE_TESSELLATION_CONTROL:
        return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    case I3_RBK_SHADER_STAGE_TESSELLATION_EVALUATION:
        return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    case I3_RBK_SHADER_STAGE_GEOMETRY:
        return VK_SHADER_STAGE_GEOMETRY_BIT;
    case I3_RBK_SHADER_STAGE_FRAGMENT:
        return VK_SHADER_STAGE_FRAGMENT_BIT;
    case I3_RBK_SHADER_STAGE_COMPUTE:
        return VK_SHADER_STAGE_COMPUTE_BIT;
    case I3_RBK_SHADER_STAGE_RAYGEN:
        return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    case I3_RBK_SHADER_STAGE_ANY_HIT:
        return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    case I3_RBK_SHADER_STAGE_CLOSEST_HIT:
        return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    case I3_RBK_SHADER_STAGE_MISS:
        return VK_SHADER_STAGE_MISS_BIT_KHR;
    case I3_RBK_SHADER_STAGE_INTERSECTION:
        return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    case I3_RBK_SHADER_STAGE_CALLABLE:
        return VK_SHADER_STAGE_CALLABLE_BIT_KHR;
    case I3_RBK_SHADER_STAGE_TASK:
        return VK_SHADER_STAGE_TASK_BIT_NV;
    case I3_RBK_SHADER_STAGE_MESH:
        return VK_SHADER_STAGE_MESH_BIT_NV;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported shader stage: %d", stage);
        return VK_SHADER_STAGE_VERTEX_BIT;
    }
    }
}

// vertex input rate
VkVertexInputRate i3_vk_convert_vertex_input_rate(i3_rbk_vertex_input_rate_t rate)
{
    switch (rate)
    {
    case I3_RBK_VERTEX_INPUT_RATE_VERTEX:
        return VK_VERTEX_INPUT_RATE_VERTEX;
    case I3_RBK_VERTEX_INPUT_RATE_INSTANCE:
        return VK_VERTEX_INPUT_RATE_INSTANCE;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported vertex input rate: %d", rate);
        return VK_VERTEX_INPUT_RATE_VERTEX;
    }
    }
}

VkPrimitiveTopology i3_vk_convert_primitive_topology(i3_rbk_primitive_topology_t topology)
{
    switch (topology)
    {
    case I3_RBK_PRIMITIVE_TOPOLOGY_POINT_LIST:
        return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case I3_RBK_PRIMITIVE_TOPOLOGY_LINE_LIST:
        return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case I3_RBK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
        return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case I3_RBK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case I3_RBK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case I3_RBK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case I3_RBK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
        return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
    case I3_RBK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
        return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    case I3_RBK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
    case I3_RBK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
    case I3_RBK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
        return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    default:
    {
        i3_logger_i* logger = i3_vk_get_logger();
        i3_log_wrn(logger, "Unsupported primitive topology: %d", topology);
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
    }
}