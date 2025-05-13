#pragma once

#include "native/core/common.h"
#include "native/render_window/render_window.h"

// backend
typedef struct i3_render_backend_i i3_render_backend_i;
typedef struct i3_render_backend_o i3_render_backend_o;

// device
typedef struct i3_rbk_device_i i3_rbk_device_i;
typedef struct i3_rbk_device_o i3_rbk_device_o;

// flags
typedef uint32_t i3_rbk_flags_t;

// filter
typedef enum
{
    I3_RBK_FILTER_NEAREST = 0,
    I3_RBK_FILTER_LINEAR,
} i3_rbk_filter_t;

// sampler mipmap mode
typedef enum
{
    I3_RBK_SAMPLER_MIPMAP_MODE_NEAREST = 0,
    I3_RBK_SAMPLER_MIPMAP_MODE_LINEAR,
} i3_rbk_sampler_mipmap_mode_t;

// sampler address mode
typedef enum
{
    I3_RBK_SAMPLER_ADDRESS_MODE_REPEAT = 0,
    I3_RBK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    I3_RBK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    I3_RBK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
} i3_rbk_sampler_address_mode_t;

// border color
typedef enum
{
    I3_RBK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK = 0,
    I3_RBK_BORDER_COLOR_INT_TRANSPARENT_BLACK,
    I3_RBK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
    I3_RBK_BORDER_COLOR_INT_OPAQUE_BLACK,
    I3_RBK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    I3_RBK_BORDER_COLOR_INT_OPAQUE_WHITE,
} i3_rbk_border_color_t;

// compare op
typedef enum
{
    I3_RBK_COMPARE_OP_NEVER = 0,
    I3_RBK_COMPARE_OP_LESS,
    I3_RBK_COMPARE_OP_EQUAL,
    I3_RBK_COMPARE_OP_LESS_OR_EQUAL,
    I3_RBK_COMPARE_OP_GREATER,
    I3_RBK_COMPARE_OP_NOT_EQUAL,
    I3_RBK_COMPARE_OP_GREATER_OR_EQUAL,
    I3_RBK_COMPARE_OP_ALWAYS,
} i3_rbk_compare_op_t;

// buffer flags
typedef enum
{
    I3_RBK_BUFFER_FLAG_NONE = 0,
    I3_RBK_BUFFER_FLAG_VERTEX_BUFFER = i3_flag(0),
    I3_RBK_BUFFER_FLAG_INDEX_BUFFER = i3_flag(1),
    I3_RBK_BUFFER_FLAG_INDIRECT_BUFFER = i3_flag(2),
    I3_RBK_BUFFER_FLAG_UNIFORM_BUFFER = i3_flag(3),
    I3_RBK_BUFFER_FLAG_STORAGE_BUFFER = i3_flag(4),
    I3_RBK_BUFFER_FLAG_STAGING = i3_flag(5),
} i3_rbk_buffer_flag_bits_t;

typedef i3_rbk_flags_t i3_rbk_buffer_flags_t;

// image types
typedef enum
{
    I3_RBK_IMAGE_TYPE_1D,
    I3_RBK_IMAGE_TYPE_2D,
    I3_RBK_IMAGE_TYPE_3D,
} i3_rbk_image_type_t;

// image formats
typedef enum
{
    I3_RBK_FORMAT_UNDEFINED = 0,

    // color formats
    I3_RBK_FORMAT_R8_UNORM,
    I3_RBK_FORMAT_R16_UNORM,
    I3_RBK_FORMAT_R32_SFLOAT,
    I3_RBK_FORMAT_R8G8B8A8_UNORM,
    I3_RBK_FORMAT_A2R10G10B10_UNORM,
    I3_RBK_FORMAT_R16G16_FLOAT,
    I3_RBK_FORMAT_R16G16B16A16_FLOAT,
    I3_RBK_FORMAT_R32G32B32_SFLOAT,
    I3_RBK_FORMAT_R32G32B32A32_SFLOAT,

    // depth formats
    I3_RBK_FORMAT_D16_UNORM,
    I3_RBK_FORMAT_D32_FLOAT,
    I3_RBK_FORMAT_D24_UNORM_S8_UINT,

} i3_rbk_format_t;

// image flags
typedef enum
{
    I3_RBK_IMAGE_FLAG_NONE = 0,
} i3_rbk_image_flag_bits_t;

typedef i3_rbk_flags_t i3_rbk_image_flags_t;

// image view types
typedef enum
{
    I3_RBK_IMAGE_VIEW_TYPE_UNDEFINED = 0,
    I3_RBK_IMAGE_VIEW_TYPE_1D,
    I3_RBK_IMAGE_VIEW_TYPE_2D,
    I3_RBK_IMAGE_VIEW_TYPE_3D,
    I3_RBK_IMAGE_VIEW_TYPE_CUBE,
    I3_RBK_IMAGE_VIEW_TYPE_1D_ARRAY,
    I3_RBK_IMAGE_VIEW_TYPE_2D_ARRAY,
    I3_RBK_IMAGE_VIEW_TYPE_CUBE_ARRAY,
} i3_rbk_image_view_type_t;

// component swizzle
typedef enum
{
    I3_RBK_COMPONENT_SWIZZLE_IDENTITY = 0,
    I3_RBK_COMPONENT_SWIZZLE_ZERO,
    I3_RBK_COMPONENT_SWIZZLE_ONE,
    I3_RBK_COMPONENT_SWIZZLE_R,
    I3_RBK_COMPONENT_SWIZZLE_G,
    I3_RBK_COMPONENT_SWIZZLE_B,
    I3_RBK_COMPONENT_SWIZZLE_A,
} i3_rbk_component_swizzle_t;

// aspect flags
typedef enum
{
    I3_RBK_IMAGE_ASPECT_COLOR = i3_flag(0),
    I3_RBK_IMAGE_ASPECT_DEPTH = i3_flag(1),
    I3_RBK_IMAGE_ASPECT_STENCIL = i3_flag(2),
} i3_rbk_image_aspect_flag_bits_t;

typedef i3_rbk_flags_t i3_rbk_image_aspect_flags_t;

// shader stage
typedef enum
{
    I3_RBK_SHADER_STAGE_VERTEX = i3_flag(0),
    I3_RBK_SHADER_STAGE_TESSELLATION_CONTROL = i3_flag(1),
    I3_RBK_SHADER_STAGE_TESSELLATION_EVALUATION = i3_flag(2),
    I3_RBK_SHADER_STAGE_GEOMETRY = i3_flag(3),
    I3_RBK_SHADER_STAGE_FRAGMENT = i3_flag(4),
    I3_RBK_SHADER_STAGE_COMPUTE = i3_flag(5),
    I3_RBK_SHADER_STAGE_RAYGEN = i3_flag(6),
    I3_RBK_SHADER_STAGE_ANY_HIT = i3_flag(7),
    I3_RBK_SHADER_STAGE_CLOSEST_HIT = i3_flag(8),
    I3_RBK_SHADER_STAGE_MISS = i3_flag(9),
    I3_RBK_SHADER_STAGE_INTERSECTION = i3_flag(10),
    I3_RBK_SHADER_STAGE_CALLABLE = i3_flag(11),
    I3_RBK_SHADER_STAGE_TASK = i3_flag(12),
    I3_RBK_SHADER_STAGE_MESH = i3_flag(13),
} i3_rbk_shader_stage_flag_bits_t;

typedef i3_rbk_flags_t i3_rbk_shader_stage_flags_t;

// vertex input rate
typedef enum
{
    I3_RBK_VERTEX_INPUT_RATE_VERTEX = 0,
    I3_RBK_VERTEX_INPUT_RATE_INSTANCE,
} i3_rbk_vertex_input_rate_t;

// primitive topology
typedef enum
{
    I3_RBK_PRIMITIVE_TOPOLOGY_POINT_LIST = 0,
    I3_RBK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    I3_RBK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
    I3_RBK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    I3_RBK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
    I3_RBK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
    I3_RBK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
    I3_RBK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,
    I3_RBK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
    I3_RBK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY,
    I3_RBK_PRIMITIVE_TOPOLOGY_PATCH_LIST,
} i3_rbk_primitive_topology_t;

// polygon mode
typedef enum
{
    I3_RBK_POLYGON_MODE_FILL = 0,
    I3_RBK_POLYGON_MODE_LINE,
    I3_RBK_POLYGON_MODE_POINT,
} i3_rbk_polygon_mode_t;

// cull mode
typedef enum
{
    I3_RBK_CULL_MODE_FRONT_BIT = i3_flag(0),
    I3_RBK_CULL_MODE_BACK_BIT = i3_flag(1),
    I3_RBK_CULL_MODE_FRONT_AND_BACK = I3_RBK_CULL_MODE_FRONT_BIT | I3_RBK_CULL_MODE_BACK_BIT,
} i3_rbk_cull_mode_flag_bits_t;

typedef i3_rbk_flags_t i3_rbk_cull_mode_flags_t;

// stencil op
typedef enum
{
    I3_RBK_STENCIL_OP_KEEP = 0,
    I3_RBK_STENCIL_OP_ZERO,
    I3_RBK_STENCIL_OP_REPLACE,
    I3_RBK_STENCIL_OP_INCREMENT_AND_CLAMP,
    I3_RBK_STENCIL_OP_DECREMENT_AND_CLAMP,
    I3_RBK_STENCIL_OP_INVERT,
    I3_RBK_STENCIL_OP_INCREMENT_AND_WRAP,
    I3_RBK_STENCIL_OP_DECREMENT_AND_WRAP,
} i3_rbk_stencil_op_t;

// front face
typedef enum
{
    I3_RBK_FRONT_FACE_COUNTER_CLOCKWISE = 0,
    I3_RBK_FRONT_FACE_CLOCKWISE,
} i3_rbk_front_face_t;

// logic op
typedef enum
{
    I3_RBK_LOGIC_OP_CLEAR = 0,
    I3_RBK_LOGIC_OP_AND,
    I3_RBK_LOGIC_OP_AND_REVERSE,
    I3_RBK_LOGIC_OP_COPY,
    I3_RBK_LOGIC_OP_AND_INVERTED,
    I3_RBK_LOGIC_OP_NO_OP,
    I3_RBK_LOGIC_OP_XOR,
    I3_RBK_LOGIC_OP_OR,
    I3_RBK_LOGIC_OP_NOR,
    I3_RBK_LOGIC_OP_EQUIVALENT,
    I3_RBK_LOGIC_OP_INVERT,
    I3_RBK_LOGIC_OP_OR_REVERSE,
    I3_RBK_LOGIC_OP_COPY_INVERTED,
    I3_RBK_LOGIC_OP_OR_INVERTED,
    I3_RBK_LOGIC_OP_NAND,
    I3_RBK_LOGIC_OP_SET,
} i3_rbk_logic_op_t;

// blend factor
typedef enum
{
    I3_RBK_BLEND_FACTOR_ZERO = 0,
    I3_RBK_BLEND_FACTOR_ONE,
    I3_RBK_BLEND_FACTOR_SRC_COLOR,
    I3_RBK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
    I3_RBK_BLEND_FACTOR_DST_COLOR,
    I3_RBK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
    I3_RBK_BLEND_FACTOR_SRC_ALPHA,
    I3_RBK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
    I3_RBK_BLEND_FACTOR_DST_ALPHA,
    I3_RBK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
    I3_RBK_BLEND_FACTOR_CONSTANT_COLOR,
    I3_RBK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
    I3_RBK_BLEND_FACTOR_CONSTANT_ALPHA,
    I3_RBK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
    I3_RBK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
    I3_RBK_BLEND_FACTOR_SRC1_COLOR,
    I3_RBK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
    I3_RBK_BLEND_FACTOR_SRC1_ALPHA,
    I3_RBK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
} i3_rbk_blend_factor_t;

// blend op
typedef enum
{
    I3_RBK_BLEND_OP_ADD = 0,
    I3_RBK_BLEND_OP_SUBTRACT,
    I3_RBK_BLEND_OP_REVERSE_SUBTRACT,
    I3_RBK_BLEND_OP_MIN,
    I3_RBK_BLEND_OP_MAX,
} i3_rbk_blend_op_t;

// color component flags
typedef enum
{
    I3_RBK_COLOR_COMPONENT_R_BIT = i3_flag(0),
    I3_RBK_COLOR_COMPONENT_G_BIT = i3_flag(1),
    I3_RBK_COLOR_COMPONENT_B_BIT = i3_flag(2),
    I3_RBK_COLOR_COMPONENT_A_BIT = i3_flag(3),
} i3_rbk_color_component_flag_bits_t;

typedef i3_rbk_flags_t i3_rbk_color_component_flags_t;

// dynamic state
typedef enum
{
    I3_RBK_DYNAMIC_STATE_VIEWPORT = 0,
    I3_RBK_DYNAMIC_STATE_SCISSOR,
    I3_RBK_DYNAMIC_STATE_LINE_WIDTH,
    I3_RBK_DYNAMIC_STATE_DEPTH_BIAS,
    I3_RBK_DYNAMIC_STATE_BLEND_CONSTANTS,
    I3_RBK_DYNAMIC_STATE_DEPTH_BOUNDS,
    I3_RBK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
    I3_RBK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
    I3_RBK_DYNAMIC_STATE_STENCIL_REFERENCE,
} i3_rbk_dynamic_state_t;

// descriptor type
typedef enum
{
    I3_RBK_DESCRIPTOR_TYPE_SAMPLER = 0,
    I3_RBK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    I3_RBK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    I3_RBK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    I3_RBK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
    I3_RBK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
    I3_RBK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    I3_RBK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    I3_RBK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
    I3_RBK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
    I3_RBK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
} i3_rbk_descriptor_type_t;

// index type
typedef enum
{
    I3_RBK_INDEX_TYPE_UINT16 = 0,
    I3_RBK_INDEX_TYPE_UINT32,
} i3_rbk_index_type_t;

// viewport
typedef struct i3_rbk_viewport_t
{
    float x;
    float y;
    float width;
    float height;
    float min_depth;
    float max_depth;
} i3_rbk_viewport_t;

// extents
typedef struct i3_rbk_extent2d_t
{
    uint32_t width;
    uint32_t height;
} i3_rbk_extent2d_t;

typedef struct i3_rbk_extent3d_t
{
    uint32_t width;
    uint32_t height;
    uint32_t depth;
} i3_rbk_extent3d_t;

// offsets
typedef struct i3_rbk_offset2d_t
{
    int32_t x;
    int32_t y;
} i3_rbk_offset2d_t;

typedef struct i3_rbk_offset3d_t
{
    int32_t x;
    int32_t y;
    int32_t z;
} i3_rbk_offset3d_t;

// rect
typedef struct i3_rbk_rect_t
{
    i3_rbk_offset2d_t offset;
    i3_rbk_extent2d_t extent;
} i3_rbk_rect_t;

// clear color
typedef union i3_rbk_clear_color_t
{
    float float32[4];
    int32_t int32[4];
    uint32_t uint32[4];
} i3_rbk_clear_color_t;

// resource interface
typedef struct i3_rbk_resource_o i3_rbk_resource_o;

typedef struct i3_rbk_resource_i
{
    i3_rbk_resource_o* self;

    void (*add_ref)(i3_rbk_resource_o* self);
    void (*release)(i3_rbk_resource_o* self);
    int32_t (*get_use_count)(i3_rbk_resource_o* self);
    void (*set_debug_name)(i3_rbk_resource_o* self, const char* name);

} i3_rbk_resource_i;

#define i3_rbk_resource_add_ref(resource)                                      \
    {                                                                          \
        i3_rbk_resource_i* res__ = (resource)->get_resource((resource)->self); \
        res__->add_ref(res__->self);                                           \
    }                                                                          \
    while (0)

#define i3_rbk_resource_release(resource)                                      \
    {                                                                          \
        i3_rbk_resource_i* res__ = (resource)->get_resource((resource)->self); \
        res__->release(res__->self);                                           \
    }                                                                          \
    while (0)

#define i3_rbk_resource_get_use_count(resource) \
    ((resource)->get_resource((resource)->self)->get_use_count((resource)->get_resource((resource)->self)->self))

#define i3_rbk_resource_set_debug_name(resource, name)                         \
    {                                                                          \
        i3_rbk_resource_i* res__ = (resource)->get_resource((resource)->self); \
        res__->set_debug_name(res__->self, name);                              \
    }                                                                          \
    while (0)

// sampler
typedef struct i3_rbk_sampler_desc_t
{
    i3_rbk_filter_t mag_filter;
    i3_rbk_filter_t min_filter;
    i3_rbk_sampler_mipmap_mode_t mipmap_mode;
    i3_rbk_sampler_address_mode_t address_mode_u;
    i3_rbk_sampler_address_mode_t address_mode_v;
    i3_rbk_sampler_address_mode_t address_mode_w;
    float mip_lod_bias;
    bool anisotropy_enable;
    float max_anisotropy;
    bool compare_enable;
    i3_rbk_compare_op_t compare_op;
    float min_lod;
    float max_lod;
    i3_rbk_border_color_t border_color;
    bool unnormalized_coordinates;
} i3_rbk_sampler_desc_t;

typedef struct i3_rbk_sampler_o i3_rbk_sampler_o;

typedef struct i3_rbk_sampler_i
{
    i3_rbk_sampler_o* self;

    const i3_rbk_sampler_desc_t* (*get_desc)(i3_rbk_sampler_o* self);
    i3_rbk_resource_i* (*get_resource)(i3_rbk_sampler_o* self);
    void (*destroy)(i3_rbk_sampler_o* self);

} i3_rbk_sampler_i;

// buffer
typedef struct i3_rbk_buffer_desc_t
{
    i3_rbk_buffer_flags_t flags;
    uint32_t size;
} i3_rbk_buffer_desc_t;

typedef struct i3_rbk_buffer_o i3_rbk_buffer_o;

typedef struct i3_rbk_buffer_i
{
    i3_rbk_buffer_o* self;

    const i3_rbk_buffer_desc_t* (*get_desc)(i3_rbk_buffer_o* self);
    i3_rbk_resource_i* (*get_resource)(i3_rbk_buffer_o* self);

    // map/unmap staging buffer
    void* (*map)(i3_rbk_buffer_o* self);
    void (*unmap)(i3_rbk_buffer_o* self);

    void (*destroy)(i3_rbk_buffer_o* self);
} i3_rbk_buffer_i;

// image
typedef struct i3_rbk_image_desc_t
{
    i3_rbk_image_flags_t flags;
    i3_rbk_image_type_t type;
    i3_rbk_format_t format;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t mip_levels;
    uint32_t array_layers;
    uint32_t samples;
} i3_rbk_image_desc_t;

typedef struct i3_rbk_image_o i3_rbk_image_o;

typedef struct i3_rbk_image_i
{
    i3_rbk_image_o* self;

    const i3_rbk_image_desc_t* (*get_desc)(i3_rbk_image_o* self);
    i3_rbk_resource_i* (*get_resource)(i3_rbk_image_o* self);
    void (*destroy)(i3_rbk_image_o* self);
} i3_rbk_image_i;

// image view
typedef struct i3_rbk_image_view_desc_t
{
    i3_rbk_image_view_type_t type;
    i3_rbk_format_t format;
    i3_rbk_component_swizzle_t r, g, b, a;
    i3_rbk_image_aspect_flags_t aspect_mask;
    uint32_t base_mip_level;
    uint32_t level_count;
    uint32_t base_array_layer;
    uint32_t layer_count;
} i3_rbk_image_view_desc_t;

typedef struct i3_rbk_image_view_o i3_rbk_image_view_o;

typedef struct i3_rbk_image_view_i
{
    i3_rbk_image_view_o* self;

    const i3_rbk_image_view_desc_t* (*get_desc)(i3_rbk_image_view_o* self);
    i3_rbk_image_i* (*get_image)(i3_rbk_image_view_o* self);
    i3_rbk_resource_i* (*get_resource)(i3_rbk_image_view_o* self);
    void (*destroy)(i3_rbk_image_view_o* self);
} i3_rbk_image_view_i;

// descriptor set layout
typedef struct i3_rbk_descriptor_set_layout_binding_t
{
    uint32_t binding;
    i3_rbk_descriptor_type_t descriptor_type;
    uint32_t descriptor_count;
    i3_rbk_shader_stage_flags_t stage_flags;
    i3_rbk_sampler_i* immutable_samplers;
} i3_rbk_descriptor_set_layout_binding_t;

typedef struct i3_rbk_descriptor_set_layout_desc_t
{
    uint32_t binding_count;
    const i3_rbk_descriptor_set_layout_binding_t* bindings;
} i3_rbk_descriptor_set_layout_desc_t;

typedef struct i3_rbk_descriptor_set_layout_o i3_rbk_descriptor_set_layout_o;

typedef struct i3_rbk_descriptor_set_layout_i
{
    i3_rbk_descriptor_set_layout_o* self;
    i3_rbk_resource_i* (*get_resource)(i3_rbk_descriptor_set_layout_o* self);
    void (*destroy)(i3_rbk_descriptor_set_layout_o* self);
} i3_rbk_descriptor_set_layout_i;

// pipeline layout
typedef struct i3_rbk_push_constant_range_t
{
    i3_rbk_shader_stage_flags_t stage_flags;
    uint32_t offset;
    uint32_t size;
} i3_rbk_push_constant_range_t;

typedef struct i3_rbk_pipeline_layout_desc_t
{
    uint32_t set_layout_count;
    const i3_rbk_descriptor_set_layout_i** set_layouts;
    uint32_t push_constant_range_count;
    const i3_rbk_push_constant_range_t* push_constant_ranges;
} i3_rbk_pipeline_layout_desc_t;

typedef struct i3_rbk_pipeline_layout_o i3_rbk_pipeline_layout_o;

typedef struct i3_rbk_pipeline_layout_i
{
    i3_rbk_pipeline_layout_o* self;
    i3_rbk_resource_i* (*get_resource)(i3_rbk_pipeline_layout_o* self);
    void (*destroy)(i3_rbk_pipeline_layout_o* self);
} i3_rbk_pipeline_layout_i;

// framebuffer
typedef struct i3_rbk_framebuffer_attachment_desc_t
{
    i3_rbk_image_view_i* image_view;
} i3_rbk_framebuffer_attachment_desc_t;

typedef struct i3_rbk_framebuffer_desc_t
{
    uint32_t width;
    uint32_t height;
    uint32_t layers;
    uint32_t color_attachment_count;
    const i3_rbk_framebuffer_attachment_desc_t* color_attachments;
    i3_rbk_framebuffer_attachment_desc_t* depth_attachment;
} i3_rbk_framebuffer_desc_t;

typedef struct i3_rbk_framebuffer_o i3_rbk_framebuffer_o;

typedef struct i3_rbk_framebuffer_i
{
    i3_rbk_framebuffer_o* self;
    i3_rbk_resource_i* (*get_resource)(i3_rbk_framebuffer_o* self);
    void (*destroy)(i3_rbk_framebuffer_o* self);
} i3_rbk_framebuffer_i;

// shader module
typedef struct i3_rbk_shader_module_desc_t
{
    const void* code;
    uint32_t code_size;
} i3_rbk_shader_module_desc_t;

typedef struct i3_rbk_shader_module_o i3_rbk_shader_module_o;

typedef struct i3_rbk_shader_module_i
{
    i3_rbk_shader_module_o* self;

    const i3_rbk_shader_module_desc_t* (*get_desc)(i3_rbk_shader_module_o* self);
    i3_rbk_resource_i* (*get_resource)(i3_rbk_shader_module_o* self);
    void (*destroy)(i3_rbk_shader_module_o* self);
} i3_rbk_shader_module_i;

// shader stage
typedef struct i3_rbk_pipeline_shader_stage_desc_t
{
    i3_rbk_shader_stage_flag_bits_t stage;
    i3_rbk_shader_module_i* shader_module;
    const char* entry_point;
} i3_rbk_pipeline_shader_stage_desc_t;

// vertex input
typedef struct i3_rbk_pipeline_vertex_input_binding_desc_t
{
    uint32_t binding;
    uint32_t stride;
    i3_rbk_vertex_input_rate_t input_rate;
} i3_rbk_pipeline_vertex_input_binding_desc_t;

typedef struct i3_rbk_pipeline_vertex_input_attribute_desc_t
{
    uint32_t location;
    uint32_t binding;
    i3_rbk_format_t format;
    uint32_t offset;
} i3_rbk_pipeline_vertex_input_attribute_desc_t;

typedef struct i3_rbk_pipeline_vertex_input_state_t
{
    const i3_rbk_pipeline_vertex_input_binding_desc_t* bindings;
    uint32_t binding_count;
    const i3_rbk_pipeline_vertex_input_attribute_desc_t* attributes;
    uint32_t attribute_count;
} i3_rbk_pipeline_vertex_input_state_t;

// input assembly
typedef struct i3_rbk_pipeline_input_assembly_state_t
{
    i3_rbk_primitive_topology_t topology;
    bool primitive_restart_enable;
} i3_rbk_pipeline_input_assembly_state_t;

// tessellation
typedef struct i3_rbk_pipeline_tessellation_state_t
{
    int path_control_points;
} i3_rbk_pipeline_tessellation_state_t;

// viewport
typedef struct i3_rbk_pipeline_viewport_state_t
{
    uint32_t viewport_count;
    const i3_rbk_viewport_t* viewports;
    uint32_t scissor_count;
    const i3_rbk_rect_t* scissors;
} i3_rbk_pipeline_viewport_state_t;

// rasterization
typedef struct i3_rbk_pipeline_rasterization_state_t
{
    bool depth_clamp_enable;
    bool rasterizer_discard_enable;
    i3_rbk_polygon_mode_t polygon_mode;
    i3_rbk_cull_mode_flags_t cull_mode;
    i3_rbk_front_face_t front_face;
    bool depth_bias_enable;
    float depth_bias_constant_factor;
    float depth_bias_clamp;
    float depth_bias_slope_factor;
    float line_width;
} i3_rbk_pipeline_rasterization_state_t;

// multisample
typedef struct i3_rbk_pipeline_multisample_state_t
{
    uint32_t rasterization_samples;
    bool sample_shading_enable;
    float min_sample_shading;
    const uint32_t* sample_mask;
    bool alpha_to_coverage_enable;
    bool alpha_to_one_enable;
} i3_rbk_pipeline_multisample_state_t;

// depth stencil
typedef struct i3_rbk_stencil_op_state_t
{
    i3_rbk_stencil_op_t fail_op;
    i3_rbk_stencil_op_t pass_op;
    i3_rbk_stencil_op_t depth_fail_op;
    i3_rbk_compare_op_t compare_op;
    uint32_t compare_mask;
    uint32_t write_mask;
    uint32_t reference;
} i3_rbk_stencil_op_state_t;

typedef struct i3_rbk_pipeline_depth_stencil_state_t
{
    bool depth_test_enable;
    bool depth_write_enable;
    i3_rbk_compare_op_t depth_compare_op;
    bool depth_bounds_test_enable;
    bool stencil_test_enable;
    i3_rbk_stencil_op_state_t front;
    i3_rbk_stencil_op_state_t back;
    float min_depth_bounds;
    float max_depth_bounds;
} i3_rbk_pipeline_depth_stencil_state_t;

// color blend
typedef struct i3_rbk_pipeline_color_blend_attachment_state_t
{
    bool blend_enable;
    i3_rbk_blend_factor_t src_color_blend_factor;
    i3_rbk_blend_factor_t dst_color_blend_factor;
    i3_rbk_blend_op_t color_blend_op;
    i3_rbk_blend_factor_t src_alpha_blend_factor;
    i3_rbk_blend_factor_t dst_alpha_blend_factor;
    i3_rbk_blend_op_t alpha_blend_op;
    i3_rbk_color_component_flags_t color_write_mask;
} i3_rbk_pipeline_color_blend_attachment_state_t;

typedef struct i3_rbk_pipeline_color_blend_state_t
{
    bool logic_op_enable;
    i3_rbk_logic_op_t logic_op;
    uint32_t attachment_count;
    const i3_rbk_pipeline_color_blend_attachment_state_t* attachments;
    float blend_constants[4];
} i3_rbk_pipeline_color_blend_state_t;

// dynamic state
typedef struct i3_rbk_pipeline_dynamic_state_t
{
    uint32_t dynamic_state_count;
    const i3_rbk_dynamic_state_t* dynamic_states;
} i3_rbk_pipeline_dynamic_state_t;

// graphics pipeline
typedef struct i3_rbk_graphics_pipeline_desc_t
{
    uint32_t stage_count;
    const i3_rbk_pipeline_shader_stage_desc_t* stages;
    const i3_rbk_pipeline_vertex_input_state_t* vertex_input;
    const i3_rbk_pipeline_input_assembly_state_t* input_assembly;
    const i3_rbk_pipeline_tessellation_state_t* tessellation;
    const i3_rbk_pipeline_viewport_state_t* viewport;
    const i3_rbk_pipeline_rasterization_state_t* rasterization;
    const i3_rbk_pipeline_multisample_state_t* multisample;
    const i3_rbk_pipeline_depth_stencil_state_t* depth_stencil;
    const i3_rbk_pipeline_color_blend_state_t* color_blend;
    const i3_rbk_pipeline_dynamic_state_t* dynamic;
    i3_rbk_framebuffer_i* framebuffer;
    i3_rbk_pipeline_layout_i* layout;
} i3_rbk_graphics_pipeline_desc_t;

// compute pipeline
typedef struct i3_rbk_compute_pipeline_desc_t
{
    i3_rbk_pipeline_shader_stage_desc_t stage;
    i3_rbk_pipeline_layout_i* layout;
} i3_rbk_compute_pipeline_desc_t;

// pipeline interface
typedef struct i3_rbk_pipeline_o i3_rbk_pipeline_o;

typedef struct i3_rbk_pipeline_i
{
    i3_rbk_pipeline_o* self;

    i3_rbk_resource_i* (*get_resource)(i3_rbk_pipeline_o* self);
    void (*destroy)(i3_rbk_pipeline_o* self);
} i3_rbk_pipeline_i;

// swapchain
typedef struct i3_rbk_swapchain_desc_t
{
    uint32_t requested_image_count;
    bool srgb;
    bool vsync;
} i3_rbk_swapchain_desc_t;

typedef struct i3_rbk_swapchain_o i3_rbk_swapchain_o;

typedef struct i3_rbk_swapchain_i
{
    i3_rbk_swapchain_o* self;

    const i3_rbk_swapchain_desc_t* (*get_desc)(i3_rbk_swapchain_o* self);
    i3_rbk_resource_i* (*get_resource)(i3_rbk_swapchain_o* self);
    void (*destroy)(i3_rbk_swapchain_o* self);
} i3_rbk_swapchain_i;

// cmd buffer
typedef struct i3_rbk_cmd_buffer_o i3_rbk_cmd_buffer_o;

typedef struct i3_rbk_cmd_buffer_i
{
    i3_rbk_cmd_buffer_o* self;

    i3_rbk_resource_i* (*get_resource)(i3_rbk_cmd_buffer_o* self);

    void (*write_buffer)(i3_rbk_cmd_buffer_o* self,
                         i3_rbk_buffer_i* buffer,
                         uint32_t dst_offset,
                         uint32_t size,
                         const void* data);

    void (*copy_buffer)(i3_rbk_cmd_buffer_o* self,
                        i3_rbk_buffer_i* src_buffer,
                        i3_rbk_buffer_i* dst_buffer,
                        uint32_t src_offset,
                        uint32_t dst_offset,
                        uint32_t size);

    void (*clear_image)(i3_rbk_cmd_buffer_o* self, i3_rbk_image_i* image, const i3_rbk_clear_color_t* color);

    void (*bind_vertex_buffers)(i3_rbk_cmd_buffer_o* self,
                                uint32_t first_binding,
                                uint32_t binding_count,
                                i3_rbk_buffer_i** buffers,
                                const uint32_t* offsets);

    void (*bind_index_buffer)(i3_rbk_cmd_buffer_o* self,
                              i3_rbk_buffer_i* buffer,
                              uint32_t offset,
                              i3_rbk_index_type_t index_type);

    void (*bind_pipeline)(i3_rbk_cmd_buffer_o* self, i3_rbk_pipeline_i* pipeline);

    void (*set_viewports)(i3_rbk_cmd_buffer_o* self,
                          uint32_t first_viewport,
                          uint32_t viewport_count,
                          const i3_rbk_viewport_t* viewports);

    void (*set_scissors)(i3_rbk_cmd_buffer_o* self,
                         uint32_t first_scissor,
                         uint32_t scissor_count,
                         const i3_rbk_rect_t* scissors);

    void (*begin_rendering)(i3_rbk_cmd_buffer_o* self,
                            i3_rbk_framebuffer_i* framebuffer,
                            const i3_rbk_rect_t* render_area);

    void (*end_rendering)(i3_rbk_cmd_buffer_o* self);

    void (*push_constants)(i3_rbk_cmd_buffer_o* self,
                           i3_rbk_pipeline_layout_i* layout,
                           i3_rbk_shader_stage_flags_t stage_flags,
                           uint32_t offset,
                           uint32_t size,
                           const void* data);

    void (*draw)(i3_rbk_cmd_buffer_o* self,
                 uint32_t vertex_count,
                 uint32_t instance_count,
                 uint32_t first_vertex,
                 uint32_t first_instance);

    void (*draw_indexed)(i3_rbk_cmd_buffer_o* self,
                         uint32_t index_count,
                         uint32_t instance_count,
                         uint32_t first_index,
                         int32_t vertex_offset,
                         uint32_t first_instance);

    void (*draw_indirect)(i3_rbk_cmd_buffer_o* self,
                          i3_rbk_buffer_i* buffer,
                          uint32_t offset,
                          uint32_t draw_count,
                          uint32_t stride);

    void (*draw_indexed_indirect)(i3_rbk_cmd_buffer_o* self,
                                  i3_rbk_buffer_i* buffer,
                                  uint32_t offset,
                                  uint32_t draw_count,
                                  uint32_t stride);

    void (*destroy)(i3_rbk_cmd_buffer_o* self);
} i3_rbk_cmd_buffer_i;

// device description
typedef struct i3_rbk_device_desc_t
{
    const char* name;
} i3_rbk_device_desc_t;

// device interface
struct i3_rbk_device_i
{
    i3_rbk_device_o* self;

    // create sampler
    i3_rbk_sampler_i* (*create_sampler)(i3_rbk_device_o* self, const i3_rbk_sampler_desc_t* desc);

    // create buffer
    i3_rbk_buffer_i* (*create_buffer)(i3_rbk_device_o* self, const i3_rbk_buffer_desc_t* desc);

    // create image
    i3_rbk_image_i* (*create_image)(i3_rbk_device_o* self, const i3_rbk_image_desc_t* desc);

    // create image view
    i3_rbk_image_view_i* (*create_image_view)(i3_rbk_device_o* self,
                                              i3_rbk_image_i* image,
                                              const i3_rbk_image_view_desc_t* info);

    // create descriptor set layout
    i3_rbk_descriptor_set_layout_i* (*create_descriptor_set_layout)(i3_rbk_device_o* self,
                                                                    const i3_rbk_descriptor_set_layout_desc_t* desc);

    // create pipeline layout
    i3_rbk_pipeline_layout_i* (*create_pipeline_layout)(i3_rbk_device_o* self,
                                                        const i3_rbk_pipeline_layout_desc_t* desc);

    // create framebuffer
    i3_rbk_framebuffer_i* (*create_framebuffer)(i3_rbk_device_o* self, const i3_rbk_framebuffer_desc_t* desc);

    // create shader module
    i3_rbk_shader_module_i* (*create_shader_module)(i3_rbk_device_o* self, const i3_rbk_shader_module_desc_t* desc);

    // create graphics pipeline
    i3_rbk_pipeline_i* (*create_graphics_pipeline)(i3_rbk_device_o* self, const i3_rbk_graphics_pipeline_desc_t* desc);

    // create compute pipeline
    i3_rbk_pipeline_i* (*create_compute_pipeline)(i3_rbk_device_o* self, const i3_rbk_compute_pipeline_desc_t* desc);

    // create swapchain
    i3_rbk_swapchain_i* (*create_swapchain)(i3_rbk_device_o* self,
                                            i3_render_window_i* window,
                                            const i3_rbk_swapchain_desc_t* desc);

    // create cmb buffer
    i3_rbk_cmd_buffer_i* (*create_cmd_buffer)(i3_rbk_device_o* self);

    // submit cmd buffers
    void (*submit_cmd_buffers)(i3_rbk_device_o* self, i3_rbk_cmd_buffer_i** cmd_buffers, uint32_t cmd_buffer_count);

    // present swapchain
    void (*present)(i3_rbk_device_o* self, i3_rbk_swapchain_i* swapchain, i3_rbk_image_view_i* image_view);

    // end frame
    void (*end_frame)(i3_rbk_device_o* self);

    // wait idle
    void (*wait_idle)(i3_rbk_device_o* self);

    // destroy device
    void (*destroy)(i3_rbk_device_o* self);
};

// backend interface
struct i3_render_backend_i
{
    i3_render_backend_o* self;

    // get render device description
    const i3_rbk_device_desc_t* (*get_device_desc)(i3_render_backend_o* self, uint32_t index);
    uint32_t (*get_device_count)(i3_render_backend_o* self);

    // create render window
    i3_render_window_i* (*create_render_window)(i3_render_backend_o* self,
                                                const char* title,
                                                uint32_t width,
                                                uint32_t height);

    // create render device
    i3_rbk_device_i* (*create_device)(i3_render_backend_o* self, uint32_t desc_index);

    // destroy
    void (*destroy)(i3_render_backend_o* self);
};
