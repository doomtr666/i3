#pragma once

#include "native/core/common.h"
#include "native/render_window/render_window.h"

// backend
typedef struct i3_render_backend_i i3_render_backend_i;
typedef struct i3_render_backend_o i3_render_backend_o;

// device
typedef struct i3_rbk_device_i i3_rbk_device_i;
typedef struct i3_rbk_device_o i3_rbk_device_o;

// enums definition
#define I3_RBK_ENUMS()                                                      \
    /* formats */                                                           \
    I3_RBK_BEGIN_ENUM(format)                                               \
    I3_RBK_ENUM_VALUE(FORMAT, UNDEFINED, 0)                                 \
    I3_RBK_ENUM_VALUE(FORMAT, R8_UNORM, 1)                                  \
    I3_RBK_ENUM_VALUE(FORMAT, R16_UNORM, 2)                                 \
    I3_RBK_ENUM_VALUE(FORMAT, R32_SFLOAT, 3)                                \
    I3_RBK_ENUM_VALUE(FORMAT, R8G8B8A8_UNORM, 4)                            \
    I3_RBK_ENUM_VALUE(FORMAT, A2R10G10B10_UNORM, 5)                         \
    I3_RBK_ENUM_VALUE(FORMAT, R16G16_FLOAT, 6)                              \
    I3_RBK_ENUM_VALUE(FORMAT, R16G16B16A16_FLOAT, 7)                        \
    I3_RBK_ENUM_VALUE(FORMAT, R32G32_SFLOAT, 8)                             \
    I3_RBK_ENUM_VALUE(FORMAT, R32G32B32_SFLOAT, 9)                          \
    I3_RBK_ENUM_VALUE(FORMAT, R32G32B32A32_SFLOAT, 10)                      \
    I3_RBK_ENUM_VALUE(FORMAT, D16_UNORM, 11)                                \
    I3_RBK_ENUM_VALUE(FORMAT, D32_FLOAT, 12)                                \
    I3_RBK_ENUM_VALUE(FORMAT, D24_UNORM_S8_UINT, 13)                        \
    I3_RBK_END_ENUM(format)                                                 \
    /* filter */                                                            \
    I3_RBK_BEGIN_ENUM(filter)                                               \
    I3_RBK_ENUM_VALUE(FILTER, NEAREST, 0)                                   \
    I3_RBK_ENUM_VALUE(FILTER, LINEAR, 1)                                    \
    I3_RBK_END_ENUM(filter)                                                 \
    /* sampler mipmap mode */                                               \
    I3_RBK_BEGIN_ENUM(sampler_mipmap_mode)                                  \
    I3_RBK_ENUM_VALUE(SAMPLER_MIPMAP_MODE, NEAREST, 0)                      \
    I3_RBK_ENUM_VALUE(SAMPLER_MIPMAP_MODE, LINEAR, 1)                       \
    I3_RBK_END_ENUM(sampler_mipmap_mode)                                    \
    /* sampler address mode */                                              \
    I3_RBK_BEGIN_ENUM(sampler_address_mode)                                 \
    I3_RBK_ENUM_VALUE(SAMPLER_ADDRESS_MODE, REPEAT, 0)                      \
    I3_RBK_ENUM_VALUE(SAMPLER_ADDRESS_MODE, MIRRORED_REPEAT, 1)             \
    I3_RBK_ENUM_VALUE(SAMPLER_ADDRESS_MODE, CLAMP_TO_EDGE, 2)               \
    I3_RBK_ENUM_VALUE(SAMPLER_ADDRESS_MODE, CLAMP_TO_BORDER, 3)             \
    I3_RBK_END_ENUM(sampler_address_mode)                                   \
    /* border color */                                                      \
    I3_RBK_BEGIN_ENUM(border_color)                                         \
    I3_RBK_ENUM_VALUE(BORDER_COLOR, FLOAT_TRANSPARENT_BLACK, 0)             \
    I3_RBK_ENUM_VALUE(BORDER_COLOR, INT_TRANSPARENT_BLACK, 1)               \
    I3_RBK_ENUM_VALUE(BORDER_COLOR, FLOAT_OPAQUE_BLACK, 2)                  \
    I3_RBK_ENUM_VALUE(BORDER_COLOR, INT_OPAQUE_BLACK, 3)                    \
    I3_RBK_ENUM_VALUE(BORDER_COLOR, FLOAT_OPAQUE_WHITE, 4)                  \
    I3_RBK_ENUM_VALUE(BORDER_COLOR, INT_OPAQUE_WHITE, 5)                    \
    I3_RBK_END_ENUM(border_color)                                           \
    /* compare op */                                                        \
    I3_RBK_BEGIN_ENUM(compare_op)                                           \
    I3_RBK_ENUM_VALUE(COMPARE_OP, NEVER, 0)                                 \
    I3_RBK_ENUM_VALUE(COMPARE_OP, LESS, 1)                                  \
    I3_RBK_ENUM_VALUE(COMPARE_OP, EQUAL, 2)                                 \
    I3_RBK_ENUM_VALUE(COMPARE_OP, LESS_OR_EQUAL, 3)                         \
    I3_RBK_ENUM_VALUE(COMPARE_OP, GREATER, 4)                               \
    I3_RBK_ENUM_VALUE(COMPARE_OP, NOT_EQUAL, 5)                             \
    I3_RBK_ENUM_VALUE(COMPARE_OP, GREATER_OR_EQUAL, 6)                      \
    I3_RBK_ENUM_VALUE(COMPARE_OP, ALWAYS, 7)                                \
    I3_RBK_END_ENUM(compare_op)                                             \
    /* buffer flags */                                                      \
    I3_RBK_BEGIN_ENUM(buffer_flags)                                         \
    I3_RBK_ENUM_VALUE(BUFFER_FLAG, NONE, 0)                                 \
    I3_RBK_ENUM_VALUE(BUFFER_FLAG, VERTEX_BUFFER, i3_flag(0))               \
    I3_RBK_ENUM_VALUE(BUFFER_FLAG, INDEX_BUFFER, i3_flag(1))                \
    I3_RBK_ENUM_VALUE(BUFFER_FLAG, INDIRECT_BUFFER, i3_flag(2))             \
    I3_RBK_ENUM_VALUE(BUFFER_FLAG, UNIFORM_BUFFER, i3_flag(3))              \
    I3_RBK_ENUM_VALUE(BUFFER_FLAG, STORAGE_BUFFER, i3_flag(4))              \
    I3_RBK_ENUM_VALUE(BUFFER_FLAG, STAGING, i3_flag(5))                     \
    I3_RBK_END_ENUM(buffer_flags)                                           \
    /* image types */                                                       \
    I3_RBK_BEGIN_ENUM(image_type)                                           \
    I3_RBK_ENUM_VALUE(IMAGE_TYPE, D1, 0)                                    \
    I3_RBK_ENUM_VALUE(IMAGE_TYPE, D2, 1)                                    \
    I3_RBK_ENUM_VALUE(IMAGE_TYPE, D3, 2)                                    \
    I3_RBK_END_ENUM(image_type)                                             \
    /* image flags */                                                       \
    I3_RBK_BEGIN_ENUM(image_flags)                                          \
    I3_RBK_ENUM_VALUE(IMAGE_FLAG, NONE, 0)                                  \
    I3_RBK_END_ENUM(image_flags)                                            \
    /* image view types */                                                  \
    I3_RBK_BEGIN_ENUM(image_view_type)                                      \
    I3_RBK_ENUM_VALUE(IMAGE_VIEW_TYPE, UNDEFINED, 0)                        \
    I3_RBK_ENUM_VALUE(IMAGE_VIEW_TYPE, D1, 1)                               \
    I3_RBK_ENUM_VALUE(IMAGE_VIEW_TYPE, D2, 2)                               \
    I3_RBK_ENUM_VALUE(IMAGE_VIEW_TYPE, D3, 3)                               \
    I3_RBK_ENUM_VALUE(IMAGE_VIEW_TYPE, CUBE, 4)                             \
    I3_RBK_ENUM_VALUE(IMAGE_VIEW_TYPE, D1_ARRAY, 5)                         \
    I3_RBK_ENUM_VALUE(IMAGE_VIEW_TYPE, D2_ARRAY, 6)                         \
    I3_RBK_ENUM_VALUE(IMAGE_VIEW_TYPE, CUBE_ARRAY, 7)                       \
    I3_RBK_END_ENUM(image_view_type)                                        \
    /* aspect flags */                                                      \
    I3_RBK_BEGIN_ENUM(image_aspect_flags)                                   \
    I3_RBK_ENUM_VALUE(IMAGE_ASPECT, COLOR, i3_flag(0))                      \
    I3_RBK_ENUM_VALUE(IMAGE_ASPECT, DEPTH, i3_flag(1))                      \
    I3_RBK_ENUM_VALUE(IMAGE_ASPECT, STENCIL, i3_flag(2))                    \
    I3_RBK_END_ENUM(image_aspect_flags)                                     \
    /* component swizzle */                                                 \
    I3_RBK_BEGIN_ENUM(component_swizzle)                                    \
    I3_RBK_ENUM_VALUE(COMPONENT_SWIZZLE, IDENTITY, 0)                       \
    I3_RBK_ENUM_VALUE(COMPONENT_SWIZZLE, ZERO, 1)                           \
    I3_RBK_ENUM_VALUE(COMPONENT_SWIZZLE, ONE, 2)                            \
    I3_RBK_ENUM_VALUE(COMPONENT_SWIZZLE, R, 3)                              \
    I3_RBK_ENUM_VALUE(COMPONENT_SWIZZLE, G, 4)                              \
    I3_RBK_ENUM_VALUE(COMPONENT_SWIZZLE, B, 5)                              \
    I3_RBK_ENUM_VALUE(COMPONENT_SWIZZLE, A, 6)                              \
    I3_RBK_END_ENUM(component_swizzle)                                      \
    /* shader stage */                                                      \
    I3_RBK_BEGIN_ENUM(shader_stage_flags)                                   \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, VERTEX, i3_flag(0))                     \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, TESSELLATION_CONTROL, i3_flag(1))       \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, TESSELLATION_EVALUATION, i3_flag(2))    \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, GEOMETRY, i3_flag(3))                   \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, FRAGMENT, i3_flag(4))                   \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, COMPUTE, i3_flag(5))                    \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, RAYGEN, i3_flag(6))                     \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, ANY_HIT, i3_flag(7))                    \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, CLOSEST_HIT, i3_flag(8))                \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, MISS, i3_flag(9))                       \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, INTERSECTION, i3_flag(10))              \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, CALLABLE, i3_flag(11))                  \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, TASK, i3_flag(12))                      \
    I3_RBK_ENUM_VALUE(SHADER_STAGE, MESH, i3_flag(13))                      \
    I3_RBK_END_ENUM(shader_stage_flags)                                     \
    /* vertex input rate */                                                 \
    I3_RBK_BEGIN_ENUM(vertex_input_rate)                                    \
    I3_RBK_ENUM_VALUE(VERTEX_INPUT_RATE, VERTEX, 0)                         \
    I3_RBK_ENUM_VALUE(VERTEX_INPUT_RATE, INSTANCE, 1)                       \
    I3_RBK_END_ENUM(vertex_input_rate)                                      \
    /* primitive topology */                                                \
    I3_RBK_BEGIN_ENUM(primitive_topology)                                   \
    I3_RBK_ENUM_VALUE(PRIMITIVE_TOPOLOGY, POINT_LIST, 0)                    \
    I3_RBK_ENUM_VALUE(PRIMITIVE_TOPOLOGY, LINE_LIST, 1)                     \
    I3_RBK_ENUM_VALUE(PRIMITIVE_TOPOLOGY, LINE_STRIP, 2)                    \
    I3_RBK_ENUM_VALUE(PRIMITIVE_TOPOLOGY, TRIANGLE_LIST, 3)                 \
    I3_RBK_ENUM_VALUE(PRIMITIVE_TOPOLOGY, TRIANGLE_STRIP, 4)                \
    I3_RBK_ENUM_VALUE(PRIMITIVE_TOPOLOGY, TRIANGLE_FAN, 5)                  \
    I3_RBK_ENUM_VALUE(PRIMITIVE_TOPOLOGY, LINE_LIST_WITH_ADJACENCY, 6)      \
    I3_RBK_ENUM_VALUE(PRIMITIVE_TOPOLOGY, LINE_STRIP_WITH_ADJACENCY, 7)     \
    I3_RBK_ENUM_VALUE(PRIMITIVE_TOPOLOGY, TRIANGLE_LIST_WITH_ADJACENCY, 8)  \
    I3_RBK_ENUM_VALUE(PRIMITIVE_TOPOLOGY, TRIANGLE_STRIP_WITH_ADJACENCY, 9) \
    I3_RBK_ENUM_VALUE(PRIMITIVE_TOPOLOGY, PATCH_LIST, 10)                   \
    I3_RBK_END_ENUM(primitive_topology)                                     \
    /* polygon mode */                                                      \
    I3_RBK_BEGIN_ENUM(polygon_mode)                                         \
    I3_RBK_ENUM_VALUE(POLYGON_MODE, FILL, 0)                                \
    I3_RBK_ENUM_VALUE(POLYGON_MODE, LINE, 1)                                \
    I3_RBK_ENUM_VALUE(POLYGON_MODE, POINT, 2)                               \
    I3_RBK_END_ENUM(polygon_mode)                                           \
    /* cull mode */                                                         \
    I3_RBK_BEGIN_ENUM(cull_mode_flags)                                      \
    I3_RBK_ENUM_VALUE(CULL_MODE, NONE, 0)                                   \
    I3_RBK_ENUM_VALUE(CULL_MODE, FRONT, i3_flag(0))                         \
    I3_RBK_ENUM_VALUE(CULL_MODE, BACK, i3_flag(1))                          \
    I3_RBK_ENUM_VALUE(CULL_MODE, FRONT_AND_BACK, i3_flag(0) | i3_flag(1))   \
    I3_RBK_END_ENUM(cull_mode_flags)                                        \
    /* stencil op */                                                        \
    I3_RBK_BEGIN_ENUM(stencil_op)                                           \
    I3_RBK_ENUM_VALUE(STENCIL_OP, KEEP, 0)                                  \
    I3_RBK_ENUM_VALUE(STENCIL_OP, ZERO, 1)                                  \
    I3_RBK_ENUM_VALUE(STENCIL_OP, REPLACE, 2)                               \
    I3_RBK_ENUM_VALUE(STENCIL_OP, INCREMENT_AND_CLAMP, 3)                   \
    I3_RBK_ENUM_VALUE(STENCIL_OP, DECREMENT_AND_CLAMP, 4)                   \
    I3_RBK_ENUM_VALUE(STENCIL_OP, INVERT, 5)                                \
    I3_RBK_ENUM_VALUE(STENCIL_OP, INCREMENT_AND_WRAP, 6)                    \
    I3_RBK_ENUM_VALUE(STENCIL_OP, DECREMENT_AND_WRAP, 7)                    \
    I3_RBK_END_ENUM(stencil_op)                                             \
    /* front face */                                                        \
    I3_RBK_BEGIN_ENUM(front_face)                                           \
    I3_RBK_ENUM_VALUE(FRONT_FACE, COUNTER_CLOCKWISE, 0)                     \
    I3_RBK_ENUM_VALUE(FRONT_FACE, CLOCKWISE, 1)                             \
    I3_RBK_END_ENUM(front_face)                                             \
    /* logic op */                                                          \
    I3_RBK_BEGIN_ENUM(logic_op)                                             \
    I3_RBK_ENUM_VALUE(LOGIC_OP, CLEAR, 0)                                   \
    I3_RBK_ENUM_VALUE(LOGIC_OP, AND, 1)                                     \
    I3_RBK_ENUM_VALUE(LOGIC_OP, AND_REVERSE, 2)                             \
    I3_RBK_ENUM_VALUE(LOGIC_OP, COPY, 3)                                    \
    I3_RBK_ENUM_VALUE(LOGIC_OP, AND_INVERTED, 4)                            \
    I3_RBK_ENUM_VALUE(LOGIC_OP, NO_OP, 5)                                   \
    I3_RBK_ENUM_VALUE(LOGIC_OP, XOR, 6)                                     \
    I3_RBK_ENUM_VALUE(LOGIC_OP, OR, 7)                                      \
    I3_RBK_ENUM_VALUE(LOGIC_OP, NOR, 8)                                     \
    I3_RBK_ENUM_VALUE(LOGIC_OP, EQUIVALENT, 9)                              \
    I3_RBK_ENUM_VALUE(LOGIC_OP, INVERT, 10)                                 \
    I3_RBK_ENUM_VALUE(LOGIC_OP, OR_REVERSE, 11)                             \
    I3_RBK_ENUM_VALUE(LOGIC_OP, COPY_INVERTED, 12)                          \
    I3_RBK_ENUM_VALUE(LOGIC_OP, OR_INVERTED, 13)                            \
    I3_RBK_ENUM_VALUE(LOGIC_OP, NAND, 14)                                   \
    I3_RBK_ENUM_VALUE(LOGIC_OP, SET, 15)                                    \
    I3_RBK_END_ENUM(logic_op)                                               \
    /* blend factor */                                                      \
    I3_RBK_BEGIN_ENUM(blend_factor)                                         \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, ZERO, 0)                                \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, ONE, 1)                                 \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, SRC_COLOR, 2)                           \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, ONE_MINUS_SRC_COLOR, 3)                 \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, DST_COLOR, 4)                           \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, ONE_MINUS_DST_COLOR, 5)                 \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, SRC_ALPHA, 6)                           \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, ONE_MINUS_SRC_ALPHA, 7)                 \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, DST_ALPHA, 8)                           \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, ONE_MINUS_DST_ALPHA, 9)                 \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, CONSTANT_COLOR, 10)                     \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, ONE_MINUS_CONSTANT_COLOR, 11)           \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, CONSTANT_ALPHA, 12)                     \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, ONE_MINUS_CONSTANT_ALPHA, 13)           \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, SRC_ALPHA_SATURATE, 14)                 \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, SRC1_COLOR, 15)                         \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, ONE_MINUS_SRC1_COLOR, 16)               \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, SRC1_ALPHA, 17)                         \
    I3_RBK_ENUM_VALUE(BLEND_FACTOR, ONE_MINUS_SRC1_ALPHA, 18)               \
    I3_RBK_END_ENUM(blend_factor)                                           \
    /* blend op */                                                          \
    I3_RBK_BEGIN_ENUM(blend_op)                                             \
    I3_RBK_ENUM_VALUE(BLEND_OP, ADD, 0)                                     \
    I3_RBK_ENUM_VALUE(BLEND_OP, SUBTRACT, 1)                                \
    I3_RBK_ENUM_VALUE(BLEND_OP, REVERSE_SUBTRACT, 2)                        \
    I3_RBK_ENUM_VALUE(BLEND_OP, MIN, 3)                                     \
    I3_RBK_ENUM_VALUE(BLEND_OP, MAX, 4)                                     \
    I3_RBK_END_ENUM(blend_op)                                               \
    /* color component flags */                                             \
    I3_RBK_BEGIN_ENUM(color_component_flags)                                \
    I3_RBK_ENUM_VALUE(COLOR_COMPONENT, R, i3_flag(0))                       \
    I3_RBK_ENUM_VALUE(COLOR_COMPONENT, G, i3_flag(1))                       \
    I3_RBK_ENUM_VALUE(COLOR_COMPONENT, B, i3_flag(2))                       \
    I3_RBK_ENUM_VALUE(COLOR_COMPONENT, A, i3_flag(3))                       \
    I3_RBK_END_ENUM(color_component_flags)                                  \
    /* dynamic states */                                                    \
    I3_RBK_BEGIN_ENUM(dynamic_state)                                        \
    I3_RBK_ENUM_VALUE(DYNAMIC_STATE, VIEWPORT, 0)                           \
    I3_RBK_ENUM_VALUE(DYNAMIC_STATE, SCISSOR, 1)                            \
    I3_RBK_ENUM_VALUE(DYNAMIC_STATE, LINE_WIDTH, 2)                         \
    I3_RBK_ENUM_VALUE(DYNAMIC_STATE, DEPTH_BIAS, 3)                         \
    I3_RBK_ENUM_VALUE(DYNAMIC_STATE, BLEND_CONSTANTS, 4)                    \
    I3_RBK_ENUM_VALUE(DYNAMIC_STATE, DEPTH_BOUNDS, 5)                       \
    I3_RBK_ENUM_VALUE(DYNAMIC_STATE, STENCIL_COMPARE_MASK, 6)               \
    I3_RBK_ENUM_VALUE(DYNAMIC_STATE, STENCIL_WRITE_MASK, 7)                 \
    I3_RBK_ENUM_VALUE(DYNAMIC_STATE, STENCIL_REFERENCE, 8)                  \
    I3_RBK_END_ENUM(dynamic_state)                                          \
    /* descriptor type */                                                   \
    I3_RBK_BEGIN_ENUM(descriptor_type)                                      \
    I3_RBK_ENUM_VALUE(DESCRIPTOR_TYPE, SAMPLER, 0)                          \
    I3_RBK_ENUM_VALUE(DESCRIPTOR_TYPE, COMBINED_IMAGE_SAMPLER, 1)           \
    I3_RBK_ENUM_VALUE(DESCRIPTOR_TYPE, SAMPLED_IMAGE, 2)                    \
    I3_RBK_ENUM_VALUE(DESCRIPTOR_TYPE, STORAGE_IMAGE, 3)                    \
    I3_RBK_ENUM_VALUE(DESCRIPTOR_TYPE, UNIFORM_TEXEL_BUFFER, 4)             \
    I3_RBK_ENUM_VALUE(DESCRIPTOR_TYPE, STORAGE_TEXEL_BUFFER, 5)             \
    I3_RBK_ENUM_VALUE(DESCRIPTOR_TYPE, UNIFORM_BUFFER, 6)                   \
    I3_RBK_ENUM_VALUE(DESCRIPTOR_TYPE, STORAGE_BUFFER, 7)                   \
    I3_RBK_ENUM_VALUE(DESCRIPTOR_TYPE, UNIFORM_BUFFER_DYNAMIC, 8)           \
    I3_RBK_ENUM_VALUE(DESCRIPTOR_TYPE, STORAGE_BUFFER_DYNAMIC, 9)           \
    I3_RBK_ENUM_VALUE(DESCRIPTOR_TYPE, INPUT_ATTACHMENT, 10)                \
    I3_RBK_END_ENUM(descriptor_type)                                        \
    /* index type */                                                        \
    I3_RBK_BEGIN_ENUM(index_type)                                           \
    I3_RBK_ENUM_VALUE(INDEX_TYPE, UINT16, 0)                                \
    I3_RBK_ENUM_VALUE(INDEX_TYPE, UINT32, 1)                                \
    I3_RBK_END_ENUM(index_type)                                             \
    /* attachment load/store ops */                                         \
    I3_RBK_BEGIN_ENUM(attachment_load_op)                                   \
    I3_RBK_ENUM_VALUE(ATTACHMENT_LOAD_OP, LOAD, 0)                          \
    I3_RBK_ENUM_VALUE(ATTACHMENT_LOAD_OP, CLEAR, 1)                         \
    I3_RBK_ENUM_VALUE(ATTACHMENT_LOAD_OP, DONT_CARE, 2)                     \
    I3_RBK_END_ENUM(attachment_load_op)                                     \
    /* attachment store ops */                                              \
    I3_RBK_BEGIN_ENUM(attachment_store_op)                                  \
    I3_RBK_ENUM_VALUE(ATTACHMENT_STORE_OP, STORE, 0)                        \
    I3_RBK_ENUM_VALUE(ATTACHMENT_STORE_OP, DONT_CARE, 1)                    \
    I3_RBK_END_ENUM(attachment_store_op)

// generate enums
#define I3_RBK_BEGIN_ENUM(name) \
    typedef enum                \
    {
#define I3_RBK_ENUM_VALUE(prefix, name, value) I3_RBK_##prefix##_##name = value,
#define I3_RBK_END_ENUM(name) \
    }                         \
    i3_rbk_##name##_t;

I3_RBK_ENUMS()

#undef I3_RBK_BEGIN_ENUM
#undef I3_RBK_ENUM_VALUE
#undef I3_RBK_END_ENUM

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
typedef union i3_rbk_clear_color_value_t
{
    float float32[4];
    int32_t int32[4];
    uint32_t uint32[4];
} i3_rbk_clear_color_value_t;

// clear depth stencil
typedef struct i3_rbk_clear_depth_stencil_value_t
{
    float depth;
    uint32_t stencil;
} i3_rbk_clear_depth_stencil_value_t;

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

// descriptor set
typedef struct i3_rbk_descriptor_set_o i3_rbk_descriptor_set_o;

typedef struct i3_rbk_descriptor_set_write_t
{
    uint32_t binding;
    uint32_t array_element;
    i3_rbk_descriptor_type_t descriptor_type;
    const i3_rbk_sampler_i* sampler;
    const i3_rbk_image_view_i* image;
    const i3_rbk_buffer_i* buffer;
} i3_rbk_descriptor_set_write_t;

typedef struct i3_rbk_descriptor_set_i
{
    i3_rbk_descriptor_set_o* self;
    i3_rbk_resource_i* (*get_resource)(i3_rbk_descriptor_set_o* self);
    void (*update)(i3_rbk_descriptor_set_o* self, uint32_t write_count, const i3_rbk_descriptor_set_write_t* writes);
    void (*destroy)(i3_rbk_descriptor_set_o* self);
} i3_rbk_descriptor_set_i;

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

// attachment description
typedef struct i3_rbk_attachment_desc_t
{
    i3_rbk_format_t format;
    uint32_t samples;
} i3_rbk_attachment_desc_t;

// shader stage
typedef struct i3_rbk_pipeline_shader_stage_desc_t
{
    i3_rbk_shader_stage_flags_t stage;
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
    int patch_control_points;
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
    uint32_t color_attachment_count;
    const i3_rbk_attachment_desc_t* color_attachments;
    const i3_rbk_attachment_desc_t* depth_stencil_attachment;
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
} i3_rbk_graphics_pipeline_desc_t;

// compute pipeline
typedef struct i3_rbk_compute_pipeline_desc_t
{
    i3_rbk_pipeline_shader_stage_desc_t stage;
} i3_rbk_compute_pipeline_desc_t;

// pipeline interface
typedef struct i3_rbk_pipeline_o i3_rbk_pipeline_o;

typedef struct i3_rbk_pipeline_i
{
    i3_rbk_pipeline_o* self;

    i3_rbk_resource_i* (*get_resource)(i3_rbk_pipeline_o* self);
    i3_rbk_pipeline_layout_i* (*get_layout)(i3_rbk_pipeline_o* self);

    void (*destroy)(i3_rbk_pipeline_o* self);
} i3_rbk_pipeline_i;

// framebuffer
typedef struct i3_rbk_framebuffer_desc_t
{
    uint32_t width;
    uint32_t height;
    uint32_t layers;
    i3_rbk_pipeline_i* graphics_pipeline;
    uint32_t color_attachment_count;
    i3_rbk_image_view_i** color_attachments;
    i3_rbk_image_view_i* depth_stencil_attachment;
} i3_rbk_framebuffer_desc_t;

typedef struct i3_rbk_framebuffer_o i3_rbk_framebuffer_o;

typedef struct i3_rbk_framebuffer_i
{
    i3_rbk_framebuffer_o* self;
    i3_rbk_resource_i* (*get_resource)(i3_rbk_framebuffer_o* self);
    void (*destroy)(i3_rbk_framebuffer_o* self);
} i3_rbk_framebuffer_i;

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

    void (*clear_color_image)(i3_rbk_cmd_buffer_o* self,
                              i3_rbk_image_view_i* image_view,
                              const i3_rbk_clear_color_value_t* color);

    void (*clear_depth_stencil_image)(i3_rbk_cmd_buffer_o* self,
                                      i3_rbk_image_view_i* image_view,
                                      const i3_rbk_clear_depth_stencil_value_t* depth_stencil);

    void (*bind_vertex_buffers)(i3_rbk_cmd_buffer_o* self,
                                uint32_t first_binding,
                                uint32_t binding_count,
                                i3_rbk_buffer_i** buffers,
                                const uint32_t* offsets);

    void (*bind_index_buffer)(i3_rbk_cmd_buffer_o* self,
                              i3_rbk_buffer_i* buffer,
                              uint32_t offset,
                              i3_rbk_index_type_t index_type);

    void (*bind_descriptor_sets)(i3_rbk_cmd_buffer_o* self,
                                 i3_rbk_pipeline_i* pipeline,
                                 uint32_t first_set,
                                 uint32_t descriptor_set_count,
                                 i3_rbk_descriptor_set_i** descriptor_sets);

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

    // create descriptor set
    i3_rbk_descriptor_set_i* (*create_descriptor_set)(i3_rbk_device_o* self, i3_rbk_descriptor_set_layout_i* layout);

    // create pipeline layout
    i3_rbk_pipeline_layout_i* (*create_pipeline_layout)(i3_rbk_device_o* self,
                                                        const i3_rbk_pipeline_layout_desc_t* desc);

    // create framebuffer
    i3_rbk_framebuffer_i* (*create_framebuffer)(i3_rbk_device_o* self, const i3_rbk_framebuffer_desc_t* desc);

    // create shader module
    i3_rbk_shader_module_i* (*create_shader_module)(i3_rbk_device_o* self, const i3_rbk_shader_module_desc_t* desc);

    // create graphics pipeline
    i3_rbk_pipeline_i* (*create_graphics_pipeline)(i3_rbk_device_o* self,
                                                   i3_rbk_pipeline_layout_i* layout,
                                                   const i3_rbk_graphics_pipeline_desc_t* desc);

    // create compute pipeline
    i3_rbk_pipeline_i* (*create_compute_pipeline)(i3_rbk_device_o* self,
                                                  i3_rbk_pipeline_layout_i* layout,
                                                  const i3_rbk_compute_pipeline_desc_t* desc);

    // create swapchain
    i3_rbk_swapchain_i* (*create_swapchain)(i3_rbk_device_o* self,
                                            i3_render_window_i* window,
                                            const i3_rbk_swapchain_desc_t* desc);

    // create cmd buffer
    i3_rbk_cmd_buffer_i* (*create_cmd_buffer)(i3_rbk_device_o* self);

    // submit cmd buffers
    void (*submit_cmd_buffers)(i3_rbk_device_o* self, uint32_t cmd_buffer_count, i3_rbk_cmd_buffer_i** cmd_buffers);

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
