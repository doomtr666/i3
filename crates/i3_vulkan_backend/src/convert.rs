#![allow(unused_imports)]
use ash::vk;
use i3_gfx::graph::pipeline::{
    BindingType, BlendFactor, BlendOp, ColorComponentFlags, CompareOp, CullMode, IndexType,
    InputAssemblyState, LogicOp, MipmapMode, MultisampleState, PolygonMode, PrimitiveTopology,
    RasterizationState, ShaderStageFlags, StencilOp, StencilOpState, VertexFormat, VertexInputRate,
};
use i3_gfx::graph::types::{
    AddressMode, BorderColor, BufferUsageFlags, ComponentMapping, ComponentSwizzle, Filter, Format,
    ImageAspectFlags, ImageType, ImageUsageFlags, ImageViewType, SampleCount,
};
// Removed ComponentSwizzle as it wasn't in my port of pipeline.rs yet?
// Wait, I might have missed ComponentSwizzle in pipeline.rs or types.rs?
// I checked types.rs in Step 7847, it wasn't there.
// I checked pipeline.rs in Step 7824 (creation), I didn't include ComponentSwizzle!
// I need to add ComponentSwizzle to pipeline.rs or types.rs.
// I'll add it to pipeline.rs for now, or just types.rs. Usage is usually in ImageView creation.
// Let's add it to types.rs or pipeline.rs implicitly.
// Actually, `convert.rs` from `old` had it.
// I will comment it out if missing, or add it.

pub fn convert_bool(value: bool) -> vk::Bool32 {
    if value { vk::TRUE } else { vk::FALSE }
}

pub fn convert_format(format: Format) -> vk::Format {
    match format {
        Format::Undefined => vk::Format::UNDEFINED,
        Format::R8G8B8A8_UNORM => vk::Format::R8G8B8A8_UNORM,
        Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_SRGB,
        Format::B8G8R8A8_UNORM => vk::Format::B8G8R8A8_UNORM,
        Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_SRGB,
        Format::R8G8_UNORM => vk::Format::R8G8_UNORM,
        Format::R16G16_SFLOAT => vk::Format::R16G16_SFLOAT,
        Format::R16G16B16A16_SFLOAT => vk::Format::R16G16B16A16_SFLOAT,
        Format::R11G11B10_UFLOAT => vk::Format::B10G11R11_UFLOAT_PACK32,
        Format::R32_FLOAT => vk::Format::R32_SFLOAT,
        Format::R32G32B32A32_FLOAT => vk::Format::R32G32B32A32_SFLOAT,
        Format::D32_FLOAT => vk::Format::D32_SFLOAT,
        Format::R32_SINT => vk::Format::R32_SINT,
        Format::R32G32_SINT => vk::Format::R32G32_SINT,
        Format::R32G32B32_SINT => vk::Format::R32G32B32_SINT,
        Format::R32G32B32A32_SINT => vk::Format::R32G32B32A32_SINT,
        Format::R32_UINT => vk::Format::R32_UINT,
        Format::R32G32_UINT => vk::Format::R32G32_UINT,
        Format::R32G32B32_UINT => vk::Format::R32G32B32_UINT,
        Format::R32G32B32A32_UINT => vk::Format::R32G32B32A32_UINT,
        Format::R32_SFLOAT => vk::Format::R32_SFLOAT,
        Format::R32G32_SFLOAT => vk::Format::R32G32_SFLOAT,
        Format::R32G32B32_SFLOAT => vk::Format::R32G32B32_SFLOAT,
        Format::R32G32B32A32_SFLOAT => vk::Format::R32G32B32A32_SFLOAT,
        Format::BC1_RGB_UNORM => vk::Format::BC1_RGB_UNORM_BLOCK,
        Format::BC1_RGB_SRGB => vk::Format::BC1_RGB_SRGB_BLOCK,
        Format::BC1_RGBA_UNORM => vk::Format::BC1_RGBA_UNORM_BLOCK,
        Format::BC1_RGBA_SRGB => vk::Format::BC1_RGBA_SRGB_BLOCK,
        Format::BC3_UNORM => vk::Format::BC3_UNORM_BLOCK,
        Format::BC3_SRGB => vk::Format::BC3_SRGB_BLOCK,
        Format::BC4_UNORM => vk::Format::BC4_UNORM_BLOCK,
        Format::BC4_SNORM => vk::Format::BC4_SNORM_BLOCK,
        Format::BC5_UNORM => vk::Format::BC5_UNORM_BLOCK,
        Format::BC5_SNORM => vk::Format::BC5_SNORM_BLOCK,
        Format::BC7_UNORM => vk::Format::BC7_UNORM_BLOCK,
        Format::BC7_SRGB => vk::Format::BC7_SRGB_BLOCK,
    }
}

pub fn convert_image_type(image_type: ImageType) -> vk::ImageType {
    match image_type {
        ImageType::Type1D => vk::ImageType::TYPE_1D,
        ImageType::Type2D => vk::ImageType::TYPE_2D,
        ImageType::Type3D => vk::ImageType::TYPE_3D,
    }
}

pub fn convert_image_usage_flags(usage_flags: ImageUsageFlags) -> vk::ImageUsageFlags {
    let mut vk_usage_flags = vk::ImageUsageFlags::empty();

    if usage_flags.contains(ImageUsageFlags::TRANSFER_SRC) {
        vk_usage_flags |= vk::ImageUsageFlags::TRANSFER_SRC;
    }
    if usage_flags.contains(ImageUsageFlags::TRANSFER_DST) {
        vk_usage_flags |= vk::ImageUsageFlags::TRANSFER_DST;
    }
    if usage_flags.contains(ImageUsageFlags::SAMPLED) {
        vk_usage_flags |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage_flags.contains(ImageUsageFlags::STORAGE) {
        vk_usage_flags |= vk::ImageUsageFlags::STORAGE;
    }
    if usage_flags.contains(ImageUsageFlags::COLOR_ATTACHMENT) {
        vk_usage_flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    if usage_flags.contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
        vk_usage_flags |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    }

    vk_usage_flags
}

pub fn convert_buffer_usage_flags(usage_flags: BufferUsageFlags) -> vk::BufferUsageFlags {
    let mut vk_usage_flags = vk::BufferUsageFlags::empty();

    if usage_flags.contains(BufferUsageFlags::TRANSFER_SRC) {
        vk_usage_flags |= vk::BufferUsageFlags::TRANSFER_SRC;
    }
    if usage_flags.contains(BufferUsageFlags::TRANSFER_DST) {
        vk_usage_flags |= vk::BufferUsageFlags::TRANSFER_DST;
    }
    if usage_flags.contains(BufferUsageFlags::UNIFORM_TEXEL_BUFFER) {
        vk_usage_flags |= vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER;
    }
    if usage_flags.contains(BufferUsageFlags::STORAGE_TEXEL_BUFFER) {
        vk_usage_flags |= vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER;
    }
    if usage_flags.contains(BufferUsageFlags::UNIFORM_BUFFER) {
        vk_usage_flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if usage_flags.contains(BufferUsageFlags::STORAGE_BUFFER) {
        vk_usage_flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if usage_flags.contains(BufferUsageFlags::INDEX_BUFFER) {
        vk_usage_flags |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if usage_flags.contains(BufferUsageFlags::VERTEX_BUFFER) {
        vk_usage_flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if usage_flags.contains(BufferUsageFlags::INDIRECT_BUFFER) {
        vk_usage_flags |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    if usage_flags.contains(BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
        vk_usage_flags |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS_KHR;
    }
    if usage_flags.contains(BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT) {
        vk_usage_flags |= vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
    }
    if usage_flags.contains(BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE) {
        vk_usage_flags |= vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR;
    }

    vk_usage_flags
}

pub fn convert_index_type(index_type: IndexType) -> vk::IndexType {
    match index_type {
        IndexType::Uint16 => vk::IndexType::UINT16,
        IndexType::Uint32 => vk::IndexType::UINT32,
    }
}

pub fn convert_sample_count(sample_count: SampleCount) -> vk::SampleCountFlags {
    match sample_count {
        SampleCount::Sample1 => vk::SampleCountFlags::TYPE_1,
        SampleCount::Sample2 => vk::SampleCountFlags::TYPE_2,
        SampleCount::Sample4 => vk::SampleCountFlags::TYPE_4,
        SampleCount::Sample8 => vk::SampleCountFlags::TYPE_8,
        SampleCount::Sample16 => vk::SampleCountFlags::TYPE_16,
        SampleCount::Sample32 => vk::SampleCountFlags::TYPE_32,
        SampleCount::Sample64 => vk::SampleCountFlags::TYPE_64,
    }
}

pub fn convert_vertex_format(format: VertexFormat) -> vk::Format {
    match format {
        VertexFormat::Float => vk::Format::R32_SFLOAT,
        VertexFormat::Float2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Int => vk::Format::R32_SINT,
        VertexFormat::Int2 => vk::Format::R32G32_SINT,
        VertexFormat::Int3 => vk::Format::R32G32B32_SINT,
        VertexFormat::Int4 => vk::Format::R32G32B32A32_SINT,
        VertexFormat::UInt => vk::Format::R32_UINT,
        VertexFormat::UInt2 => vk::Format::R32G32_UINT,
        VertexFormat::UInt3 => vk::Format::R32G32B32_UINT,
        VertexFormat::UInt4 => vk::Format::R32G32B32A32_UINT,
    }
}

pub fn convert_vertex_input_rate(rate: VertexInputRate) -> vk::VertexInputRate {
    match rate {
        VertexInputRate::Vertex => vk::VertexInputRate::VERTEX,
        VertexInputRate::Instance => vk::VertexInputRate::INSTANCE,
    }
}

pub fn convert_primitive_topology(topology: PrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
        PrimitiveTopology::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
    }
}

pub fn convert_polygon_mode(mode: PolygonMode) -> vk::PolygonMode {
    match mode {
        PolygonMode::Fill => vk::PolygonMode::FILL,
        PolygonMode::Line => vk::PolygonMode::LINE,
        PolygonMode::Point => vk::PolygonMode::POINT,
    }
}

pub fn convert_cull_mode(mode: CullMode) -> vk::CullModeFlags {
    match mode {
        CullMode::None => vk::CullModeFlags::NONE,
        CullMode::Front => vk::CullModeFlags::FRONT,
        CullMode::Back => vk::CullModeFlags::BACK,
    }
}

pub fn convert_blend_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
    }
}

pub fn convert_blend_op(op: BlendOp) -> vk::BlendOp {
    match op {
        BlendOp::Add => vk::BlendOp::ADD,
        BlendOp::Subtract => vk::BlendOp::SUBTRACT,
        BlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOp::Min => vk::BlendOp::MIN,
        BlendOp::Max => vk::BlendOp::MAX,
    }
}

pub fn convert_shader_stage_flags(flags: ShaderStageFlags) -> vk::ShaderStageFlags {
    let mut result = vk::ShaderStageFlags::empty();
    if flags.contains(ShaderStageFlags::Vertex) {
        result |= vk::ShaderStageFlags::VERTEX;
    }
    if flags.contains(ShaderStageFlags::Fragment) {
        result |= vk::ShaderStageFlags::FRAGMENT;
    }
    if flags.contains(ShaderStageFlags::Compute) {
        result |= vk::ShaderStageFlags::COMPUTE;
    }
    if flags.contains(ShaderStageFlags::Geometry) {
        result |= vk::ShaderStageFlags::GEOMETRY;
    }
    if flags.contains(ShaderStageFlags::TessellationControl) {
        result |= vk::ShaderStageFlags::TESSELLATION_CONTROL;
    }
    if flags.contains(ShaderStageFlags::TessellationEvaluation) {
        result |= vk::ShaderStageFlags::TESSELLATION_EVALUATION;
    }
    result
}

pub fn convert_color_component_flags(flags: ColorComponentFlags) -> vk::ColorComponentFlags {
    let mut vk_flags = vk::ColorComponentFlags::empty();
    if flags.contains(ColorComponentFlags::R) {
        vk_flags |= vk::ColorComponentFlags::R;
    }
    if flags.contains(ColorComponentFlags::G) {
        vk_flags |= vk::ColorComponentFlags::G;
    }
    if flags.contains(ColorComponentFlags::B) {
        vk_flags |= vk::ColorComponentFlags::B;
    }
    if flags.contains(ColorComponentFlags::A) {
        vk_flags |= vk::ColorComponentFlags::A;
    }
    vk_flags
}

pub fn convert_stencil_op(op: StencilOp) -> vk::StencilOp {
    match op {
        StencilOp::Keep => vk::StencilOp::KEEP,
        StencilOp::Zero => vk::StencilOp::ZERO,
        StencilOp::Replace => vk::StencilOp::REPLACE,
        StencilOp::IncrementAndClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        StencilOp::DecrementAndClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        StencilOp::Invert => vk::StencilOp::INVERT,
        StencilOp::IncrementAndWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        StencilOp::DecrementAndWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

pub fn convert_compare_op(op: CompareOp) -> vk::CompareOp {
    match op {
        CompareOp::Never => vk::CompareOp::NEVER,
        CompareOp::Less => vk::CompareOp::LESS,
        CompareOp::Equal => vk::CompareOp::EQUAL,
        CompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareOp::Greater => vk::CompareOp::GREATER,
        CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareOp::Always => vk::CompareOp::ALWAYS,
    }
}

pub fn convert_stencil_op_state(state: &StencilOpState) -> vk::StencilOpState {
    vk::StencilOpState {
        fail_op: convert_stencil_op(state.fail_op),
        pass_op: convert_stencil_op(state.pass_op),
        depth_fail_op: convert_stencil_op(state.depth_fail_op),
        compare_op: convert_compare_op(state.compare_op),
        compare_mask: state.compare_mask,
        write_mask: state.write_mask,
        reference: state.reference,
    }
}

pub fn convert_binding_type_to_descriptor(binding_type: BindingType) -> vk::DescriptorType {
    match binding_type {
        BindingType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        BindingType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        BindingType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
        BindingType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        BindingType::Sampler => vk::DescriptorType::SAMPLER,
        BindingType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        BindingType::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        BindingType::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        BindingType::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        BindingType::Unknown => vk::DescriptorType::UNIFORM_BUFFER, // fallback
    }
}

pub fn convert_image_view_type(view_type: ImageViewType) -> vk::ImageViewType {
    match view_type {
        ImageViewType::Type1D => vk::ImageViewType::TYPE_1D,
        ImageViewType::Type2D => vk::ImageViewType::TYPE_2D,
        ImageViewType::Type3D => vk::ImageViewType::TYPE_3D,
        ImageViewType::TypeCube => vk::ImageViewType::CUBE,
        ImageViewType::Type1DArray => vk::ImageViewType::TYPE_1D_ARRAY,
        ImageViewType::Type2DArray => vk::ImageViewType::TYPE_2D_ARRAY,
        ImageViewType::TypeCubeArray => vk::ImageViewType::CUBE_ARRAY,
    }
}

pub fn convert_component_swizzle(swizzle: ComponentSwizzle) -> vk::ComponentSwizzle {
    match swizzle {
        ComponentSwizzle::Identity => vk::ComponentSwizzle::IDENTITY,
        ComponentSwizzle::Zero => vk::ComponentSwizzle::ZERO,
        ComponentSwizzle::One => vk::ComponentSwizzle::ONE,
        ComponentSwizzle::R => vk::ComponentSwizzle::R,
        ComponentSwizzle::G => vk::ComponentSwizzle::G,
        ComponentSwizzle::B => vk::ComponentSwizzle::B,
        ComponentSwizzle::A => vk::ComponentSwizzle::A,
    }
}

pub fn convert_component_mapping(mapping: ComponentMapping) -> vk::ComponentMapping {
    vk::ComponentMapping {
        r: convert_component_swizzle(mapping.r),
        g: convert_component_swizzle(mapping.g),
        b: convert_component_swizzle(mapping.b),
        a: convert_component_swizzle(mapping.a),
    }
}

pub fn convert_logic_op(op: LogicOp) -> vk::LogicOp {
    match op {
        LogicOp::Clear => vk::LogicOp::CLEAR,
        LogicOp::And => vk::LogicOp::AND,
        LogicOp::AndReverse => vk::LogicOp::AND_REVERSE,
        LogicOp::Copy => vk::LogicOp::COPY,
        LogicOp::AndInverted => vk::LogicOp::AND_INVERTED,
        LogicOp::NoOp => vk::LogicOp::NO_OP,
        LogicOp::Xor => vk::LogicOp::XOR,
        LogicOp::Or => vk::LogicOp::OR,
        LogicOp::Nor => vk::LogicOp::NOR,
        LogicOp::Equivalent => vk::LogicOp::EQUIVALENT,
        LogicOp::Invert => vk::LogicOp::INVERT,
        LogicOp::OrReverse => vk::LogicOp::OR_REVERSE,
        LogicOp::CopyInverted => vk::LogicOp::COPY_INVERTED,
        LogicOp::OrInverted => vk::LogicOp::OR_INVERTED,
        LogicOp::Nand => vk::LogicOp::NAND,
        LogicOp::Set => vk::LogicOp::SET,
    }
}

pub fn convert_vk_format(format: vk::Format) -> Format {
    match format {
        vk::Format::B8G8R8A8_UNORM => Format::B8G8R8A8_UNORM,
        vk::Format::B8G8R8A8_SRGB => Format::B8G8R8A8_SRGB,
        vk::Format::R8G8B8A8_UNORM => Format::R8G8B8A8_UNORM,
        vk::Format::R8G8B8A8_SRGB => Format::R8G8B8A8_SRGB,
        vk::Format::R8G8_UNORM => Format::R8G8_UNORM,
        vk::Format::R16G16_SFLOAT => Format::R16G16_SFLOAT,
        vk::Format::R16G16B16A16_SFLOAT => Format::R16G16B16A16_SFLOAT,
        vk::Format::B10G11R11_UFLOAT_PACK32 => Format::R11G11B10_UFLOAT,
        vk::Format::R32_SFLOAT => Format::R32_FLOAT,
        vk::Format::R32G32B32A32_SFLOAT => Format::R32G32B32A32_FLOAT,
        vk::Format::D32_SFLOAT => Format::D32_FLOAT,
        vk::Format::BC1_RGB_UNORM_BLOCK => Format::BC1_RGB_UNORM,
        vk::Format::BC1_RGB_SRGB_BLOCK => Format::BC1_RGB_SRGB,
        vk::Format::BC1_RGBA_UNORM_BLOCK => Format::BC1_RGBA_UNORM,
        vk::Format::BC1_RGBA_SRGB_BLOCK => Format::BC1_RGBA_SRGB,
        vk::Format::BC3_UNORM_BLOCK => Format::BC3_UNORM,
        vk::Format::BC3_SRGB_BLOCK => Format::BC3_SRGB,
        vk::Format::BC4_UNORM_BLOCK => Format::BC4_UNORM,
        vk::Format::BC4_SNORM_BLOCK => Format::BC4_SNORM,
        vk::Format::BC5_UNORM_BLOCK => Format::BC5_UNORM,
        vk::Format::BC5_SNORM_BLOCK => Format::BC5_SNORM,
        vk::Format::BC7_UNORM_BLOCK => Format::BC7_UNORM,
        vk::Format::BC7_SRGB_BLOCK => Format::BC7_SRGB,
        _ => Format::Undefined,
    }
}

pub fn convert_u32_format(id: u32) -> vk::Format {
    match id {
        0 => vk::Format::UNDEFINED,
        1 => vk::Format::R8G8B8A8_UNORM,
        2 => vk::Format::R8G8B8A8_SRGB,
        3 => vk::Format::B8G8R8A8_UNORM,
        4 => vk::Format::B8G8R8A8_SRGB,
        5 => vk::Format::R8G8_UNORM,
        6 => vk::Format::R16G16_SFLOAT,
        7 => vk::Format::R16G16B16A16_SFLOAT,
        8 => vk::Format::B10G11R11_UFLOAT_PACK32,
        9 => vk::Format::R32_SFLOAT, // Was R32G32B32A32_FLOAT? No, let's match types.rs
        10 => vk::Format::R32G32B32A32_SFLOAT,
        11 => vk::Format::D32_SFLOAT,
        12 => vk::Format::R32_SINT,
        13 => vk::Format::R32G32_SINT,
        14 => vk::Format::R32G32B32_SINT,
        15 => vk::Format::R32G32B32A32_SINT,
        16 => vk::Format::R32_UINT,
        17 => vk::Format::R32G32_UINT,
        18 => vk::Format::R32G32B32_UINT,
        19 => vk::Format::R32G32B32A32_UINT,
        20 => vk::Format::R32_SFLOAT,
        21 => vk::Format::R32G32_SFLOAT,
        22 => vk::Format::R32G32B32_SFLOAT,
        23 => vk::Format::R32G32B32A32_SFLOAT,
        _ => vk::Format::UNDEFINED,
    }
}

pub fn convert_u32_vertex_format(id: u32) -> vk::Format {
    match id {
        0 => vk::Format::R32_SFLOAT,
        1 => vk::Format::R32G32_SFLOAT,
        2 => vk::Format::R32G32B32_SFLOAT,
        3 => vk::Format::R32G32B32A32_SFLOAT,
        4 => vk::Format::R32_SINT,
        5 => vk::Format::R32G32_SINT,
        6 => vk::Format::R32G32B32_SINT,
        7 => vk::Format::R32G32B32A32_SINT,
        8 => vk::Format::R32_UINT,
        9 => vk::Format::R32G32_UINT,
        10 => vk::Format::R32G32B32_UINT,
        11 => vk::Format::R32G32B32A32_UINT,
        _ => vk::Format::UNDEFINED,
    }
}

pub fn convert_u32_topology(id: u32) -> vk::PrimitiveTopology {
    match id {
        0 => vk::PrimitiveTopology::POINT_LIST,
        1 => vk::PrimitiveTopology::LINE_LIST,
        2 => vk::PrimitiveTopology::LINE_STRIP,
        3 => vk::PrimitiveTopology::TRIANGLE_LIST,
        4 => vk::PrimitiveTopology::TRIANGLE_STRIP,
        5 => vk::PrimitiveTopology::TRIANGLE_FAN,
        _ => vk::PrimitiveTopology::TRIANGLE_LIST,
    }
}

pub fn convert_u32_polygon_mode(id: u32) -> vk::PolygonMode {
    match id {
        0 => vk::PolygonMode::FILL,
        1 => vk::PolygonMode::LINE,
        2 => vk::PolygonMode::POINT,
        _ => vk::PolygonMode::FILL,
    }
}

pub fn convert_u32_cull_mode(id: u32) -> vk::CullModeFlags {
    match id {
        0 => vk::CullModeFlags::NONE,
        1 => vk::CullModeFlags::FRONT,
        2 => vk::CullModeFlags::BACK,
        3 => vk::CullModeFlags::FRONT_AND_BACK,
        _ => vk::CullModeFlags::NONE,
    }
}

pub fn convert_u32_compare_op(id: u32) -> vk::CompareOp {
    match id {
        0 => vk::CompareOp::NEVER,
        1 => vk::CompareOp::LESS,
        2 => vk::CompareOp::EQUAL,
        3 => vk::CompareOp::LESS_OR_EQUAL,
        4 => vk::CompareOp::GREATER,
        5 => vk::CompareOp::NOT_EQUAL,
        6 => vk::CompareOp::GREATER_OR_EQUAL,
        7 => vk::CompareOp::ALWAYS,
        _ => vk::CompareOp::ALWAYS,
    }
}

pub fn convert_u32_blend_factor(id: u32) -> vk::BlendFactor {
    match id {
        0 => vk::BlendFactor::ZERO,
        1 => vk::BlendFactor::ONE,
        2 => vk::BlendFactor::SRC_COLOR,
        3 => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        4 => vk::BlendFactor::DST_COLOR,
        5 => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        6 => vk::BlendFactor::SRC_ALPHA,
        7 => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        8 => vk::BlendFactor::DST_ALPHA,
        9 => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        _ => vk::BlendFactor::ONE,
    }
}

pub fn convert_u32_blend_op(id: u32) -> vk::BlendOp {
    match id {
        0 => vk::BlendOp::ADD,
        1 => vk::BlendOp::SUBTRACT,
        2 => vk::BlendOp::REVERSE_SUBTRACT,
        3 => vk::BlendOp::MIN,
        4 => vk::BlendOp::MAX,
        _ => vk::BlendOp::ADD,
    }
}

pub fn convert_u32_descriptor_type(id: u32) -> vk::DescriptorType {
    match id {
        0 => vk::DescriptorType::UNIFORM_BUFFER,
        1 => vk::DescriptorType::STORAGE_BUFFER,
        2 => vk::DescriptorType::SAMPLED_IMAGE,
        3 => vk::DescriptorType::STORAGE_IMAGE,
        4 => vk::DescriptorType::SAMPLER,
        5 => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        _ => vk::DescriptorType::STORAGE_BUFFER,
    }
}

pub fn convert_u32_shader_stage_flags(bits: u32) -> vk::ShaderStageFlags {
    let mut flags = vk::ShaderStageFlags::empty();
    if bits & 0x01 != 0 {
        flags |= vk::ShaderStageFlags::VERTEX;
    }
    if bits & 0x02 != 0 {
        flags |= vk::ShaderStageFlags::FRAGMENT;
    }
    if bits & 0x04 != 0 {
        flags |= vk::ShaderStageFlags::COMPUTE;
    }
    if bits & 0x08 != 0 {
        flags |= vk::ShaderStageFlags::GEOMETRY;
    }
    if bits & 0x10 != 0 {
        flags |= vk::ShaderStageFlags::TESSELLATION_CONTROL;
    }
    if bits & 0x20 != 0 {
        flags |= vk::ShaderStageFlags::TESSELLATION_EVALUATION;
    }
    flags
}

pub fn convert_u32_stencil_op(id: u32) -> vk::StencilOp {
    match id {
        0 => vk::StencilOp::KEEP,
        1 => vk::StencilOp::ZERO,
        2 => vk::StencilOp::REPLACE,
        3 => vk::StencilOp::INCREMENT_AND_CLAMP,
        4 => vk::StencilOp::DECREMENT_AND_CLAMP,
        5 => vk::StencilOp::INVERT,
        6 => vk::StencilOp::INCREMENT_AND_WRAP,
        7 => vk::StencilOp::DECREMENT_AND_WRAP,
        _ => vk::StencilOp::KEEP,
    }
}
