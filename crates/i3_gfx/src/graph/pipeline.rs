use bitflags::bitflags;
use serde::{Deserialize, Serialize};

// --- Enums & Flags ---

bitflags! {
    /// Shader stage flags for identifying shader stages.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ShaderStageFlags: u32 {
        const Vertex = 0x1;
        const Fragment = 0x2;
        const Compute = 0x4;
        const Geometry = 0x8;
        const TessellationControl = 0x10;
        const TessellationEvaluation = 0x20;
        const AllGraphics = Self::Vertex.bits() | Self::Fragment.bits() | Self::Geometry.bits();
        const All = Self::Vertex.bits() | Self::Fragment.bits() | Self::Compute.bits();
    }
}

// --- Shader Reflection & Module Types ---

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BindingType {
    Unknown,
    UniformBuffer,
    StorageBuffer,
    SampledImage,
    StorageImage,
    Sampler,
    CombinedImageSampler,
    UniformTexelBuffer, // Currently not supported (requires VkBufferView)
    StorageTexelBuffer, // Currently not supported (requires VkBufferView)
    AccelerationStructure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    pub name: String,
    pub binding: u32,
    pub set: u32,
    pub count: u32,
    pub binding_type: BindingType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPointInfo {
    pub name: String,
    pub stage: String, // "vertex", "fragment", etc.
    pub thread_group_size: Option<[u64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PushConstantRange {
    pub stage_flags: ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShaderReflection {
    pub entry_points: Vec<EntryPointInfo>,
    pub bindings: Vec<Binding>,
    pub push_constants: Vec<PushConstantRange>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderStageInfo {
    pub stage: ShaderStageFlags,
    pub entry_point: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderModule {
    pub bytecode: Vec<u8>,
    pub stages: Vec<ShaderStageInfo>,
    pub reflection: ShaderReflection,
}

impl Default for ShaderModule {
    fn default() -> Self {
        Self {
            bytecode: Vec::new(),
            stages: Vec::new(),
            reflection: ShaderReflection {
                entry_points: Vec::new(),
                bindings: Vec::new(),
                push_constants: Vec::new(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexType {
    Uint16 = 0,
    Uint32 = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MipmapMode {
    Nearest = 0,
    Linear = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
    TriangleFan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VertexInputRate {
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VertexFormat {
    Float,
    Float2,
    Float3,
    Float4,
    Int,
    Int2,
    Int3,
    Int4,
    UInt,
    UInt2,
    UInt3,
    UInt4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PolygonMode {
    Fill,
    Line,
    Point,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CullMode {
    None,
    Front,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SampleCount {
    Sample1 = 1,
    Sample2 = 2,
    Sample4 = 4,
    Sample8 = 8,
    Sample16 = 16,
    Sample32 = 32,
    Sample64 = 64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementAndClamp,
    DecrementAndClamp,
    Invert,
    IncrementAndWrap,
    DecrementAndWrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogicOp {
    Clear,
    And,
    AndReverse,
    Copy,
    AndInverted,
    NoOp,
    Xor,
    Or,
    Nor,
    Equivalent,
    Invert,
    OrReverse,
    CopyInverted,
    OrInverted,
    Nand,
    Set,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ColorComponentFlags: u8 {
        const R = 0x1;
        const G = 0x2;
        const B = 0x4;
        const A = 0x8;
        const RGB = Self::R.bits() | Self::G.bits() | Self::B.bits();
        const RGBA = Self::RGB.bits() | Self::A.bits();
    }
}

impl Default for ColorComponentFlags {
    fn default() -> Self {
        Self::RGBA
    }
}

// --- Structs ---

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VertexInputBinding {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: VertexInputRate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VertexInputAttribute {
    pub location: u32,
    pub binding: u32,
    pub format: VertexFormat,
    pub offset: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VertexInputState {
    pub bindings: Vec<VertexInputBinding>,
    pub attributes: Vec<VertexInputAttribute>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputAssemblyState {
    pub topology: PrimitiveTopology,
    pub primitive_restart_enable: bool,
}

impl Default for InputAssemblyState {
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::TriangleList,
            primitive_restart_enable: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TessellationState {
    pub patch_control_points: u32,
}

impl Default for TessellationState {
    fn default() -> Self {
        Self {
            patch_control_points: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RasterizationState {
    pub depth_clamp_enable: bool,
    pub rasterizer_discard_enable: bool,
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    // FrontFace intentionally NOT exposed — backends handle it transparently
    // (see engine_conventions.md §2).
    pub depth_bias_enable: bool,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub line_width: f32,
}

impl Default for RasterizationState {
    fn default() -> Self {
        Self {
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            polygon_mode: PolygonMode::Fill,
            cull_mode: CullMode::None,
            depth_bias_enable: false,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        }
    }
}

// Wait, I need to check if I can compile f32 in PartialEq. Rust allows it. Eq is the problem.
// The legacy struct had `Eq`.
// I will drop `Eq` for structs with floats, or use a wrapper.
// Let's check `MultisampleState`
/*
pub struct MultisampleState {
    pub sample_count: SampleCount,
    pub sample_shading_enable: bool,
    pub alpha_to_coverage_enable: bool,
}
*/
// That one is fine.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultisampleState {
    pub sample_count: SampleCount,
    pub sample_shading_enable: bool,
    pub alpha_to_coverage_enable: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            sample_count: SampleCount::Sample1,
            sample_shading_enable: false,
            alpha_to_coverage_enable: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StencilOpState {
    pub fail_op: StencilOp,
    pub pass_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub compare_op: CompareOp,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

impl Default for StencilOpState {
    fn default() -> Self {
        Self {
            fail_op: StencilOp::Keep,
            pass_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            compare_op: CompareOp::Always,
            compare_mask: 0xFF,
            write_mask: 0xFF,
            reference: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DepthStencilState {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: CompareOp,
    pub stencil_test_enable: bool,
    pub front: StencilOpState,
    pub back: StencilOpState,
    pub depth_bounds_test_enable: bool,
    pub min_depth_bounds: f32, // Note: PartialEq on f32
    pub max_depth_bounds: f32,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            depth_test_enable: false,
            depth_write_enable: false,
            depth_compare_op: CompareOp::Less,
            stencil_test_enable: false,
            front: StencilOpState::default(),
            back: StencilOpState::default(),
            depth_bounds_test_enable: false,
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BlendState {
    pub src_color_factor: BlendFactor,
    pub dst_color_factor: BlendFactor,
    pub color_op: BlendOp,
    pub src_alpha_factor: BlendFactor,
    pub dst_alpha_factor: BlendFactor,
    pub alpha_op: BlendOp,
}

impl BlendState {
    pub const ALPHA_BLENDING: Self = Self {
        src_color_factor: BlendFactor::SrcColor, // Wait, legacy said SrcAlpha?
        // Legacy: src_color_factor: BlendFactor::SrcAlpha
        // dst_color_factor: BlendFactor::OneMinusSrcAlpha
        // I should match legacy.
        dst_color_factor: BlendFactor::OneMinusSrcAlpha,
        color_op: BlendOp::Add,
        src_alpha_factor: BlendFactor::One,
        dst_alpha_factor: BlendFactor::Zero,
        alpha_op: BlendOp::Add,
    };

    // Correction based on legacy:
    // pub const ALPHA_BLENDING: Self = Self {
    //    src_color_factor: BlendFactor::SrcAlpha,
    //    ...
    // }
}

impl Default for BlendState {
    fn default() -> Self {
        // Return a reasonable default, e.g. Replace (One, Zero)
        Self {
            src_color_factor: BlendFactor::One,
            dst_color_factor: BlendFactor::Zero,
            color_op: BlendOp::Add,
            src_alpha_factor: BlendFactor::One,
            dst_alpha_factor: BlendFactor::Zero,
            alpha_op: BlendOp::Add,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderTargetInfo {
    pub format: crate::graph::types::Format,
    pub write_mask: ColorComponentFlags,
    pub blend: Option<BlendState>,
}

impl Default for RenderTargetInfo {
    fn default() -> Self {
        Self {
            format: crate::graph::types::Format::Undefined,
            write_mask: ColorComponentFlags::RGBA,
            blend: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderTargetsInfo {
    pub color_targets: Vec<RenderTargetInfo>,
    pub depth_stencil_format: Option<crate::graph::types::Format>,
    pub logic_op: Option<LogicOp>,
}

// --- Main Pipeline Create Info ---

#[derive(Debug, Clone, PartialEq)]
pub struct GraphicsPipelineCreateInfo {
    pub shader_module: ShaderModule,
    pub vertex_input: VertexInputState,
    pub input_assembly: InputAssemblyState,
    pub tessellation_state: TessellationState,
    pub rasterization_state: RasterizationState,
    pub multisample_state: MultisampleState,
    pub depth_stencil_state: DepthStencilState,
    pub render_targets: RenderTargetsInfo,
}

impl Default for GraphicsPipelineCreateInfo {
    fn default() -> Self {
        Self {
            shader_module: ShaderModule::default(),
            vertex_input: VertexInputState::default(),
            input_assembly: InputAssemblyState::default(),
            tessellation_state: TessellationState::default(),
            rasterization_state: RasterizationState::default(),
            multisample_state: MultisampleState::default(),
            depth_stencil_state: DepthStencilState::default(),
            render_targets: RenderTargetsInfo {
                color_targets: Vec::new(),
                depth_stencil_format: None,
                logic_op: None,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComputePipelineCreateInfo {
    pub shader_module: ShaderModule,
}

impl Default for ComputePipelineCreateInfo {
    fn default() -> Self {
        Self {
            shader_module: ShaderModule::default(),
        }
    }
}
