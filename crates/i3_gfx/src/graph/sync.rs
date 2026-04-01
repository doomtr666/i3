use std::collections::HashMap;
use bitflags::bitflags;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct AccessFlags: u64 {
        const NONE = 0;
        const INDIRECT_COMMAND_READ = 1 << 0;
        const INDEX_READ = 1 << 1;
        const VERTEX_ATTRIBUTE_READ = 1 << 2;
        const UNIFORM_READ = 1 << 3;
        const INPUT_ATTACHMENT_READ = 1 << 4;
        const SHADER_READ = 1 << 5;
        const SHADER_WRITE = 1 << 6;
        const COLOR_ATTACHMENT_READ = 1 << 7;
        const COLOR_ATTACHMENT_WRITE = 1 << 8;
        const DEPTH_STENCIL_ATTACHMENT_READ = 1 << 9;
        const DEPTH_STENCIL_ATTACHMENT_WRITE = 1 << 10;
        const TRANSFER_READ = 1 << 11;
        const TRANSFER_WRITE = 1 << 12;
        const HOST_READ = 1 << 13;
        const HOST_WRITE = 1 << 14;
        const MEMORY_READ = 1 << 15;
        const MEMORY_WRITE = 1 << 16;
        const ACCELERATION_STRUCTURE_READ = 1 << 17;
        const ACCELERATION_STRUCTURE_WRITE = 1 << 18;
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct StageFlags: u64 {
        const TOP_OF_PIPE = 1 << 0;
        const DRAW_INDIRECT = 1 << 1;
        const VERTEX_INPUT = 1 << 2;
        const VERTEX_SHADER = 1 << 3;
        const TESSELLATION_CONTROL_SHADER = 1 << 4;
        const TESSELLATION_EVALUATION_SHADER = 1 << 5;
        const GEOMETRY_SHADER = 1 << 6;
        const FRAGMENT_SHADER = 1 << 7;
        const EARLY_FRAGMENT_TESTS = 1 << 8;
        const LATE_FRAGMENT_TESTS = 1 << 9;
        const COLOR_ATTACHMENT_OUTPUT = 1 << 10;
        const COMPUTE_SHADER = 1 << 11;
        const TRANSFER = 1 << 12;
        const BOTTOM_OF_PIPE = 1 << 13;
        const HOST = 1 << 14;
        const ALL_GRAPHICS = 1 << 15;
        const ALL_COMMANDS = 1 << 16;
        const COPY = 1 << 17;
        const RESOLVE = 1 << 18;
        const BLIT = 1 << 19;
        const CLEAR = 1 << 20;
        const INDEX_INPUT = 1 << 21;
        const VERTEX_ATTRIBUTE_INPUT = 1 << 22;
        const PRE_RASTERIZATION_SHADERS = 1 << 23;
        const RAY_TRACING_SHADER = 1 << 24;
        const ACCELERATION_STRUCTURE_BUILD = 1 << 25;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ImageLayout {
    #[default]
    Undefined,
    General,
    ColorAttachmentOptimal,
    DepthStencilAttachmentOptimal,
    DepthStencilReadOnlyOptimal,
    ShaderReadOnlyOptimal,
    TransferSrcOptimal,
    TransferDstOptimal,
    Preinitialized,
    DepthReadOnlyStencilAttachmentOptimal,
    DepthAttachmentStencilReadOnlyOptimal,
    DepthAttachmentOptimal,
    DepthReadOnlyOptimal,
    StencilAttachmentOptimal,
    StencilReadOnlyOptimal,
    PresentSrc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    Image,
    Buffer,
    AccelStruct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ResourceState {
    pub layout: ImageLayout,
    pub access: AccessFlags,
    pub stage: StageFlags,
    pub queue_family: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransitionKind {
    Regular,
    Release,
    Acquire,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractTransition {
    pub resource_id: u64,
    pub resource_kind: ResourceKind,
    pub old_state: ResourceState,
    pub new_state: ResourceState,
    pub kind: TransitionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadOp {
    Load,
    Clear,
    DontCare,
}

#[derive(Debug, Clone, Default)]
pub struct PassSyncData {
    pub pre_transitions: Vec<AbstractTransition>,
    pub post_transitions: Vec<AbstractTransition>,
    pub load_ops: HashMap<u64, LoadOp>,
}

#[derive(Debug, Clone, Default)]
pub struct SyncPlan {
    pub passes: Vec<PassSyncData>,
    pub final_states: HashMap<u64, ResourceState>,
}
