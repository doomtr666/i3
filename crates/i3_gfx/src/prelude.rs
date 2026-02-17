//! i3_gfx prelude
//!
//! This module re-exports the most commonly used types and traits for the i3_gfx engine.
//! Use this to avoid "import hell" in your render passes and tests.
//!
//! ```rust
//! use i3_gfx::prelude::*;
//! ```

pub use crate::graph::{
    backend::{
        DeviceInfo, DeviceType, Event, PassContext, RenderBackend, SwapchainConfig, WindowDesc,
    },
    compiler::{CompiledGraph, FrameGraph},
    pass::{Node, PassBuilder},
    pipeline::{
        Binding, BindingType, BlendFactor, BlendOp, BlendState, BufferUsageFlags,
        ColorComponentFlags, CompareOp, CullMode, DepthStencilState, FrontFace,
        GraphicsPipelineCreateInfo, MultisampleState, PolygonMode, PrimitiveTopology,
        PushConstantRange, RasterizationState, RenderTargetInfo, RenderTargetsInfo, ShaderModule,
        ShaderReflection, ShaderStageFlags, StencilOp, StencilOpState, VertexFormat,
        VertexInputRate,
    },
    types::{
        BufferDesc, BufferHandle, Format, ImageDesc, ImageHandle, PassDomain, PipelineHandle,
        ResourceUsage, SwapChainImageHandle, SymbolId, WindowHandle,
    },
};
