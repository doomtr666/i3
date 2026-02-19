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
        BackendBuffer, BackendImage, BackendPipeline, DescriptorBufferInfo, DescriptorImageInfo,
        DescriptorImageLayout, DescriptorSetHandle, DescriptorWrite, DeviceInfo, DeviceType, Event,
        KeyCode, PassContext, PassDescriptor, RenderBackend, SamplerHandle, SwapchainConfig,
        WindowDesc,
    },
    compiler::{CompiledGraph, FrameGraph},
    pass::{Node, PassBuilder},
    pipeline::{
        Binding, BindingType, BlendFactor, BlendOp, BlendState, ColorComponentFlags, CompareOp,
        CullMode, DepthStencilState, FrontFace, GraphicsPipelineCreateInfo, IndexType,
        InputAssemblyState, LogicOp, MultisampleState, PolygonMode, PrimitiveTopology,
        PushConstantRange, RasterizationState, RenderTargetInfo, RenderTargetsInfo, ShaderModule,
        ShaderReflection, ShaderStageFlags, StencilOp, StencilOpState, VertexFormat,
        VertexInputAttribute, VertexInputBinding, VertexInputRate, VertexInputState,
    },
    types::{
        AddressMode, BufferDesc, BufferHandle, BufferUsageFlags, ComponentMapping,
        ComponentSwizzle, Filter, Format, ImageDesc, ImageHandle, ImageUsageFlags, ImageViewType,
        MemoryType, MipmapMode, PassDomain, PipelineHandle, ResourceUsage, SamplerDesc,
        SwapChainImageHandle, SymbolId, WindowHandle,
    },
};
