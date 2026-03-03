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
        KeyCode, PassContext, PassContextExt, RenderBackend, RenderBackendExt,
        RenderBackendInternal, SamplerHandle, SwapchainConfig, WindowDesc,
    },
    compiler::{CompiledGraph, FrameGraph},
    pass::{PassBuilder, RenderPass},
    pipeline::{
        Binding, BindingType, BlendFactor, BlendOp, BlendState, ColorComponentFlags, CompareOp,
        ComputePipelineCreateInfo, CullMode, DepthStencilState, GraphicsPipelineCreateInfo,
        IndexType, InputAssemblyState, LogicOp, MultisampleState, PolygonMode, PrimitiveTopology,
        PushConstantRange, RasterizationState, RenderTargetInfo, RenderTargetsInfo, ShaderModule,
        ShaderReflection, ShaderStageFlags, StencilOp, StencilOpState, VertexFormat,
        VertexInputAttribute, VertexInputBinding, VertexInputRate, VertexInputState,
    },
    types::{
        AddressMode, BufferDesc, BufferHandle, BufferUsageFlags, ComponentMapping,
        ComponentSwizzle, Filter, Format, GraphError, ImageDesc, ImageHandle, ImageUsageFlags,
        ImageViewType, MemoryType, MipmapMode, PipelineHandle, ResourceUsage, SamplerDesc,
        SwapChainImageHandle, SymbolId, WindowHandle,
    },
};
