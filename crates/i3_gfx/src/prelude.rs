//! i3_gfx prelude
//!
//! This module re-exports the most commonly used types and traits for the i3_gfx engine.
//! Use this to avoid "import hell" in your render passes and tests.
//!
//! ```rust
//! use i3_gfx::prelude::*;
//! ```

pub use crate::graph::{
    backend::{GraphicsPipelineDesc, PassContext, RenderBackend},
    compiler::{CompiledGraph, FrameGraph},
    pass::{Node, PassBuilder},
    types::{
        BufferDesc, BufferHandle, ImageDesc, ImageHandle, PassDomain, PipelineHandle,
        ResourceUsage, SwapChainImageHandle, SymbolId, WindowHandle,
    },
};
