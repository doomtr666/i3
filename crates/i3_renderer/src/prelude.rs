//! i3_renderer prelude
//!
//! Commonly used types for the default i3 renderer.

pub use crate::gpu_buffers::{CameraData, GpuBuffers};
pub use crate::groups::{ClusteringGroup, PostProcessGroup, sync::SyncGroup};
pub use crate::render_graph::{CommonData, DefaultRenderGraph, RenderConfig};
pub use crate::scene::{LightData, LightId, LightType, Mesh, ObjectData, ObjectId, SceneProvider};
