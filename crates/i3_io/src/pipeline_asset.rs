//! Baked pipeline asset format.
//!
//! Binary-compatible with GraphicsPipelineCreateInfo for zero-copy loading.

use crate::asset::Asset;
use crate::{AssetHeader, Result};
use bytemuck::{Pod, Zeroable};
use uuid::{Uuid, uuid};

/// UUID for pipeline assets
pub const PIPELINE_ASSET_TYPE: Uuid = uuid!("882a1762-b9e7-4f4a-9c7a-55e76a6d6562");

/// Type of pipeline (Graphics or Compute).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct PipelineType(pub u32);

impl PipelineType {
    pub const GRAPHICS: PipelineType = PipelineType(0);
    pub const COMPUTE: PipelineType = PipelineType(1);
}

/// Header for a baked pipeline asset (64 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PipelineHeader {
    /// Pipeline type (Graphics/Compute).
    pub pipeline_type: PipelineType,
    /// Offset to the pipeline state data (BakeableGraphicsPipeline or equivalent).
    pub state_offset: u32,
    /// Size of the state data.
    pub state_size: u32,
    /// Offset to the shader reflection data (serialized).
    pub reflection_offset: u32,
    /// Size of the reflection data.
    pub reflection_size: u32,
    /// Offset to the shader bytecode (SPIR-V).
    pub bytecode_offset: u32,
    /// Size of the bytecode.
    pub bytecode_size: u32,
    /// Reserved for future use.
    pub _reserved: [u32; 9],
}

/// Binary-compatible version of GraphicsPipelineCreateInfo fields.
/// All enums are replaced with u32.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BakeableRasterizationState {
    pub depth_clamp_enable: u32,
    pub rasterizer_discard_enable: u32,
    pub polygon_mode: u32,
    pub cull_mode: u32,
    pub depth_bias_enable: u32,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub line_width: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BakeableDepthStencilState {
    pub depth_test_enable: u32,
    pub depth_write_enable: u32,
    pub depth_compare_op: u32,
    pub stencil_test_enable: u32,
    pub front: BakeableStencilOpState,
    pub back: BakeableStencilOpState,
    pub depth_bounds_test_enable: u32,
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BakeableStencilOpState {
    pub fail_op: u32,
    pub pass_op: u32,
    pub depth_fail_op: u32,
    pub compare_op: u32,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BakeableRenderTarget {
    pub format: u32,
    pub blend_enable: u32,
    pub src_color_factor: u32,
    pub dst_color_factor: u32,
    pub color_op: u32,
    pub src_alpha_factor: u32,
    pub dst_alpha_factor: u32,
    pub alpha_op: u32,
    pub write_mask: u32,
}

/// Flattened representation of a vertex attribute.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BakeableVertexAttribute {
    pub location: u32,
    pub binding: u32,
    pub format: u32,
    pub offset: u32,
}

/// Flattened representation of a vertex binding.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BakeableVertexBinding {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: u32,
}

/// Complete pipeline state for a graphics pipeline.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BakeableGraphicsPipeline {
    pub rasterization: BakeableRasterizationState,
    pub depth_stencil: BakeableDepthStencilState,
    pub topology: u32,
    pub primitive_restart_enable: u32,
    pub patch_control_points: u32,
    pub sample_count: u32,
    pub sample_shading_enable: u32,
    pub alpha_to_coverage_enable: u32,
    pub logic_op_enable: u32,
    pub logic_op: u32,
    pub color_target_count: u32,
    pub color_targets: [BakeableRenderTarget; 8],
    pub depth_stencil_format: u32,
    pub vertex_binding_count: u32,
    pub vertex_bindings: [BakeableVertexBinding; 8],
    pub vertex_attribute_count: u32,
    pub vertex_attributes: [BakeableVertexAttribute; 16],
}

/// Loaded pipeline asset.
pub struct PipelineAsset {
    pub header: PipelineHeader,
    pub type_info: PipelineType,
    pub state: Option<BakeableGraphicsPipeline>,
    pub reflection_data: Vec<u8>,
    pub bytecode: Vec<u8>,
}

impl Asset for PipelineAsset {
    const ASSET_TYPE_ID: [u8; 16] = *PIPELINE_ASSET_TYPE.as_bytes();

    fn load(_header: &AssetHeader, data: &[u8]) -> Result<Self> {
        if data.len() < std::mem::size_of::<PipelineHeader>() {
            return Err(crate::IoError::InvalidData {
                message: "Pipeline data too small for header".into(),
            });
        }

        let header: PipelineHeader = bytemuck::pod_read_unaligned(&data[..std::mem::size_of::<PipelineHeader>()]);

        let state = if header.pipeline_type == PipelineType::GRAPHICS {
            if data.len() < (header.state_offset + header.state_size) as usize {
                 return Err(crate::IoError::InvalidData {
                    message: "Pipeline state data out of bounds".into(),
                });
            }
            let state_data = &data[header.state_offset as usize..(header.state_offset + header.state_size) as usize];
            Some(bytemuck::pod_read_unaligned(state_data))
        } else {
            None
        };

        if data.len() < (header.reflection_offset + header.reflection_size) as usize {
             return Err(crate::IoError::InvalidData {
                message: "Reflection data out of bounds".into(),
            });
        }
        let reflection_data = data[header.reflection_offset as usize..(header.reflection_offset + header.reflection_size) as usize].to_vec();

        if data.len() < (header.bytecode_offset + header.bytecode_size) as usize {
             return Err(crate::IoError::InvalidData {
                message: "Bytecode data out of bounds".into(),
            });
        }
        let bytecode = data[header.bytecode_offset as usize..(header.bytecode_offset + header.bytecode_size) as usize].to_vec();

        Ok(PipelineAsset {
            header,
            type_info: header.pipeline_type,
            state,
            reflection_data,
            bytecode,
        })
    }
}
