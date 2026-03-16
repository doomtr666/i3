use crate::pipeline::{BakeContext, BakeOutput, ImportedData, Importer};
use crate::Result;
use i3_io::pipeline_asset::{PipelineHeader, PipelineType, BakeableGraphicsPipeline, PIPELINE_ASSET_TYPE};
use i3_slang::{SlangCompiler, ShaderTarget, ShaderModule};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::path::{Path, PathBuf};
use regex::Regex;

/// High-level ergonomic pipeline definition (RON).
#[derive(Debug, Serialize, Deserialize)]
pub enum ShaderSource {
    Path(String),
    Inline {
        code: String,
        #[serde(default = "default_slang_path")]
        virtual_path: String,
    },
}

fn default_slang_path() -> String {
    "inline.slang".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub name: String,
    pub shader: ShaderSource,
    pub graphics: Option<GraphicsConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphicsConfig {
    #[serde(default)]
    pub rasterization: i3_gfx::graph::pipeline::RasterizationState,
    #[serde(default)]
    pub depth_stencil: i3_gfx::graph::pipeline::DepthStencilState,
    pub targets: Vec<TargetConfig>,
    pub vertex_layout: Option<VertexLayoutConfig>,
    #[serde(default = "default_topology")]
    pub topology: i3_gfx::graph::pipeline::PrimitiveTopology,
    #[serde(default)]
    pub multisample: i3_gfx::graph::pipeline::MultisampleState,
    #[serde(default)]
    pub logic_op: Option<i3_gfx::graph::pipeline::LogicOp>,
    pub depth_stencil_format: Option<i3_gfx::graph::types::Format>,
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            rasterization: Default::default(),
            depth_stencil: Default::default(),
            targets: Vec::new(),
            vertex_layout: None,
            topology: default_topology(),
            multisample: Default::default(),
            logic_op: None,
            depth_stencil_format: None,
        }
    }
}

fn default_topology() -> i3_gfx::graph::pipeline::PrimitiveTopology {
    i3_gfx::graph::pipeline::PrimitiveTopology::TriangleList
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TargetConfig {
    pub format: i3_gfx::graph::types::Format,
    #[serde(default)]
    pub write_mask: i3_gfx::graph::pipeline::ColorComponentFlags,
    #[serde(default)]
    pub blend: Option<i3_gfx::graph::pipeline::BlendState>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VertexLayoutConfig {
    pub bindings: Vec<i3_gfx::graph::pipeline::VertexInputBinding>,
    pub attributes: Vec<i3_gfx::graph::pipeline::VertexInputAttribute>,
}

impl GraphicsConfig {
    pub fn to_bakeable(&self) -> BakeableGraphicsPipeline {
        let mut bakeable = BakeableGraphicsPipeline {
            rasterization: i3_io::pipeline_asset::BakeableRasterizationState {
                depth_clamp_enable: self.rasterization.depth_clamp_enable as u32,
                rasterizer_discard_enable: self.rasterization.rasterizer_discard_enable as u32,
                polygon_mode: self.rasterization.polygon_mode as u32,
                cull_mode: self.rasterization.cull_mode as u32,
                depth_bias_enable: self.rasterization.depth_bias_enable as u32,
                depth_bias_constant_factor: self.rasterization.depth_bias_constant_factor,
                depth_bias_clamp: self.rasterization.depth_bias_clamp,
                depth_bias_slope_factor: self.rasterization.depth_bias_slope_factor,
                line_width: self.rasterization.line_width,
            },
            depth_stencil: i3_io::pipeline_asset::BakeableDepthStencilState {
                depth_test_enable: self.depth_stencil.depth_test_enable as u32,
                depth_write_enable: self.depth_stencil.depth_write_enable as u32,
                depth_compare_op: self.depth_stencil.depth_compare_op as u32,
                stencil_test_enable: self.depth_stencil.stencil_test_enable as u32,
                front: i3_io::pipeline_asset::BakeableStencilOpState {
                    fail_op: self.depth_stencil.front.fail_op as u32,
                    pass_op: self.depth_stencil.front.pass_op as u32,
                    depth_fail_op: self.depth_stencil.front.depth_fail_op as u32,
                    compare_op: self.depth_stencil.front.compare_op as u32,
                    compare_mask: self.depth_stencil.front.compare_mask,
                    write_mask: self.depth_stencil.front.write_mask,
                    reference: self.depth_stencil.front.reference,
                },
                back: i3_io::pipeline_asset::BakeableStencilOpState {
                    fail_op: self.depth_stencil.back.fail_op as u32,
                    pass_op: self.depth_stencil.back.pass_op as u32,
                    depth_fail_op: self.depth_stencil.back.depth_fail_op as u32,
                    compare_op: self.depth_stencil.back.compare_op as u32,
                    compare_mask: self.depth_stencil.back.compare_mask,
                    write_mask: self.depth_stencil.back.write_mask,
                    reference: self.depth_stencil.back.reference,
                },
                depth_bounds_test_enable: self.depth_stencil.depth_bounds_test_enable as u32,
                min_depth_bounds: self.depth_stencil.min_depth_bounds,
                max_depth_bounds: self.depth_stencil.max_depth_bounds,
            },
            topology: self.topology as u32,
            primitive_restart_enable: 0,
            patch_control_points: 0,
            sample_count: self.multisample.sample_count as u32,
            sample_shading_enable: self.multisample.sample_shading_enable as u32,
            alpha_to_coverage_enable: self.multisample.alpha_to_coverage_enable as u32,
            logic_op_enable: self.logic_op.is_some() as u32,
            logic_op: self.logic_op.map(|op| op as u32).unwrap_or(0),
            color_target_count: self.targets.len() as u32,
            color_targets: [i3_io::pipeline_asset::BakeableRenderTarget {
                format: 0,
                blend_enable: 0,
                src_color_factor: 0,
                dst_color_factor: 0,
                color_op: 0,
                src_alpha_factor: 0,
                dst_alpha_factor: 0,
                alpha_op: 0,
                write_mask: 0,
            }; 8],
            depth_stencil_format: self.depth_stencil_format.map(|f| f as u32).unwrap_or(0),
            vertex_binding_count: 0,
            vertex_bindings: [i3_io::pipeline_asset::BakeableVertexBinding {
                binding: 0,
                stride: 0,
                input_rate: 0,
            }; 8],
            vertex_attribute_count: 0,
            vertex_attributes: [i3_io::pipeline_asset::BakeableVertexAttribute {
                location: 0,
                binding: 0,
                format: 0,
                offset: 0,
            }; 16],
        };

        // Fill color targets
        for (i, target) in self.targets.iter().enumerate().take(8) {
            bakeable.color_targets[i] = i3_io::pipeline_asset::BakeableRenderTarget {
                format: target.format as u32,
                blend_enable: target.blend.is_some() as u32,
                src_color_factor: target.blend.map(|b| b.src_color_factor as u32).unwrap_or(0),
                dst_color_factor: target.blend.map(|b| b.dst_color_factor as u32).unwrap_or(0),
                color_op: target.blend.map(|b| b.color_op as u32).unwrap_or(0),
                src_alpha_factor: target.blend.map(|b| b.src_alpha_factor as u32).unwrap_or(0),
                dst_alpha_factor: target.blend.map(|b| b.dst_alpha_factor as u32).unwrap_or(0),
                alpha_op: target.blend.map(|b| b.alpha_op as u32).unwrap_or(0),
                write_mask: target.write_mask.bits() as u32,
            };
        }

        // Depth stencil format
        // Logic: if depth_test_enable is true, we should probably have a format?
        // But the user proposed a separate `format` field in `depth_stencil`.

        // Vertex layout
        if let Some(layout) = &self.vertex_layout {
            bakeable.vertex_binding_count = layout.bindings.len() as u32;
            for (i, binding) in layout.bindings.iter().enumerate().take(8) {
                bakeable.vertex_bindings[i] = i3_io::pipeline_asset::BakeableVertexBinding {
                    binding: binding.binding,
                    stride: binding.stride,
                    input_rate: binding.input_rate as u32,
                };
            }
            bakeable.vertex_attribute_count = layout.attributes.len() as u32;
            for (i, attr) in layout.attributes.iter().enumerate().take(16) {
                bakeable.vertex_attributes[i] = i3_io::pipeline_asset::BakeableVertexAttribute {
                    location: attr.location,
                    binding: attr.binding,
                    format: attr.format as u32,
                    offset: attr.offset,
                };
            }
        }

        bakeable
    }
}

/// Intermediate data for a pipeline.
pub struct ImportedPipelineData {
    pub source_path: PathBuf,
    pub config: PipelineConfig,
    pub shader_module: ShaderModule,
}

impl ImportedData for ImportedPipelineData {
    fn source_path(&self) -> &Path {
        &self.source_path
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Clone, Copy)]
pub struct PipelineImporter;

impl PipelineImporter {
    pub fn new() -> Self {
        Self
    }
}

impl Importer for PipelineImporter {
    fn name(&self) -> &str {
        "PipelineImporter"
    }

    fn source_extensions(&self) -> &[&str] {
        &["i3p"]
    }

    fn import(&self, source_path: &Path) -> Result<Box<dyn ImportedData>> {
        let content = std::fs::read_to_string(source_path)
            .map_err(|e| crate::error::BakerError::Os { path: source_path.to_path_buf(), source: e })?;
        
        let config: PipelineConfig = ron::from_str(&content)
            .map_err(|e| crate::error::BakerError::Pipeline(format!("Failed to parse RON {}: {}", source_path.display(), e)))?;

        let compiler = SlangCompiler::new()
            .map_err(|e| crate::error::BakerError::Pipeline(e))?;

        let shader_module = match &config.shader {
            ShaderSource::Path(rel_path) => {
                let shader_full_path = source_path.parent().unwrap().join(rel_path);
                
                let module_name = shader_full_path.file_stem().expect("Invalid shader path").to_str().unwrap();
                let shader_dir = shader_full_path.parent().unwrap().to_str().unwrap();

                compiler.compile_file(module_name, ShaderTarget::Spirv, &[shader_dir])
                    .map_err(|e| crate::error::BakerError::Pipeline(e))?
            }
            ShaderSource::Inline { code, virtual_path } => {
                compiler.compile_inline("inline_shader", virtual_path, code, ShaderTarget::Spirv)
                    .map_err(|e| crate::error::BakerError::Pipeline(e))?
            }
        };

        Ok(Box::new(ImportedPipelineData {
            source_path: source_path.to_path_buf(),
            config,
            shader_module,
        }))
    }

    fn get_dependencies(&self, source_path: &Path) -> Result<Vec<PathBuf>> {
        let content = std::fs::read_to_string(source_path)
            .map_err(|e| crate::error::BakerError::Os { path: source_path.to_path_buf(), source: e })?;
        
        let config: PipelineConfig = ron::from_str(&content)
            .map_err(|e| crate::error::BakerError::Pipeline(format!("Failed to parse RON: {}", e)))?;

        let mut deps = Vec::new();
        match &config.shader {
            ShaderSource::Path(rel_path) => {
                let shader_path = source_path.parent().unwrap().join(rel_path);
                if shader_path.exists() {
                    deps.push(shader_path.clone());
                    // Scan the shader file for includes
                    if let Ok(shader_content) = std::fs::read_to_string(&shader_path) {
                        deps.extend(scan_includes(&shader_path, &shader_content));
                    }
                }
            }
            ShaderSource::Inline { code, .. } => {
                deps.extend(scan_includes(source_path, code));
            }
        }
        Ok(deps)
    }

    fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        let data = data.as_any().downcast_ref::<ImportedPipelineData>().unwrap();
        
        let bytecode = &data.shader_module.bytecode;
        
        // Serialize reflection
        let reflection_bin = postcard::to_allocvec(&data.shader_module.reflection)
            .map_err(|e| crate::error::BakerError::Pipeline(format!("Postcard serialize failed: {}", e)))?;

        let pipeline_type = if data.config.graphics.is_some() {
            PipelineType::GRAPHICS
        } else {
            PipelineType::COMPUTE
        };

        // Header
        let header = PipelineHeader {
            pipeline_type,
            state_offset: std::mem::size_of::<PipelineHeader>() as u32,
            state_size: if pipeline_type == PipelineType::GRAPHICS { std::mem::size_of::<BakeableGraphicsPipeline>() as u32 } else { 0 },
            reflection_offset: (std::mem::size_of::<PipelineHeader>() + if pipeline_type == PipelineType::GRAPHICS { std::mem::size_of::<BakeableGraphicsPipeline>() } else { 0 }) as u32,
            reflection_size: reflection_bin.len() as u32,
            bytecode_offset: (std::mem::size_of::<PipelineHeader>() + if pipeline_type == PipelineType::GRAPHICS { std::mem::size_of::<BakeableGraphicsPipeline>() } else { 0 } + reflection_bin.len()) as u32,
            bytecode_size: bytecode.len() as u32,
            _reserved: [0; 9],
        };

        let mut final_data = bytemuck::bytes_of(&header).to_vec();
        
        if let Some(graphics) = &data.config.graphics {
            let state = graphics.to_bakeable();
            final_data.extend_from_slice(bytemuck::bytes_of(&state));
        }

        final_data.extend_from_slice(&reflection_bin);
        final_data.extend_from_slice(&bytecode);

        let asset_id = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, data.config.name.as_bytes());
        
        Ok(vec![BakeOutput {
            asset_id,
            asset_type: PIPELINE_ASSET_TYPE,
            data: final_data,
            name: data.config.name.clone(),
        }])
    }
}

fn scan_includes(base_path: &Path, content: &str) -> Vec<PathBuf> {
    let mut deps = Vec::new();
    let re = Regex::new(r#"(?m)^\s*#include\s+["<]([^">]+)[">]"#).unwrap();
    let dir = base_path.parent().unwrap();

    for cap in re.captures_iter(content) {
        let include_path = &cap[1];
        let full_path = dir.join(include_path);
        if full_path.exists() {
            deps.push(full_path);
        }
    }
    deps
}
