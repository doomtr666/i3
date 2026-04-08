// i3fx_slang - Slang shader compiler wrapper for i3fx
//
// This crate provides a clean, ergonomic wrapper around the Slang shader compiler
// for use in the i3_gfx rendering framework.

use i3_gfx::graph::pipeline::ShaderStageFlags;
use shader_slang as slang;

// Re-export types from i3_gfx for convenience
pub use i3_gfx::graph::pipeline::{
    Binding, BindingType, EntryPointInfo, PushConstantRange, ShaderModule, ShaderReflection,
    ShaderStageInfo,
};

pub mod prelude {
    pub use crate::{
        Binding, BindingType, EntryPointInfo, PushConstantRange, ShaderModule, ShaderReflection,
        ShaderStageInfo, ShaderTarget, SlangCompiler,
    };
}

/// Shader compilation target (SPIRV, DXIL, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderTarget {
    Spirv,
}

impl ShaderTarget {
    pub fn to_slang_target(&self) -> slang::CompileTarget {
        match self {
            ShaderTarget::Spirv => slang::CompileTarget::Spirv,
        }
    }

    pub fn default_profile(&self, global_session: &slang::GlobalSession) -> slang::ProfileID {
        match self {
            ShaderTarget::Spirv => global_session.find_profile("glsl_460"),
        }
    }
}

/// Slang compiler wrapper
pub struct SlangCompiler {
    global_session: slang::GlobalSession,
}

impl SlangCompiler {
    /// Create a new Slang compiler instance
    pub fn new() -> Result<Self, String> {
        let global_session = slang::GlobalSession::new()
            .ok_or_else(|| "Failed to create Slang global session".to_string())?;

        Ok(Self { global_session })
    }

    /// Extract reflection data from linked program
    fn extract_reflection(
        &self,
        linked_program: &slang::ComponentType,
    ) -> Result<ShaderReflection, String> {
        let reflection = linked_program
            .layout(0)
            .map_err(|e| format!("Failed to get reflection layout: {:?}", e))?;

        // Extract entry points
        let mut entry_points = Vec::new();
        let mut all_stages = ShaderStageFlags::empty();

        for entry_point_ref in reflection.entry_points() {
            let name = entry_point_ref.name().unwrap_or("unknown").to_string();
            let stage = entry_point_ref.stage();
            let stage_name = match stage {
                slang::Stage::Vertex => "vertex",
                slang::Stage::Fragment => "fragment",
                slang::Stage::Compute => "compute",
                slang::Stage::Geometry => "geometry",
                _ => "unknown",
            }
            .to_string();

            // Track all stages for push constants
            match stage {
                slang::Stage::Vertex => all_stages |= ShaderStageFlags::Vertex,
                slang::Stage::Fragment => all_stages |= ShaderStageFlags::Fragment,
                slang::Stage::Compute => all_stages |= ShaderStageFlags::Compute,
                slang::Stage::Geometry => all_stages |= ShaderStageFlags::Geometry,
                _ => {}
            }

            // For compute shaders, get thread group size
            let thread_group_size = if matches!(stage, slang::Stage::Compute) {
                Some(entry_point_ref.compute_thread_group_size())
            } else {
                None
            };

            entry_points.push(EntryPointInfo {
                name,
                stage: stage_name,
                thread_group_size,
            });
        }

        // Extract bindings from parameters
        // Use a HashMap to deduplicate bindings (same binding can appear in multiple stages)
        use std::collections::HashMap;
        let mut unique_bindings: HashMap<(u32, u32), Binding> = HashMap::new();
        let mut push_constants: Vec<PushConstantRange> = Vec::new();

        // Check global shader parameters
        for param in reflection.parameters() {
            let name = param.name().unwrap_or("unknown").to_string();

            // Skip system value semantics (SV_*) and unknown names
            if name.starts_with("SV_") || name == "unknown" {
                continue;
            }

            // Check if this is a push constant (category is PushConstantBuffer)
            let category = param.category();

            if matches!(category, Some(slang::ParameterCategory::PushConstantBuffer)) {
                // Extract push constant size from type layout
                if let Some(type_layout) = param.type_layout() {
                    // Get the element type layout for the actual data size
                    let size = if let Some(element_type_layout) = type_layout.element_type_layout()
                    {
                        // Use Uniform category for proper size calculation
                        element_type_layout.size(slang::ParameterCategory::Uniform) as u32
                    } else {
                        // Fallback to the type layout size
                        type_layout.size(slang::ParameterCategory::Uniform) as u32
                    };
                    // Ensure size is at least 4 bytes and aligned to 4 bytes
                    let aligned_size = ((size.max(4) + 3) / 4) * 4;
                    if aligned_size > 0 {
                        push_constants.push(PushConstantRange {
                            stage_flags: all_stages,
                            offset: 0,
                            size: aligned_size,
                        });
                    }
                }
                continue; // Don't add to regular bindings
            }

            let binding = param.binding_index();
            let set = param.binding_space();

            let (binding_type, count) = if let Some(type_layout) = param.type_layout() {
                let ty = Self::determine_binding_type(type_layout);
                let count = match type_layout.kind() {
                    slang::TypeKind::Array => type_layout.element_count().unwrap_or(0) as u32,
                    _ => 1,
                };

                (ty, count)
            } else {
                (BindingType::Unknown, 1)
            };

            // Use (set, binding) as key, merging Texture+Sampler into CombinedImageSampler
            let key = (set, binding);
            if let Some(existing) = unique_bindings.get(&key) {
                // Merge Sampler + Texture = CombinedImageSampler
                let merged_type = match (&existing.binding_type, &binding_type) {
                    (BindingType::Sampler, BindingType::SampledImage)
                    | (BindingType::SampledImage, BindingType::Sampler) => {
                        BindingType::CombinedImageSampler
                    }
                    _ => binding_type, // Keep the new one
                };
                unique_bindings.insert(
                    key,
                    Binding {
                        name: existing.name.clone(), // Keep existing name
                        binding,
                        set,
                        count,
                        binding_type: merged_type,
                    },
                );
            } else {
                unique_bindings.insert(
                    key,
                    Binding {
                        name,
                        binding,
                        set,
                        count,
                        binding_type,
                    },
                );
            }
        }

        // Also check entry point parameters (for parameters declared in entry point signature)
        for entry_point_ref in reflection.entry_points() {
            for param in entry_point_ref.parameters() {
                let name = param.name().unwrap_or("unknown").to_string();

                // Skip system value semantics (SV_*)
                if name.starts_with("SV_") || name == "unknown" {
                    continue;
                }

                // Skip parameters without a resource category (VaryingInput, VaryingOutput, etc.)
                // Only include descriptorTableSlot, ShaderResource, UnorderedAccess, etc.
                let category = param.category();
                match category {
                    Some(slang::ParameterCategory::DescriptorTableSlot)
                    | Some(slang::ParameterCategory::ShaderResource)
                    | Some(slang::ParameterCategory::UnorderedAccess)
                    | Some(slang::ParameterCategory::ConstantBuffer)
                    | Some(slang::ParameterCategory::SamplerState) => {
                        // These are valid descriptor bindings, continue processing
                    }
                    _ => {
                        // Skip non-descriptor parameters (VaryingInput, etc.)
                        continue;
                    }
                }

                let binding = param.binding_index();
                let set = param.binding_space();

                let (binding_type, count) = if let Some(type_layout) = param.type_layout() {
                    let ty = Self::determine_binding_type(type_layout);
                    let count = match type_layout.kind() {
                        slang::TypeKind::Array => type_layout.element_count().unwrap_or(0) as u32,
                        _ => 1,
                    };

                    (ty, count)
                } else {
                    (BindingType::Unknown, 1)
                };

                // Use (set, binding) as key to avoid duplicates
                unique_bindings.insert(
                    (set, binding),
                    Binding {
                        name,
                        binding,
                        set,
                        count,
                        binding_type,
                    },
                );
            }
        }

        // Convert to sorted Vec for deterministic ordering
        let mut bindings: Vec<Binding> = unique_bindings.into_values().collect();
        bindings.sort_by_key(|b| (b.set, b.binding));

        for b in &bindings {
            tracing::debug!(
                "Reflected binding: set={}, binding={}, type={:?}, count={}, name={}",
                b.set,
                b.binding,
                b.binding_type,
                b.count,
                b.name
            );
        }

        Ok(ShaderReflection {
            entry_points,
            bindings,
            push_constants,
        })
    }

    /// Determine binding type from type layout
    fn determine_binding_type(type_layout: &slang::reflection::TypeLayout) -> BindingType {
        let kind = type_layout.kind();
        use slang::TypeKind;

        let result = match kind {
            TypeKind::ConstantBuffer => BindingType::UniformBuffer,
            TypeKind::Array => {
                // For arrays like Texture2D[], recurse on the element type
                if let Some(element_type) = type_layout.element_type_layout() {
                    Self::determine_binding_type(&element_type)
                } else {
                    BindingType::Unknown
                }
            }
            TypeKind::Resource => {
                // Check binding range type for more precise classification
                if type_layout.binding_range_count() > 0 {
                    match type_layout.binding_range_type(0) {
                        slang::BindingType::ConstantBuffer => BindingType::UniformBuffer,
                        slang::BindingType::Texture => BindingType::SampledImage,
                        slang::BindingType::MutableTeture => BindingType::StorageImage,
                        slang::BindingType::Sampler => BindingType::Sampler,
                        slang::BindingType::TypedBuffer => BindingType::UniformTexelBuffer,
                        slang::BindingType::MutableTypedBuffer => BindingType::StorageTexelBuffer,
                        slang::BindingType::RawBuffer
                        | slang::BindingType::MutableRawBuffer => BindingType::StorageBuffer,
                        slang::BindingType::RayTracingAccelerationStructure => {
                            BindingType::AccelerationStructure
                        }
                        _ => BindingType::StorageBuffer,
                    }
                } else {
                    // Try to deduce from type layout
                    BindingType::SampledImage
                }
            }
            TypeKind::ParameterBlock => BindingType::UniformBuffer,
            TypeKind::SamplerState => BindingType::Sampler,
            _ => BindingType::Unknown,
        };

        tracing::debug!("Reflected TypeKind::{:?} -> {:?}", kind, result);
        result
    }

    /// Compile Slang shader from inline source code
    ///
    /// Automatically discovers all entry points marked with [shader("stage")] in the module.
    /// Returns a single SPIR-V module containing all entry points.
    ///
    /// # Arguments
    /// * `module_name` - Name to give the module (e.g., "inline_shader")
    /// * `source_path` - Virtual path for error messages (e.g., "shader.slang")
    /// * `source_code` - Slang shader source code
    /// * `target` - Compilation target (SPIRV, etc.)
    pub fn compile_inline(
        &self,
        module_name: &str,
        source_path: &str,
        source_code: &str,
        target: ShaderTarget,
    ) -> Result<ShaderModule, String> {
        // Validate inputs
        if module_name.is_empty() {
            return Err("module_name cannot be empty".to_string());
        }
        if source_code.is_empty() {
            return Err("source_code cannot be empty".to_string());
        }

        // Configure compiler options
        let session_options = slang::CompilerOptions::default()
            .optimization(slang::OptimizationLevel::High)
            .matrix_layout_column(true);

        // Configure target
        let target_desc = slang::TargetDesc::default()
            .format(target.to_slang_target())
            .profile(target.default_profile(&self.global_session));

        let targets = [target_desc];

        let session_desc = slang::SessionDesc::default()
            .targets(&targets)
            .options(&session_options);

        // Create compilation session
        let session = self
            .global_session
            .create_session(&session_desc)
            .ok_or_else(|| "Failed to create Slang session".to_string())?;

        // Load module from inline source
        let module = session
            .load_module_from_source_string(module_name, source_path, source_code)
            .map_err(|e| format!("Failed to load Slang module from source: {:?}", e))?;

        // Auto-discover all entry points
        let entry_point_count = module.entry_point_count();
        if entry_point_count == 0 {
            return Err("No entry points found in module".to_string());
        }

        // Build composite with module + all entry points
        let mut components: Vec<slang::ComponentType> = vec![module.clone().into()];
        for i in 0..entry_point_count {
            if let Some(entry_point) = module.entry_point_by_index(i) {
                components.push(entry_point.into());
            }
        }

        // Create composite component
        let program = session
            .create_composite_component_type(&components)
            .map_err(|e| format!("Failed to create composite component: {:?}", e))?;

        // Link program
        let linked_program = program
            .link()
            .map_err(|e| format!("Failed to link program: {:?}", e))?;

        // Extract reflection data
        let reflection = self.extract_reflection(&linked_program)?;

        // Get complete bytecode with ALL entry points
        let bytecode_blob = linked_program
            .target_code(0)
            .map_err(|e| format!("Failed to get target code: {:?}", e))?;

        let bytecode = bytecode_blob.as_slice().to_vec();

        // Convert entry points to shader stages
        let stages = reflection
            .entry_points
            .iter()
            .map(|ep| {
                let stage = match ep.stage.as_str() {
                    "vertex" => ShaderStageFlags::Vertex,
                    "fragment" => ShaderStageFlags::Fragment,
                    "compute" => ShaderStageFlags::Compute,
                    "geometry" => ShaderStageFlags::Geometry,
                    _ => return Err(format!("Unknown shader stage: {}", ep.stage)),
                };

                Ok(ShaderStageInfo {
                    stage,
                    entry_point: ep.name.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(ShaderModule {
            bytecode,
            stages,
            reflection,
        })
    }

    /// Compile Slang shader from file
    ///
    /// Automatically discovers all entry points marked with [shader("stage")] in the module.
    /// Returns a single SPIR-V module containing all entry points.
    ///
    /// # Arguments
    /// * `module_name` - Name of the .slang file (without extension)
    /// * `target` - Compilation target (SPIRV, etc.)
    /// * `search_paths` - Directories to search for the module and its includes
    pub fn compile_file(
        &self,
        module_name: &str,
        target: ShaderTarget,
        search_paths: &[&str],
    ) -> Result<ShaderModule, String> {
        // Validate inputs
        if module_name.is_empty() {
            return Err("module_name cannot be empty".to_string());
        }

        // Convert search paths to CString
        let mut final_search_paths: Vec<String> =
            search_paths.iter().map(|s| s.to_string()).collect();

        // 1. Add EXE-relative shaders folder
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                final_search_paths.push(exe_dir.join("shaders").to_string_lossy().to_string());
            }
        }

        // 2. Add Root fallback (for cargo run)
        final_search_paths.push("crates/i3_renderer/assets/shaders".to_string());

        let search_path_cstrings: Vec<_> = final_search_paths
            .iter()
            .map(|p| {
                std::ffi::CString::new(p.as_str())
                    .map_err(|e| format!("Invalid search path '{}': {}", p, e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let search_path_ptrs: Vec<_> = search_path_cstrings.iter().map(|s| s.as_ptr()).collect();

        // Configure compiler options
        let session_options = slang::CompilerOptions::default()
            .optimization(slang::OptimizationLevel::High)
            .matrix_layout_column(true);

        // Configure target
        let target_desc = slang::TargetDesc::default()
            .format(target.to_slang_target())
            .profile(target.default_profile(&self.global_session));

        let targets = [target_desc];

        let session_desc = slang::SessionDesc::default()
            .targets(&targets)
            .search_paths(&search_path_ptrs)
            .options(&session_options);

        // Create compilation session
        let session = self
            .global_session
            .create_session(&session_desc)
            .ok_or_else(|| "Failed to create Slang session".to_string())?;

        // Load module from file
        let module = session
            .load_module(module_name)
            .map_err(|e| format!("Failed to load Slang module: {:?}", e))?;

        // Auto-discover all entry points
        let entry_point_count = module.entry_point_count();
        if entry_point_count == 0 {
            return Err("No entry points found in module".to_string());
        }

        // Build composite with module + all entry points
        let mut components: Vec<slang::ComponentType> = vec![module.clone().into()];
        for i in 0..entry_point_count {
            if let Some(entry_point) = module.entry_point_by_index(i) {
                components.push(entry_point.into());
            }
        }

        // Create composite component
        let program = session
            .create_composite_component_type(&components)
            .map_err(|e| format!("Failed to create composite component: {:?}", e))?;

        // Link program
        let linked_program = program
            .link()
            .map_err(|e| format!("Failed to link program: {:?}", e))?;

        // Extract reflection data
        let reflection = self.extract_reflection(&linked_program)?;

        // Get complete bytecode with ALL entry points
        let bytecode_blob = linked_program
            .target_code(0)
            .map_err(|e| format!("Failed to get target code: {:?}", e))?;

        let bytecode = bytecode_blob.as_slice().to_vec();

        // Convert entry points to shader stages
        let stages = reflection
            .entry_points
            .iter()
            .map(|ep| {
                let stage = match ep.stage.as_str() {
                    "vertex" => ShaderStageFlags::Vertex,
                    "fragment" => ShaderStageFlags::Fragment,
                    "compute" => ShaderStageFlags::Compute,
                    "geometry" => ShaderStageFlags::Geometry,
                    _ => return Err(format!("Unknown shader stage: {}", ep.stage)),
                };

                Ok(ShaderStageInfo {
                    stage,
                    entry_point: ep.name.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(ShaderModule {
            bytecode,
            stages,
            reflection,
        })
    }
}

#[cfg(test)]
#[path = "tests/slang_compiler.rs"]
mod tests;
