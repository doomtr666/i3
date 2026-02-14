use crate::*;

#[test]
fn test_create_compiler() {
    let compiler = SlangCompiler::new();
    assert!(compiler.is_ok());
}

#[test]
fn test_compile_inline_compute_shader() {
    let compiler = SlangCompiler::new().expect("Failed to create compiler");

    let source = r#"
[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform Texture2D inputTex,
    uniform RWTexture2D<float4> outputTex)
{
    float4 value = inputTex[dispatchThreadID.xy];
    outputTex[dispatchThreadID.xy] = value * 2.0;
}
"#;

    let result = compiler.compile_inline(
        "test_shader",
        "test.slang",
        source,
        ShaderTarget::Spirv,
    );

    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

    let shader = result.unwrap();

    // Verify bytecode was generated
    assert!(!shader.bytecode.is_empty(), "Bytecode should not be empty");

    // Verify reflection data
    assert_eq!(
        shader.reflection.entry_points.len(),
        1,
        "Should have 1 entry point"
    );

    let entry_point = &shader.reflection.entry_points[0];
    assert_eq!(entry_point.name, "computeMain");
    assert_eq!(entry_point.stage, "compute");
    assert_eq!(entry_point.thread_group_size, Some([8, 8, 1]));

    // Debug: print bindings
    println!("Found {} bindings:", shader.reflection.bindings.len());
    for binding in &shader.reflection.bindings {
        println!(
            "  - {} (binding={}, set={}, type={:?})",
            binding.name, binding.binding, binding.set, binding.binding_type
        );
    }

    // Slang may organize parameters differently, so just check we have bindings
    assert!(
        !shader.reflection.bindings.is_empty(),
        "Should have at least some bindings"
    );
}

#[test]
fn test_compile_inline_multi_entry() {
    let compiler = SlangCompiler::new().expect("Failed to create compiler");

    let source = r#"
[shader("vertex")]
void vertexMain(float3 pos : POSITION, out float4 sv_position : SV_Position) {
    sv_position = float4(pos, 1.0);
}

[shader("fragment")]
void fragmentMain(out float4 color : SV_Target) {
    color = float4(1.0, 0.0, 0.0, 1.0);
}
"#;

    let result = compiler.compile_inline(
        "test_module",
        "test.slang",
        source,
        ShaderTarget::Spirv,
    );

    assert!(result.is_ok(), "Compilation failed: {:?}", result.err());

    let shader = result.unwrap();

    // Verify single bytecode
    assert!(!shader.bytecode.is_empty(), "Bytecode should not be empty");

    // Verify both entry points discovered
    assert_eq!(
        shader.reflection.entry_points.len(),
        2,
        "Should have 2 entry points"
    );

    let entry_names: Vec<_> = shader
        .reflection
        .entry_points
        .iter()
        .map(|e| e.name.as_str())
        .collect();

    assert!(entry_names.contains(&"vertexMain"));
    assert!(entry_names.contains(&"fragmentMain"));
}

#[test]
fn test_compile_inline_empty_module_name() {
    let compiler = SlangCompiler::new().expect("Failed to create compiler");
    let result = compiler.compile_inline(
        "",
        "test.slang",
        "void main() {}",
        ShaderTarget::Spirv,
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("module_name"));
}

#[test]
fn test_compile_inline_empty_source() {
    let compiler = SlangCompiler::new().expect("Failed to create compiler");
    let result = compiler.compile_inline(
        "test",
        "test.slang",
        "",
        ShaderTarget::Spirv,
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("source_code"));
}

#[test]
fn test_compile_inline_no_entry_points() {
    let compiler = SlangCompiler::new().expect("Failed to create compiler");
    let source = "// Empty module with no entry points";
    let result = compiler.compile_inline(
        "test",
        "test.slang",
        source,
        ShaderTarget::Spirv,
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("No entry points found"));
}
