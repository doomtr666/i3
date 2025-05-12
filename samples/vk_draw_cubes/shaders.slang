// shaders.slang


//
// This file provides a simple vertex and fragment shader that can be compiled
// using Slang. This code should also be valid as HLSL, and thus it does not
// use any of the new language features supported by Slang.
//

// Uniform data to be passed from application -> shader.
[vk::push_constant]
cbuffer Uniforms
{
    float4x4 modelViewProjection;
}

// Per-vertex attributes to be assembled from bound vertex buffers.
struct AssembledVertex
{
    float3	position : POSITION;
    float3	normal   : NORMAL;
};

// Output of the vertex shader, and input to the fragment shader.
struct CoarseVertex
{
    float3 normal;
};

// Output of the fragment shader
struct Fragment
{
    float4 color;
};

// Vertex  Shader
struct VertexStageOutput
{
    CoarseVertex coarseVertex    : CoarseVertex;
    float4 sv_position     : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    VertexStageOutput output;

    float3 position = assembledVertex.position;
    float3 normal   = assembledVertex.normal;

    output.coarseVertex.normal = normal;
    output.sv_position = mul(modelViewProjection, float4(position, 1.0));

    return output;
}

// Fragment Shader

[shader("fragment")]
Fragment fragmentMain(
    CoarseVertex coarseVertex : CoarseVertex) : SV_Target
{
    float3 normal = coarseVertex.normal;

    float intensity = dot(normal, float3(1,0,0));

    Fragment output;
    output.color = float4(1,0,0,1) * (0.5f * intensity + 0.5f); //float4(intensity * float3(1,1,1), 1.0);
    return output;
}