#include "shaders_common.slangh"

[vk::push_constant]
cbuffer PushConstants
{
    float4x4 world;
}

// Per-vertex attributes to be assembled from bound vertex buffers.
struct AssembledVertex
{
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float3 tangent;
    [[vk::location(3)]] float3 binormal;
    [[vk::location(4)]] float2 uv;
};

// Output of the vertex shader, and input to the fragment shader.
struct CoarseVertex
{
    float3 normal;
    float3 tangent;
    float3 binormal;
    float2 uv;
};

// Output of the fragment shader
struct Fragment
{
    float4 normal : SV_Target0;
    float4 albedo : SV_Target1;
    float4 metalic_roughness : SV_Target2;
};

// Vertex  Shader

struct VertexStageOutput
{
    CoarseVertex coarseVertex : CoarseVertex;
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    VertexStageOutput output;

    float3 position = assembledVertex.position;
    float3 normal   = assembledVertex.normal;

    output.coarseVertex.normal = mul(world,float4(normal,0)).xyz;
    output.coarseVertex.tangent = mul(world,float4(assembledVertex.tangent,0)).xyz;
    output.coarseVertex.binormal = mul(world,float4(assembledVertex.binormal,0)).xyz;
    output.coarseVertex.uv = assembledVertex.uv;

    output.sv_position = mul(mul(projView, world), float4(position, 1.0));

    return output;
}

// Fragment Shader

[shader("fragment")]
Fragment fragmentMain(
    CoarseVertex coarseVertex : CoarseVertex)
{
    Fragment output;
    output.albedo = float4(1,0,0,0);
    output.normal = encode_normal(normalize(coarseVertex.normal));
    output.metalic_roughness = 0;

    return output;
}