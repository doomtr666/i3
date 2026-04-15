// prez.slang — Depth-only vertex shader for the PreZ pass.
//
// Reads the same BDA-based vertex layout as gbuffer.slang but only outputs
// clip-space position. No fragment shader — the rasterizer writes depth only.
#include "common.slangh"

[[vk::binding(0, 0)]] StructuredBuffer<GpuMeshDescriptor> meshDescriptors;
[[vk::binding(1, 0)]] StructuredBuffer<GpuInstanceData>   instances;

struct PushConstants { float4x4 viewProjection; };
[[vk::push_constant]] PushConstants pc;

[shader("vertex")]
float4 main(
    uint indexId    : SV_VertexID,
    uint instanceId : SV_InstanceID,
    [[vk::builtin("BaseInstance")]] uint baseInstance : SV_StartInstanceLocation
) : SV_Position
{
    uint absInstId = baseInstance + instanceId;
    GpuInstanceData   instance = instances[absInstId];
    GpuMeshDescriptor mesh     = meshDescriptors[instance.meshIdx];

    // Index fetch (U16 or U32) — same logic as gbuffer.slang.
    uint vertexIdx;
    if (mesh.indexCount > 0u && mesh.indexBufferAddress != 0)
    {
        uint indexWithOffset = indexId + mesh.firstIndex;
        vertexIdx = (mesh.indexStride == 2u)
            ? (uint)((uint16_t *)mesh.indexBufferAddress)[indexWithOffset]
            : ((uint *)mesh.indexBufferAddress)[indexWithOffset];
    }
    else
    {
        vertexIdx = indexId;
    }

    // Position-only fetch via BDA byte arithmetic.
    // Vertex layout: [+0] float3 position (12B) | [+12] float3 normal | ...
    uint64_t vBase = mesh.vertexBufferAddress
                   + uint64_t(vertexIdx + mesh.vertexOffset) * uint64_t(mesh.vertexStride);
    float3 pos = *(float3 *)(vBase);

    return mul(pc.viewProjection, mul(instance.worldTransform, float4(pos, 1.0)));
}
