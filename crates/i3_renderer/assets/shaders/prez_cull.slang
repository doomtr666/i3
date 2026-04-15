#include "common.slangh"

// PreZ Cull — Compute shader (1 thread per instance).
//
// Emits a draw call for each instance that:
//   1. Passes frustum culling against the current frame view-projection.
//   2. Was visible in frame N-1 (bit set in visibilityBitsetHistory).
//
// This produces the draw list for the PreZ depth-only pass, which builds
// the HiZ pyramid used by the subsequent occlusion cull pass.
//
// On the first frame (or after a scene reset), visibilityBitsetHistory is
// all-zero → nothing is drawn in the PreZ → HiZPreZ is all 0 (far plane) →
// the occlusion cull in OcclusionCullPass passes everything conservatively.

struct PreZCullPushConstants
{
    float4x4 viewProjection;
    uint instanceCount;
    uint maxDrawCalls;
    uint2 _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuMeshDescriptor>     meshDescriptors;
[[vk::binding(1, 0)]] StructuredBuffer<GpuInstanceData>       instances;
[[vk::binding(2, 0)]] RWStructuredBuffer<DrawIndirectCommand>  drawCalls;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>                 drawCount;
[[vk::binding(4, 0)]] StructuredBuffer<uint>                   visibilityBitsetHistory; // frame N-1

[[vk::push_constant]] PreZCullPushConstants pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= pc.instanceCount)
        return;

    GpuInstanceData instance = instances[idx];

    // 1. Frustum cull — skip if outside current frame frustum.
    if (!frustumTest(instance.worldAabbMin, instance.worldAabbMax, pc.viewProjection))
        return;

    // 2. Temporal visibility filter — skip if not visible in frame N-1.
    if (!visibilityGet(visibilityBitsetHistory, idx))
        return;

    // Emit draw call.
    GpuMeshDescriptor mesh = meshDescriptors[instance.meshIdx];
    uint drawIdx;
    InterlockedAdd(drawCount[0], 1, drawIdx);
    if (drawIdx < pc.maxDrawCalls)
    {
        DrawIndirectCommand cmd;
        cmd.vertexCount   = mesh.indexCount;
        cmd.instanceCount = mesh.indexCount > 0 ? 1 : 0;
        cmd.firstVertex   = 0;
        cmd.firstInstance = idx; // passed as gl_BaseInstance to the vertex shader
        drawCalls[drawIdx] = cmd;
    }
}
