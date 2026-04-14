#include "common.slangh"

struct DrawIndirectCommand
{
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct DrawCallGenPushConstants
{
    float4x4 viewProjection;
    uint instanceCount;
    uint maxDrawCalls;
    uint2 _pad;
};

[[vk::binding(0, 0)]]
StructuredBuffer<GpuMeshDescriptor> meshDescriptors;
[[vk::binding(1, 0)]]
StructuredBuffer<GpuInstanceData> instances;
[[vk::binding(2, 0)]]
RWStructuredBuffer<DrawIndirectCommand> drawCalls;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> drawCount;

[[vk::push_constant]]
DrawCallGenPushConstants pc;

// Standard frustum culling test.
// Returns true if the AABB is at least partially in the frustum.
bool frustumTest(float3 minBound, float3 maxBound, float4x4 vp)
{
    float3 corners[8] = {
        float3(minBound.x, minBound.y, minBound.z),
        float3(maxBound.x, minBound.y, minBound.z),
        float3(minBound.x, maxBound.y, minBound.z),
        float3(maxBound.x, maxBound.y, minBound.z),
        float3(minBound.x, minBound.y, maxBound.z),
        float3(maxBound.x, minBound.y, maxBound.z),
        float3(minBound.x, maxBound.y, maxBound.z),
        float3(maxBound.x, maxBound.y, maxBound.z),
    };

    for (int p = 0; p < 6; p++)
    {
        int outsideCount = 0;
        for (int c = 0; c < 8; c++)
        {
            float4 clip = mul(vp, float4(corners[c], 1.0));
            bool isOutside = false;
            switch (p)
            {
            case 0:
                isOutside = clip.x < -clip.w;
                break; // Left
            case 1:
                isOutside = clip.x > clip.w;
                break; // Right
            case 2:
                isOutside = clip.y < -clip.w;
                break; // Bottom
            case 3:
                isOutside = clip.y > clip.w;
                break; // Top
            case 4:
                isOutside = clip.z < 0.0;
                break; // Near (Reverse-Z)
            case 5:
                isOutside = clip.z > clip.w;
                break; // Far (Reverse-Z)
            }
            if (isOutside)
                outsideCount++;
        }
        if (outsideCount == 8)
            return false;
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= pc.instanceCount)
        return;

    GpuInstanceData instance = instances[idx];

    // 1. Frustum Culling
    if (!frustumTest(instance.worldAabbMin, instance.worldAabbMax, pc.viewProjection))
        return;

    GpuMeshDescriptor mesh = meshDescriptors[instance.meshIdx];

    uint drawIdx;
    InterlockedAdd(drawCount[0], 1, drawIdx);

    if (drawIdx < pc.maxDrawCalls)
    {
        DrawIndirectCommand cmd;
        cmd.vertexCount = mesh.indexCount;
        cmd.instanceCount = mesh.indexCount > 0 ? 1 : 0;
        cmd.firstVertex = 0;
        cmd.firstInstance = idx;

        drawCalls[drawIdx] = cmd;
    }
}
