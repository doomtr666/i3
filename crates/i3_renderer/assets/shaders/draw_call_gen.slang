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
    uint instanceCount;
    uint maxDrawCalls;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= pc.instanceCount)
        return;

    GpuInstanceData instance = instances[idx];
    GpuMeshDescriptor mesh = meshDescriptors[instance.meshIdx];

    uint drawIdx;
    InterlockedAdd(drawCount[0], 1, drawIdx);

    if (drawIdx < pc.maxDrawCalls)
    {
        DrawIndirectCommand cmd;
        cmd.vertexCount = mesh.indexCount;
        cmd.instanceCount = mesh.indexCount > 0 ? 1 : 0; // Drop empty meshes entirely (prevents out-of-bounds BDA reads)
        cmd.firstVertex = 0; // Starts with SV_VertexID = 0
        cmd.firstInstance = idx; // Absolute instance index passed via gl_InstanceIndex (Vulkan)

        drawCalls[drawIdx] = cmd;
    }
}
