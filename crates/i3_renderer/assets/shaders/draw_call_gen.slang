struct GpuMeshDescriptor {
    uint2    vertexBufferAddress;
    uint2    indexBufferAddress;
    uint     indexCount;
    uint     vertexStride;
    uint     firstIndex;
    int      vertexOffset;
    float3   aabbMin;
    float    _pad0;
    float3   aabbMax;
    float    _pad1;
};

struct GpuInstanceData {
    float4x4 worldTransform;
    float4x4 prevTransform;
    uint     meshIdx;
    uint     materialId;
    uint     flags;
    uint     _pad;
    float3   worldAabbMin;
    float    _pad2;
    float3   worldAabbMax;
    float    _pad3;
};

struct DrawIndexedIndirectCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int  vertexOffset;
    uint firstInstance;
};

struct PushConstants {
    uint instanceCount;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]] StructuredBuffer<GpuMeshDescriptor> meshDescriptors;
[[vk::binding(1, 0)]] StructuredBuffer<GpuInstanceData> instances;
[[vk::binding(2, 0)]] RWStructuredBuffer<DrawIndexedIndirectCommand> drawCalls;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> drawCount;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= pc.instanceCount) return;

    GpuInstanceData instance = instances[idx];
    GpuMeshDescriptor mesh = meshDescriptors[instance.meshIdx];
    
    uint drawIdx;
    InterlockedAdd(drawCount[0], 1, drawIdx);

    DrawIndexedIndirectCommand cmd;
    cmd.indexCount    = mesh.indexCount;
    cmd.instanceCount = 1;
    cmd.firstIndex    = mesh.firstIndex;
    cmd.vertexOffset  = mesh.vertexOffset;
    cmd.firstInstance = idx;

    drawCalls[drawIdx] = cmd;
}
