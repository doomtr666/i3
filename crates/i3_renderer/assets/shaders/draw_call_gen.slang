struct GpuMeshDescriptor
{
    uint64_t vertexBufferAddress; // BDA — cohérent avec common.slangh et GpuMeshDescriptor Rust (u64)
    uint64_t indexBufferAddress;  // BDA — cohérent avec common.slangh et GpuMeshDescriptor Rust (u64)
    uint indexCount;
    uint vertexStride;
    uint firstIndex;
    int vertexOffset;
    float aabbMinX;
    float aabbMinY;
    float aabbMinZ;
    uint indexStride; // 2 = U16, 4 = U32 — cohérent avec common.slangh et scene.rs
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
    float _pad1;
};

struct GpuInstanceData
{
    float4x4 worldTransform;
    float4x4 prevTransform;
    uint meshIdx;
    uint materialId;
    uint flags;
    uint _pad;
    float worldAabbMinX;
    float worldAabbMinY;
    float worldAabbMinZ;
    float _pad2;
    float worldAabbMaxX;
    float worldAabbMaxY;
    float worldAabbMaxZ;
    float _pad3;
};

struct DrawIndirectCommand
{
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct PushConstants
{
    uint instanceCount;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]]
StructuredBuffer<GpuMeshDescriptor> meshDescriptors;
[[vk::binding(1, 0)]]
StructuredBuffer<GpuInstanceData> instances;
[[vk::binding(2, 0)]]
RWStructuredBuffer<DrawIndirectCommand> drawCalls;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> drawCount;

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

    DrawIndirectCommand cmd;
    cmd.vertexCount = mesh.indexCount > 0 ? mesh.indexCount : 3; // Fallback to 3 if no indices? No, mesh.indexCount should be vertices to draw
    cmd.instanceCount = 1;
    cmd.firstVertex = 0;
    cmd.firstInstance = idx;

    drawCalls[drawIdx] = cmd;
}
