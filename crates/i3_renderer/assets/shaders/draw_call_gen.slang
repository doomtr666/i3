#include "common.slangh"

struct DrawCallGenPushConstants
{
    float4x4 viewProjection;
    float2 hizSize;   // HiZ render target size (HIZ_PREZ_WIDTH × HIZ_PREZ_HEIGHT)
    uint hizMipCount; // number of mip levels in HiZPreZ (0 = no HiZ, skip occlusion test)
    uint instanceCount;
    uint maxDrawCalls;
    uint _pad;
};

[[vk::binding(0, 0)]]
StructuredBuffer<GpuMeshDescriptor> meshDescriptors;
[[vk::binding(1, 0)]]
StructuredBuffer<GpuInstanceData> instances;
[[vk::binding(2, 0)]]
RWStructuredBuffer<DrawIndirectCommand> drawCalls;
[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> drawCount;
[[vk::binding(4, 0)]]
Texture2D<float> hizPyramid; // HiZPreZ — MAX pyramid, reverse-Z
[[vk::binding(5, 0)]]
RWStructuredBuffer<uint> visibleBitset; // VisibilityBitset frame N (write)

[[vk::push_constant]]
DrawCallGenPushConstants pc;

// Occlusion test against a reverse-Z MAX Hi-Z pyramid.
//
// Returns true  = instance is visible (or conservatively assumed visible).
// Returns false = instance is fully occluded by geometry in HiZPreZ.
//
// Algorithm (GPU-driven occlusion, same-frame VP):
//   1. Project all 8 AABB corners to NDC.
//   2. Take the screen-space bounding rect (UV) and the max NDC Z (front face, reverse-Z).
//   3. Select HiZ mip whose texel footprint covers the whole rect.
//   4. Sample the 4 corners of the rect and take the MAX (most conservative).
//   5. If HiZ MAX (closest occluder) > AABB front face depth → AABB is behind the occluder → CULL.
bool occlusionTest(float3 aabbMin, float3 aabbMax)
{
    // With hizMipCount == 0 (no HiZ available yet — e.g. very first frame before PreZ runs),
    // skip the test conservatively (pass everything through).
    if (pc.hizMipCount == 0)
        return true;

    float3 corners[] = {
        float3(aabbMin.x, aabbMin.y, aabbMin.z),
        float3(aabbMax.x, aabbMin.y, aabbMin.z),
        float3(aabbMin.x, aabbMax.y, aabbMin.z),
        float3(aabbMax.x, aabbMax.y, aabbMin.z),
        float3(aabbMin.x, aabbMin.y, aabbMax.z),
        float3(aabbMax.x, aabbMin.y, aabbMax.z),
        float3(aabbMin.x, aabbMax.y, aabbMax.z),
        float3(aabbMax.x, aabbMax.y, aabbMax.z),
    };

    float3 uvMin = float3(1e10, 1e10, 1e10);
    float3 uvMax = float3(-1e10, -1e10, -1e10);

    for (int i = 0; i < 8; i++)
    {
        float3 corner = corners[i];
        float4 clip = mul(pc.viewProjection, float4(corner, 1.0));

        if (clip.w <= 0.0)
            return true; // behind camera — conservative

        float3 ndc = clamp(clip.xyz / clip.w, float3(-1.0, -1.0, 0.0), float3(1.0, 1.0, 1.0));
        float3 uv = float3(ndc.xy * 0.5 + 0.5, ndc.z);
        uv.y = 1.0 - uv.y;

        uvMin = min(uvMin, uv);
        uvMax = max(uvMax, uv);
    }

    // 4-corner HiZ test.
    //
    // Mip = ceil(log2(footprint)) - 1: each texel covers half the footprint, so the
    // 4 corners together span the whole projected AABB.  The conservative MAX pyramid
    // (now built with ceil downsampling) guarantees no occluder is missed.
    // MIN across 4 samples: cull only if every corner is blocked.
    // SAMPLER_LINEAR_BORDER returns 0.0 (far plane) outside [0,1] → conservative.
    float2 footprint = (uvMax.xy - uvMin.xy) * pc.hizSize;
    float mipf = max(ceil(log2(max(max(footprint.x, footprint.y), 1.0))) - 1.0, 0.0);
    uint mip = (uint)clamp(mipf, 0.0, float(pc.hizMipCount - 1));

    float d0 = hizPyramid.SampleLevel(bindlessSamplers[SAMPLER_LINEAR_BORDER], float2(uvMin.x, uvMin.y), mip).r;
    float d1 = hizPyramid.SampleLevel(bindlessSamplers[SAMPLER_LINEAR_BORDER], float2(uvMax.x, uvMin.y), mip).r;
    float d2 = hizPyramid.SampleLevel(bindlessSamplers[SAMPLER_LINEAR_BORDER], float2(uvMin.x, uvMax.y), mip).r;
    float d3 = hizPyramid.SampleLevel(bindlessSamplers[SAMPLER_LINEAR_BORDER], float2(uvMax.x, uvMax.y), mip).r;
    float hizDepth = min(min(d0, d1), min(d2, d3));

    // Reverse-Z MAX HiZ occlusion decision:
    //   hizDepth  = depth of closest occluder in footprint (larger = closer to camera)
    //   ndcMaxZ   = depth of AABB's front face           (larger = closer to camera)
    //
    //   hizDepth > ndcMaxZ + kBias  →  occluder clearly in front → CULL
    //   hizDepth ≤ ndcMaxZ + kBias  →  AABB visible (or close call → conservative pass)
    //
    // kBias guards against self-occlusion: objects drawn in PreZ are present in HiZPreZ,
    // so their own depth would appear as the closest occluder. Without a bias, floating
    // point rounding causes hizDepth ≈ ndcMaxZ → the test becomes ambiguous.
    // 1/65536 ≈ 1.5e-5 in [0,1] depth space; small enough to avoid leaking through thin walls.
    static const float kBias = 1.0 / 65536.0;
    return hizDepth <= uvMax.z + kBias;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= pc.instanceCount)
        return;

    GpuInstanceData instance = instances[idx];

    // 1. Frustum cull.
    if (!frustumTest(instance.worldAabbMin, instance.worldAabbMax, pc.viewProjection))
        return;

    // 2. Occlusion cull against HiZPreZ (same-frame VP — no reprojection artefacts).
    // if (!occlusionTest(instance.worldAabbMin, instance.worldAabbMax))
    //    return;

    // 3. Instance is visible: record in VisibilityBitset for PreZCull next frame.
    visibilitySet(visibleBitset, idx);

    // 4. Emit draw call.
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
