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
    float4x4 prevViewProjection;
    float2 screenSize;
    uint hizMipCount;
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
[[vk::binding(4, 0)]]
Texture2D hizPyramid;

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

    for (int p = 0; p < 6; p++) {
        int outsideCount = 0;
        for (int c = 0; c < 8; c++) {
            float4 clip = mul(vp, float4(corners[c], 1.0));
            bool isOutside = false;
            switch(p) {
                case 0: isOutside = clip.x < -clip.w; break; // Left
                case 1: isOutside = clip.x >  clip.w; break; // Right
                case 2: isOutside = clip.y < -clip.w; break; // Bottom
                case 3: isOutside = clip.y >  clip.w; break; // Top
                case 4: isOutside = clip.z <  0.0;     break; // Near (Reverse-Z)
                case 5: isOutside = clip.z >  clip.w; break; // Far (Reverse-Z)
            }
            if (isOutside) outsideCount++;
        }
        if (outsideCount == 8) return false;
    }
    return true;
}

// Hierarchical Z-Buffer Occlusion Test
bool occlusionTest(float3 minBound, float3 maxBound, float4x4 prevVp)
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

    float minZ = 1.0;
    float2 minXY = float2(1.0);
    float2 maxXY = float2(0.0);

    for (int i = 0; i < 8; i++)
    {
        float4 clip = mul(prevVp, float4(corners[i], 1.0));
        clip.xyz /= clip.w;
        clip.xy = clip.xy * 0.5 + 0.5;
        clip.y = 1.0 - clip.y; // Flip Y for screen space

        minXY = min(minXY, clip.xy);
        maxXY = max(maxXY, clip.xy);
        minZ = min(minZ, clip.z);
    }

    float width = (maxXY.x - minXY.x) * pc.screenSize.x;
    float height = (maxXY.y - minXY.y) * pc.screenSize.y;

    float mip = ceil(log2(max(width, height)));
    mip = clamp(mip, 0, float(pc.hizMipCount - 1));

    // Sample the 4 texels that cover our screen-space AABB.
    float4 depthSamples;
    depthSamples.x = hizPyramid.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], minXY, mip).r;
    depthSamples.y = hizPyramid.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], float2(maxXY.x, minXY.y), mip).r;
    depthSamples.z = hizPyramid.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], float2(minXY.x, maxXY.y), mip).r;
    depthSamples.w = hizPyramid.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], maxXY, mip).r;
    
    float maxHizDepth = min(min(depthSamples.x, depthSamples.y), min(depthSamples.z, depthSamples.w));

    // Reverse-Z: higher ZValue means closer to camera (0.0=far, 1.0=near)
    // If our minZ (closest point of AABB) is less than the max Z (furthest point in Hi-Z area), 
    // it means background geometry is clearly in front of our closest point -> occluded.
    return minZ >= maxHizDepth;
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

    // 2. Occlusion Culling (skip if prevViewProjection is not valid yet)
    // Check if prevViewProjection is identity or uninitialized (e.g. check m33 which is 0 for perspective)
    if (pc.prevViewProjection[3][3] != 1.0)
    {
        //if (!occlusionTest(instance.worldAabbMin, instance.worldAabbMax, pc.prevViewProjection))
        //    return;
    }

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
