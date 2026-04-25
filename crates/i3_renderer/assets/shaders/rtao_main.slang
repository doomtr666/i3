// Ray-Traced Ambient Occlusion — ground-truth reference pass
//
// One visibility ray per pixel per frame.
// Accumulation: V_new = (V_old * N + sample) / (N + 1)
// sampleCount == 0 → first frame / camera moved → reset accumulation.
//

#include "common.slangh"
#include "sampling.slangh"
#include "raytracing.slangh"

struct RtaoPushConstants
{
    float4x4 invViewProj;
    float2 screenSize;
    float radius;
    uint sampleCount; // 0 = reset (camera moved / first frame)
    uint frameIndex;
    uint blueNoiseIndex;
    uint _pad0;
    uint _pad1;
};

[[vk::push_constant]]
RtaoPushConstants pc;

[[vk::binding(0, 0)]]
Texture2D<float> depthBuffer;
[[vk::binding(1, 0)]]
Texture2D<float2> gbufferNormal;
[[vk::binding(2, 0)]]
Texture2D<float> rtaoAccumIn; // previous frame accumulation
[[vk::binding(3, 0)]]
RWTexture2D<float> rtaoAccumOut; // current frame accumulation
[[vk::binding(4, 0)]]
RWTexture2D<float> aoOutput; // resolved AO for this frame
[[vk::binding(5, 0)]]
RaytracingAccelerationStructure tlas;


[numthreads(8, 8, 1)]
void computeMain(uint3 dispatchId: SV_DispatchThreadID)
{
    uint2 px = dispatchId.xy;
    if (any(px >= uint2(pc.screenSize)))
        return;

    // get world position
    float depth = depthBuffer[px].r;
    // Reverse-Z: depth==0 is sky/far plane — no geometry to occlude.
    if (depth <= 0.0f)
    {
        aoOutput[int2(px)] = 1.0f;
        rtaoAccumOut[int2(px)] = 1.0f;
        return;
    }
    float2 uv = (float2(px) + 0.5f) / pc.screenSize;
    float3 P = worldPosFromDepth(uv, depth, pc.invViewProj);

    // decode world space normal
    float3 N = decodeOctahedral(gbufferNormal[px].rg);

    // noise sample depends on position + frameIndex
    float2 noise = pcg3f(uint3(px, pc.frameIndex)).xy;

    // generate cosine weighted direction (tangent space)
    float3 tangent_dir = float3(
        cos(2 * PI * noise.x) * sqrt(noise.y),
        sin(2 * PI * noise.x) * sqrt(noise.y),
        sqrt(1.0 - noise.y));

    // transform ray direction to world space
    float3 T, B;
    buildBasis(N, T, B);
    float3 ray_dir = tangent_dir.x * T + tangent_dir.y * B + tangent_dir.z * N;

    // visibility test
    RayDesc ray;
    ray.Origin = P;
    ray.Direction = normalize(ray_dir);
    ray.TMin = 0.001f;
    ray.TMax = pc.radius;

    float visibility = traceShadow(tlas, ray) ? 0.0 : 1.0;

    // Progressive accumulation
    float accumulated;
    if (pc.sampleCount == 0)
    {
        accumulated = visibility;
    }
    else
    {
        float prev = rtaoAccumIn[int2(px)];
        accumulated = (prev * float(pc.sampleCount) + visibility) / float(pc.sampleCount + 1);
    }

    rtaoAccumOut[int2(px)] = accumulated;
    aoOutput[int2(px)] = accumulated;
}
