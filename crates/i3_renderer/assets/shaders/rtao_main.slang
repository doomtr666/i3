// Ray-Traced Ambient Occlusion — raw sample pass
//
// One cosine-weighted visibility ray per pixel per frame.
// Outputs a single noisy AO sample to RTAO_Raw.
// Temporal accumulation is handled by a separate RtaoTemporalPass (EMA + reprojection).

#include "common.slangh"
#include "sampling.slangh"
#include "raytracing.slangh"

struct RtaoPushConstants
{
    float4x4 invViewProj;
    float2   screenSize;
    float    radius;
    uint     frameIndex;
    uint     blueNoiseIndex;
};

[[vk::push_constant]]
RtaoPushConstants pc;

[[vk::binding(0, 0)]] Texture2D<float>    depthBuffer;
[[vk::binding(1, 0)]] Texture2D<float2>   gbufferNormal;
[[vk::binding(2, 0)]] RWTexture2D<float>  rtaoRaw; // one noisy sample per pixel
[[vk::binding(3, 0)]] RaytracingAccelerationStructure tlas;

[numthreads(8, 8, 1)]
void computeMain(uint3 dispatchId: SV_DispatchThreadID)
{
    uint2 px = dispatchId.xy;
    if (any(px >= uint2(pc.screenSize)))
        return;

    float depth = depthBuffer[px].r;
    if (depth <= 0.0f)
    {
        rtaoRaw[int2(px)] = 1.0f;
        return;
    }

    float2 uv = (float2(px) + 0.5f) / pc.screenSize;
    float3 P  = worldPosFromDepth(uv, depth, pc.invViewProj);
    float3 N  = decodeOctahedral(gbufferNormal[px].rg);

    float2 noise = pcg3f(uint3(px, pc.frameIndex)).xy;

    float3 tangent_dir = float3(
        cos(2 * PI * noise.x) * sqrt(noise.y),
        sin(2 * PI * noise.x) * sqrt(noise.y),
        sqrt(1.0 - noise.y));

    float3 T, B;
    buildBasis(N, T, B);
    float3 ray_dir = tangent_dir.x * T + tangent_dir.y * B + tangent_dir.z * N;

    RayDesc ray;
    ray.Origin    = P;
    ray.Direction = normalize(ray_dir);
    ray.TMin      = 0.001f;
    ray.TMax      = pc.radius;

    rtaoRaw[int2(px)] = traceShadow(tlas, ray) ? 0.0f : 1.0f;
}
