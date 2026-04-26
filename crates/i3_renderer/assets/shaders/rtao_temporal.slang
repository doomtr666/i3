// RTAO Temporal Accumulation with reprojection
//
// Identical algorithm to gtao_temporal — EMA blend with reprojection.
// History is reprojectd via prevViewProj so accumulation survives camera movement.
// Disocclusion (off-screen reprojection) falls back to the raw sample directly.

#include "common.slangh"

struct RtaoTemporalPushConstants {
    float4x4 invViewProj;
    float4x4 prevViewProj;
    float2   screenSize;
    float    alpha;
    uint     frameIndex;
};

[[vk::push_constant]]
RtaoTemporalPushConstants pc;

[[vk::binding(0, 0)]] Texture2D<float>   depthBuffer;
[[vk::binding(1, 0)]] Texture2D<float>   rtaoRaw;    // current frame noisy sample
[[vk::binding(2, 0)]] Texture2D<float>   aoHistory;  // previous frame AO_Resolved
[[vk::binding(3, 0)]] RWTexture2D<float> aoResolved; // output (becomes next frame history)

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 px = id.xy;
    if (any(px >= uint2(pc.screenSize))) return;

    float2 uv      = (float2(px) + 0.5f) / pc.screenSize;
    float  current = rtaoRaw[int2(px)];
    float  depth   = depthBuffer[int2(px)];

    // Warmup: ramp alpha 1 → target over ~32 frames to avoid history pop on init.
    float warmup_alpha    = 1.0f / float(min(pc.frameIndex + 1u, 32u));
    float effective_alpha = max(pc.alpha, warmup_alpha);

    if (depth <= 0.0f) {
        aoResolved[int2(px)] = current;
        return;
    }

    float3 worldPos = worldPosFromDepth(uv, depth, pc.invViewProj);

    float4 prevClip = mul(pc.prevViewProj, float4(worldPos, 1.0f));
    float2 prevNdc  = prevClip.xy / prevClip.w;
    float2 prevUv   = float2(prevNdc.x * 0.5f + 0.5f, 0.5f - prevNdc.y * 0.5f);

    if (any(prevUv < float2(0.0f)) || any(prevUv > float2(1.0f))) {
        aoResolved[int2(px)] = current;
        return;
    }

    float history = aoHistory.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], prevUv, 0.0f);

    aoResolved[int2(px)] = lerp(history, current, effective_alpha);
}
