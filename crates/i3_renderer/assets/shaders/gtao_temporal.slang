// GTAO Temporal Accumulation with reprojection
//
// For each pixel:
//   1. Reconstruct world-space position from depth + inv(VP_current).
//   2. Project into the previous frame with VP_prev to find the history UV.
//   3. If the reprojected UV is out of screen bounds → write raw sample directly.
//   4. Otherwise: resolved = lerp(history_at_reprojected_uv, current, alpha).
//
// Alpha ramps from 1.0 → target over the first 32 frames to avoid pop on init.

#include "common.slangh"

struct TemporalPushConstants {
    float4x4 invViewProj;   // current frame inv(V*P)
    float4x4 prevViewProj;  // previous frame V*P
    float2   screenSize;
    float    alpha;         // target blend weight (e.g. 0.1)
    uint     frameIndex;    // wrapping frame counter
};

[[vk::push_constant]]
TemporalPushConstants pc;

[[vk::binding(0, 0)]] Texture2D<float>   depthBuffer;
[[vk::binding(1, 0)]] Texture2D<float>   aoRaw;      // current frame raw GTAO
[[vk::binding(2, 0)]] Texture2D<float>   aoHistory;  // previous frame accumulated
[[vk::binding(3, 0)]] RWTexture2D<float> aoResolved; // output (also becomes next history)


[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 px = id.xy;
    if (any(px >= uint2(pc.screenSize))) return;

    float2 uv      = (float2(px) + 0.5f) / pc.screenSize;
    float  current = aoRaw[int2(px)];
    float  depth   = depthBuffer[int2(px)];

    // Warmup: ramp alpha from 1→target over ~32 frames to avoid history pop.
    float warmup_alpha   = 1.0f / float(min(pc.frameIndex + 1u, 32u));
    float effective_alpha = max(pc.alpha, warmup_alpha);

    // Sky / far-plane pixels have depth ≤ 0 in reverse-Z → skip reprojection.
    if (depth <= 0.0f) {
        aoResolved[int2(px)] = current;
        return;
    }

    // 1. Reconstruct world position.
    float3 worldPos = worldPosFromDepth(uv, depth, pc.invViewProj);

    // 2. Project into previous frame.
    float4 prevClip = mul(pc.prevViewProj, float4(worldPos, 1.0f));
    float2 prevNdc  = prevClip.xy / prevClip.w;
    // NDC → UV (Y-flip: NDC Y-up → UV Y-down)
    float2 prevUv = float2(prevNdc.x * 0.5f + 0.5f,
                           0.5f - prevNdc.y * 0.5f);

    // 3. Reject if reprojected UV is outside [0,1] (off-screen last frame).
    if (any(prevUv < float2(0.0f)) || any(prevUv > float2(1.0f))) {
        aoResolved[int2(px)] = current;
        return;
    }

    // 4. Sample history at reprojected position (bilinear).
    float history = aoHistory.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], prevUv, 0.0f);

    aoResolved[int2(px)] = lerp(history, current, effective_alpha);
}
