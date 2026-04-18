// SSR Temporal Accumulation — reprojection + variance clipping.
//
// Key fix vs earlier version:
//   When the current frame has NO hit (current.a == 0), the old code wrote
//   float4(0) directly, wiping accumulated history.  On the next frame the
//   history was black and accumulation had to restart from scratch — this is
//   why the temporal appeared broken.
//   Fix: on no-hit, decay the reprojected history (alpha * history) rather
//   than overwriting with zero.  Reflections now fade gracefully when the
//   surface stops reflecting instead of popping to black.

#include "common.slangh"

struct SsrTemporalPushConstants {
    float4x4 invViewProj;
    float4x4 prevViewProj;
    float2   screenSize;
    float    alpha;
    uint     frameIndex;
};
[[vk::push_constant]] SsrTemporalPushConstants pc;

[[vk::binding(0, 0)]] Texture2D<float>    depthBuffer;
[[vk::binding(1, 0)]] Texture2D<float4>   ssrRaw;
[[vk::binding(2, 0)]] Texture2D<float4>   ssrHistory;
[[vk::binding(3, 0)]] RWTexture2D<float4> ssrResolved;

float3 worldPosFromDepth(float2 uv, float depth)
{
    float x =  uv.x * 2.0f - 1.0f;
    float y = (1.0f - uv.y) * 2.0f - 1.0f;
    float4 wp = mul(pc.invViewProj, float4(x, y, depth, 1.0f));
    return wp.xyz / wp.w;
}

// Clip history RGB into the AABB of the 3×3 neighbourhood of ssrRaw.
float4 varianceClip(float4 history, int2 px)
{
    float3 mn = float3( 1e9f,  1e9f,  1e9f);
    float3 mx = float3(-1e9f, -1e9f, -1e9f);
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++) {
        float3 s = ssrRaw.Load(int3(px + int2(dx, dy), 0)).rgb;
        mn = min(mn, s);
        mx = max(mx, s);
    }
    return float4(clamp(history.rgb, mn, mx), history.a);
}

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 px = id.xy;
    if (any(px >= uint2(pc.screenSize))) return;

    float2 uv      = (float2(px) + 0.5f) / pc.screenSize;
    float4 current = ssrRaw.Load(int3(px, 0));
    float  depth   = depthBuffer.Load(int3(px, 0)).r;

    // Cold-start: on frame 0 the history buffer is uninitialised (black).
    // Write current directly so the buffer fills in one frame rather than
    // fading up from black over many frames.
    float effective_alpha = pc.alpha;

    // Sky / far-plane pixels have no meaningful depth — no history to blend.
    if (depth <= 0.0f) {
        ssrResolved[int2(px)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // Cold-start: history buffer is black on frame 0 — write current directly.
    if (pc.frameIndex == 0u) {
        ssrResolved[int2(px)] = current;
        return;
    }

    // Reproject surface position to find where this pixel was last frame.
    float3 worldPos = worldPosFromDepth(uv, depth);
    float4 prevClip = mul(pc.prevViewProj, float4(worldPos, 1.0f));
    float2 prevNdc  = prevClip.xy / prevClip.w;
    float2 prevUv   = float2(prevNdc.x * 0.5f + 0.5f, 0.5f - prevNdc.y * 0.5f);

    // Off-screen last frame — no valid history.
    if (any(prevUv < 0.0f) || any(prevUv > 1.0f)) {
        ssrResolved[int2(px)] = current;
        return;
    }

    float4 history = ssrHistory.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], prevUv, 0.0f);

    // No hit this frame.
    // Distinguish between two cases:
    //   • Jitter miss  — some neighbours still have hits  → graceful decay
    //                    (avoids pop-to-black on isolated jitter misses)
    //   • Real occlusion — no neighbour has a hit either  → zero out immediately
    //                    (prevents the "trail behind moving objects" ghost)
    if (current.a <= 0.0f) {
        float neighbour_conf = 0.0f;
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            neighbour_conf = max(neighbour_conf,
                ssrRaw.Load(int3(px + int2(dx, dy), 0)).a);

        if (neighbour_conf <= 0.0f) {
            ssrResolved[int2(px)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            ssrResolved[int2(px)] = history * effective_alpha;
        }
        return;
    }

    // Normal blend: clip history to local neighbourhood, then lerp.
    float4 clipped    = varianceClip(history, int2(px));
    float3 col_blend  = lerp(current.rgb, clipped.rgb, effective_alpha);
    float  conf_blend = lerp(current.a,   clipped.a,   effective_alpha);

    ssrResolved[int2(px)] = float4(col_blend, conf_blend);
}
