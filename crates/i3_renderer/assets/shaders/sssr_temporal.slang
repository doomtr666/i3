// Stochastic SSR — Temporal accumulation pass
//
// Exponential moving average over SSR_Raw frames.
// Identical logic to the previous SSR temporal pass; renamed for clarity.
//
// No-hit handling:
//   • Jitter miss (some neighbours still have hits) → graceful decay (alpha * history)
//   • Real occlusion (no neighbour has a hit)       → zero out immediately

#include "common.slangh"

struct SssrTemporalPc {
    float4x4 inv_view_proj;
    float4x4 prev_view_proj;
    float2   screen_size;
    float    alpha;
    uint     frame_index;
};
[[vk::push_constant]] SssrTemporalPc pc;

[[vk::binding(0, 0)]] Texture2D<float>    depthBuffer;
[[vk::binding(1, 0)]] Texture2D<float4>   ssrRaw;
[[vk::binding(2, 0)]] Texture2D<float4>   ssrHistory;
[[vk::binding(3, 0)]] RWTexture2D<float4> ssrResolved;

float3 worldPosFromDepth(float2 uv, float depth) {
    float x =  uv.x * 2.0f - 1.0f;
    float y = (1.0f - uv.y) * 2.0f - 1.0f;
    float4 wp = mul(pc.inv_view_proj, float4(x, y, depth, 1.0f));
    return wp.xyz / wp.w;
}

// Clip history colour into the 3×3 neighbourhood AABB of ssrRaw.
float4 varianceClip(float4 history, int2 px) {
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
void computeMain(uint3 id: SV_DispatchThreadID) {
    uint2 px = id.xy;
    if (any(px >= uint2(pc.screen_size))) return;

    float2 uv      = (float2(px) + 0.5f) / pc.screen_size;
    float4 current = ssrRaw.Load(int3(px, 0));
    float  depth   = depthBuffer.Load(int3(px, 0)).r;

    if (depth <= 0.0f) {
        ssrResolved[int2(px)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    // Cold-start: write current directly on frame 0.
    if (pc.frame_index == 0u) {
        ssrResolved[int2(px)] = current;
        return;
    }

    // Reproject surface to find previous-frame UV.
    float3 worldPos = worldPosFromDepth(uv, depth);
    float4 prevClip = mul(pc.prev_view_proj, float4(worldPos, 1.0f));
    float2 prevNdc  = prevClip.xy / prevClip.w;
    float2 prevUv   = float2(prevNdc.x * 0.5f + 0.5f, 0.5f - prevNdc.y * 0.5f);

    if (any(prevUv < 0.0f) || any(prevUv > 1.0f)) {
        ssrResolved[int2(px)] = current;
        return;
    }

    float4 history = ssrHistory.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], prevUv, 0.0f);

    // No-hit this frame.
    if (current.a <= 0.0f) {
        float neighbour_conf = 0.0f;
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            neighbour_conf = max(neighbour_conf,
                ssrRaw.Load(int3(px + int2(dx, dy), 0)).a);

        if (neighbour_conf <= 0.0f) {
            ssrResolved[int2(px)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            ssrResolved[int2(px)] = history * pc.alpha;
        }
        return;
    }

    // Normal blend: clip history, then EMA.
    float4 clipped    = varianceClip(history, int2(px));
    float3 col_blend  = lerp(current.rgb, clipped.rgb, pc.alpha);
    float  conf_blend = lerp(current.a,   clipped.a,   pc.alpha);

    ssrResolved[int2(px)] = float4(col_blend, conf_blend);
}
