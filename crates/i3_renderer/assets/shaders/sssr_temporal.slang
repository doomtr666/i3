// Stochastic SSR — Temporal accumulation pass
//
// Exponential moving average over SSR_Raw frames.
// Identical logic to the previous SSR temporal pass; renamed for clarity.
//
// No-hit handling:
//   • Jitter miss (some neighbours still have hits) → graceful decay (alpha * history)
//   • Real occlusion (no neighbour has a hit)       → zero out immediately

#include "common.slangh"

struct SssrTemporalPc
{
    float alpha;
};
[[vk::push_constant]]
SssrTemporalPc pc;

[[vk::binding(0, 0)]]
Texture2D<float> depthBuffer;
[[vk::binding(1, 0)]]
Texture2D<float4> ssrRaw;
[[vk::binding(2, 0)]]
Texture2D<float4> ssrHistory;
[[vk::binding(3, 0)]]
Texture2D<float4> hdrLit;
[[vk::binding(4, 0)]]
RWTexture2D<float4> ssrResolved;

float3 worldPosFromDepth(float2 uv, float depth)
{
    float x = uv.x * 2.0f - 1.0f;
    float y = (1.0f - uv.y) * 2.0f - 1.0f;
    float4 wp = mul(g_common.invViewProjection, float4(x, y, depth, 1.0f));
    return wp.xyz / wp.w;
}

// 1. Mise à jour de la boîte de variance
float4 varianceClip(float4 history, int2 px, float3 current_color, float current_conf)
{
    if (current_conf <= 0.0f)
        return history;

    float3 mn = current_color;
    float3 mx = current_color;

    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            if (dx == 0 && dy == 0)
                continue;

            float4 raw_neighbor = ssrRaw.Load(int3(px + int2(dx, dy), 0));

            if (raw_neighbor.a > 0.0f)
            {
                // Fetch direct sur la cible HDR courante via les UV de l'impact
                float3 s = hdrLit.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], raw_neighbor.xy, 0.0f).rgb;
                mn = min(mn, s);
                mx = max(mx, s);
            }
        }
    }
    return float4(clamp(history.rgb, mn, mx), history.a);
}

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 px = id.xy;
    if (any(px >= uint2(g_common.GetScreenWidth(), g_common.GetScreenHeight())))
        return;

    float depth = depthBuffer.Load(int3(px, 0)).r;
    if (depth <= 0.0f)
    {
        ssrResolved[int2(px)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float2 uv = (float2(px) + 0.5f) / float2(g_common.GetScreenWidth(), g_common.GetScreenHeight());

    // 1. On lit le résultat brut du raymarcher
    float4 raw_data = ssrRaw.Load(int3(px, 0));
    float2 hit_uv = raw_data.xy;
    float confidence = raw_data.a; // C'est CA la vraie confiance !

    // 2. On prépare la couleur courante (noire par défaut)
    float4 current = float4(0.0f, 0.0f, 0.0f, confidence);

    // 3. On ne lit le HDR que si le rayon a vraiment touché quelque chose
    if (confidence > 0.0f)
    {
        current.rgb = hdrLit.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], hit_uv, 0.0f).rgb;
    }

    // Cold-start: write current directly on frame 0.
    if (g_common.GetFrameIndex() == 0u)
    {
        ssrResolved[int2(px)] = current;
        return;
    }

    // Reproject surface to find previous-frame UV.
    float3 worldPos = worldPosFromDepth(uv, depth);
    float4 prevClip = mul(g_common.prevViewProjection, float4(worldPos, 1.0f));
    float2 prevNdc = prevClip.xy / prevClip.w;
    float2 prevUv = float2(prevNdc.x * 0.5f + 0.5f, 0.5f - prevNdc.y * 0.5f);

    if (any(prevUv < 0.0f) || any(prevUv > 1.0f))
    {
        ssrResolved[int2(px)] = current;
        return;
    }

    float4 history = ssrHistory.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], prevUv, 0.0f);

    // --- LE BLOC NO-HIT EST OBLIGATOIRE ---
    // Si ce pixel n'a pas de reflet valide, on doit le laisser s'éteindre doucement.
    if (confidence <= 0.0f)
    {
        float neighbour_conf = 0.0f;
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                neighbour_conf = max(neighbour_conf, ssrRaw.Load(int3(px + int2(dx, dy), 0)).a);
            }
        }

        if (neighbour_conf <= 0.0f)
        {
            // Vrai vide : on coupe tout
            ssrResolved[int2(px)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        else
        {
            // Vide dû au bruit stochastique : on garde l'historique
            ssrResolved[int2(px)] = history * pc.alpha;
        }
        return;
    }

    // Normal blend
    float4 clipped = varianceClip(history, int2(px), current.rgb, confidence);
    float3 col_blend = lerp(current.rgb, clipped.rgb, pc.alpha);
    float conf_blend = lerp(current.a, clipped.a, pc.alpha);

    ssrResolved[int2(px)] = float4(col_blend, conf_blend);
}
