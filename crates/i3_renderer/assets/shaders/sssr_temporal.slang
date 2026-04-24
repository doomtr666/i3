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
Texture2D<float4> gbufferNormal;
[[vk::binding(4, 0)]]
Texture2D<float4> gbufferRoughMetal;
[[vk::binding(5, 0)]]
Texture2D<float4> hdrLit;
[[vk::binding(6, 0)]]
RWTexture2D<float4> ssrResolved;

static const float COMPRESSION_FACTOR = 0.1f;

float3 Compress(float3 c)
{
    float luminance = max(max(c.r, c.g), c.b);
    return c / (1.0f + luminance * COMPRESSION_FACTOR);
}

float3 Decompress(float3 c)
{
    float luminance = max(max(c.r, c.g), c.b);
    // Sécurité pour éviter la division par zéro
    return c / max(1.0f - luminance * COMPRESSION_FACTOR, 1e-5f);
}

// Fonction analytique d'écrêtage directionnel
float3 ClipAABB(float3 aabb_min, float3 aabb_max, float3 prev_sample, float3 avg)
{
    float3 r = prev_sample - avg;
    float3 rmax = aabb_max - avg;
    float3 rmin = aabb_min - avg;

    const float eps = 1e-5f;

    // Calcul de l'intersection du rayon vers la boîte de variance
    float3 t = max(rmax / max(r, eps), rmin / min(r, -eps));
    float t_final = saturate(min(min(t.x, t.y), t.z));

    return avg + r * t_final;
}

float3 worldPosFromDepth(float2 uv, float depth)
{
    float x = uv.x * 2.0f - 1.0f;
    float y = (1.0f - uv.y) * 2.0f - 1.0f;
    float4 wp = mul(g_common.invViewProjection, float4(x, y, depth, 1.0f));
    return wp.xyz / wp.w;
}

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 px = id.xy;
    int w = g_common.GetScreenWidth();
    int h = g_common.GetScreenHeight();

    if (any(px >= uint2(w, h)))
        return;

    float depth = depthBuffer.Load(int3(px, 0)).r;
    if (depth <= 0.0f)
    {
        ssrResolved[int2(px)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float2 uv = (float2(px) + 0.5f) / float2(w, h);
    float4 current = ssrRaw.Load(int3(px, 0));

    // Démarrage à froid
    if (g_common.GetFrameIndex() == 0u)
    {
        ssrResolved[int2(px)] = current;
        return;
    }

    // Reprojection
    float3 worldPos = worldPosFromDepth(uv, depth);
    float4 prevClip = mul(g_common.prevViewProjection, float4(worldPos, 1.0f));
    float2 prevNdc = prevClip.xy / prevClip.w;
    float2 prevUv = float2(prevNdc.x * 0.5f + 0.5f, 0.5f - prevNdc.y * 0.5f);

    if (any(prevUv < 0.0f) || any(prevUv > 1.0f))
    {
        ssrResolved[int2(px)] = current;
        return;
    }

    // --- 1. Statistiques spatiales (Voisinage 3x3 en espace compressé) ---
    float3 m1 = float3(0.0f);
    float3 m2 = float3(0.0f);

    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            // Protection des bords d'écran
            int2 neighbor_px = clamp(int2(px) + int2(x, y), int2(0), int2(w - 1, h - 1));
            float3 sample_color = ssrRaw.Load(int3(neighbor_px, 0)).rgb;

            float3 compressed_color = Compress(sample_color);

            m1 += compressed_color;
            m2 += compressed_color * compressed_color;
        }
    }

    float3 mu = m1 / 9.0f;
    float3 variance = max(m2 / 9.0f - mu * mu, 0.0f);
    float3 sigma = sqrt(variance);

    float gamma = 1.5f;
    float3 box_min = mu - gamma * sigma;
    float3 box_max = mu + gamma * sigma;

    // --- 2. Lecture et écrêtage de l'historique ---
    float3 history = ssrHistory.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], prevUv, 0.0f).rgb;
    float3 compressed_history = Compress(history);

    float3 clipped_history = ClipAABB(box_min, box_max, compressed_history, mu);

    // --- 3. Mélange exponentiel et décompression ---
    float3 compressed_current = Compress(current.rgb);
    float3 resolved_compressed = lerp(compressed_current, clipped_history, pc.alpha);

    float3 final_color = Decompress(resolved_compressed);

    ssrResolved[int2(px)] = float4(final_color, 1.0f);
}
