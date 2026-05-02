#include "common.slangh"

struct SssrTemporalPc
{
    float alpha;
    uint  downsample_factor;
    float compression_factor;
    float gamma;
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

float3 Compress(float3 c, float k)
{
    float luminance = max(max(c.r, c.g), c.b);
    return c / (1.0f + luminance * k);
}

float3 Decompress(float3 c, float k)
{
    float luminance = max(max(c.r, c.g), c.b);
    return c / max(1.0f - luminance * k, 1e-5f);
}

// Analytical directional clipping — slab method (clips prev_sample onto AABB surface).
// Previous formula max(rmax/max(r,eps), rmin/min(r,-eps)) was wrong: for r.x>0,
// min(r.x,-eps)=-eps so rmin.x/(-eps)>>1, dominating the max and making t_final=1 always.
float3 ClipAABB(float3 aabb_min, float3 aabb_max, float3 prev_sample, float3 avg)
{
    float3 r    = prev_sample - avg;
    float3 rmin = aabb_min - avg;
    float3 rmax = aabb_max - avg;

    const float eps = 1e-5f;
    // Clamp near-zero components to avoid /0 (gives INF → non-constraining axis, correct)
    float3 r_safe = float3(
        abs(r.x) < eps ? (r.x >= 0.0f ? eps : -eps) : r.x,
        abs(r.y) < eps ? (r.y >= 0.0f ? eps : -eps) : r.y,
        abs(r.z) < eps ? (r.z >= 0.0f ? eps : -eps) : r.z
    );
    float3 t1 = rmin / r_safe;
    float3 t2 = rmax / r_safe;
    float  t_final = saturate(min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z)));

    return avg + r * t_final;
}

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 px = id.xy;
    int w = g_common.GetScreenWidth();
    int h = g_common.GetScreenHeight();
    int lw = w / int(pc.downsample_factor);
    int lh = h / int(pc.downsample_factor);

    if (any(px >= uint2(lw, lh)))
        return;

    // Sample depth at the center of the full-res block this low-res pixel represents
    int2 full_px = int2(px * pc.downsample_factor + pc.downsample_factor / 2);
    float depth = depthBuffer.Load(int3(full_px, 0)).r;
    if (depth <= 0.0f)
    {
        ssrResolved[int2(px)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float4 current = ssrRaw.Load(int3(px, 0));

    // AABB built in k=1 space — must happen before cold-start so even frame 0 is clean.
    float3 m1 = float3(0.0f);
    float3 m2 = float3(0.0f);
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++)
    {
        int2 nb = clamp(int2(px) + int2(x, y), int2(0), int2(lw - 1, lh - 1));
        float3 c = Compress(ssrRaw.Load(int3(nb, 0)).rgb, 1.0f);
        m1 += c;
        m2 += c * c;
    }
    float3 mu    = m1 / 9.0f;
    float3 sigma = sqrt(max(m2 / 9.0f - mu * mu, 0.0f));
    float3 box_min = mu - pc.gamma * sigma;
    float3 box_max = mu + pc.gamma * sigma;

    // Clip current frame to AABB: prevents firefly injection regardless of downsample factor.
    // A single bright ray sample outside the neighbourhood box is almost always a firefly.
    float3 current_hdr = Decompress(ClipAABB(box_min, box_max, Compress(current.rgb, 1.0f), mu), 1.0f);

    // Cold start: store the cleaned sample directly
    if (g_common.GetFrameIndex() == 0u)
    {
        ssrResolved[int2(px)] = float4(current_hdr, current.a);
        return;
    }

    // Reprojection (UV from low-res pixel centre — avoids diagonal offset with downsampling)
    float2 uv = (float2(px) + 0.5f) / float2(lw, lh);
    float3 worldPos = worldPosFromDepth(uv, depth, g_common.invViewProjection);
    float4 prevClip = mul(g_common.prevViewProjection, float4(worldPos, 1.0f));
    float2 prevNdc  = prevClip.xy / prevClip.w;
    float2 prevUv   = float2(prevNdc.x * 0.5f + 0.5f, 0.5f - prevNdc.y * 0.5f);

    if (any(prevUv < 0.0f) || any(prevUv > 1.0f))
    {
        ssrResolved[int2(px)] = float4(current_hdr, current.a);
        return;
    }

    // Clip history in the same k=1 space, then bring back to HDR.
    // Sample alpha to detect reprojection onto background (alpha=0): bilinear sampling
    // mixes geometry (a=1) and background (a=0) at edges, so we weight by it to prevent
    // pulling SSR toward zero when the camera moves and edges slide over background pixels.
    float4 history_s   = ssrHistory.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], prevUv, 0.0f);
    float3 history     = history_s.rgb;
    float3 clipped_hdr = Decompress(ClipAABB(box_min, box_max, Compress(history, 1.0f), mu), 1.0f);

    // EMA blend in user-controlled compression space.
    // Modulate alpha by history validity: 0 where history was background → fall back to current.
    float  k              = pc.compression_factor;
    float  effective_alpha = pc.alpha * saturate(history_s.a);
    float3 final_color    = Decompress(lerp(Compress(current_hdr, k), Compress(clipped_hdr, k), effective_alpha), k);

    float final_a = lerp(current.a, history_s.a, effective_alpha);
    ssrResolved[int2(px)] = float4(final_color, final_a);
}
