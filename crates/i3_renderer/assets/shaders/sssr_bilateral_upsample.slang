// SSSR Bilateral Upsample — joint bilateral reconstruction from low-res SSR_Resolved.
//
// For each full-res pixel, samples a 2×2 neighbourhood in low-res space and weights
// each sample by depth and normal similarity (edge-stopping criteria).
// When downsample_factor == 1 the pass degenerates to a cheap copy.
#include "common.slangh"

struct BilateralUpsamplePc
{
    uint downsample_factor;
};
[[vk::push_constant]]
BilateralUpsamplePc pc;

[[vk::binding(0, 0)]]
Texture2D<float>    depthBuffer;   // full-res
[[vk::binding(1, 0)]]
Texture2D<float2>   gbufferNormal; // full-res, octahedral-encoded
[[vk::binding(2, 0)]]
Texture2D<float4>   ssrResolved;   // low-res input
[[vk::binding(3, 0)]]
RWTexture2D<float4> ssrUpsampled;  // full-res output

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 full_dims;
    ssrUpsampled.GetDimensions(full_dims.x, full_dims.y);
    if (id.x >= full_dims.x || id.y >= full_dims.y)
        return;

    int2  full_px = int2(id.xy);
    uint  f       = pc.downsample_factor;

    if (f <= 1u)
    {
        ssrUpsampled[full_px] = ssrResolved[full_px];
        return;
    }

    float  d_c = depthBuffer[full_px].r;
    float3 n_c = decodeOctahedral(gbufferNormal[full_px].rg);

    // Linearize center depth to view-space Z for scale-invariant comparison
    float4 vp_c  = mul(g_common.invProjection, float4(0.0f, 0.0f, d_c, 1.0f));
    float  vz_c  = abs(vp_c.z / vp_c.w);

    uint2 lo_dims;
    ssrResolved.GetDimensions(lo_dims.x, lo_dims.y);
    // Fractional position of this full-res pixel in low-res space.
    // lo_center is the nearest low-res pixel (used as the 3×3 kernel centre).
    float2 lo_float  = (float2(full_px) + 0.5f) / float(f);
    int2   lo_center = int2(floor(lo_float));

    float4 acc        = float4(0.0f);
    float  w_total    = 0.0f;
    // Initialise fallback to the centre sample — safest when all bilateral weights collapse
    float4 best_sample = ssrResolved[clamp(lo_center, int2(0, 0), int2(lo_dims) - 1)];
    float  best_w      = -1.0f;

    // 3×3 neighbourhood: Gaussian spatial (σ=0.5 lo-res px) + depth + normal edge-stopping
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
    {
        int2 lo = clamp(lo_center + int2(dx, dy), int2(0, 0), int2(lo_dims) - 1);
        int2 fp = clamp(lo * int(f) + int(f) / 2, int2(0, 0), int2(full_dims) - 1);

        float  d_s  = depthBuffer[fp].r;
        // d_s==0 → sky: invProj gives w=0 → NaN in vz_s, which propagates into acc/w_total.
        if (d_s <= 0.0f) continue;
        float4 vp_s = mul(g_common.invProjection, float4(0.0f, 0.0f, d_s, 1.0f));
        float  vz_s = abs(vp_s.z / vp_s.w);
        float3 n_s  = decodeOctahedral(gbufferNormal[fp].rg);

        // Spatial: Gaussian with σ=0.5 lo-res pixels → strongly prefers nearest sample
        float2 delta     = (float2(lo) + 0.5f) - lo_float;
        float  w_spatial = exp(-dot(delta, delta) * 2.0f); // 1/(2σ²) = 2

        // Edge-stopping: relative view-space depth + normal cosine
        float  depth_rel = abs(vz_c - vz_s) / max(vz_c, 0.1f);
        float  w_depth   = exp(-depth_rel * 20.0f);
        float  w_normal  = pow(saturate(dot(n_c, n_s)), 32.0f);
        float  w         = w_spatial * w_depth * w_normal;

        float4 s = ssrResolved[lo];
        acc     += s * w;
        w_total += w;

        if (w > best_w) { best_w = w; best_sample = s; }
    }

    // Fallback: nearest-match sample (not bilinear — bilinear bleeds across edges)
    ssrUpsampled[full_px] = (w_total > 1e-4f) ? acc / w_total : best_sample;
}
