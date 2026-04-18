// Screen-Space Reflections — Main compute pass
//
// Algorithm: HiZ-accelerated screen-space ray march with binary search refinement.
//
// Phase 1 — HiZ coarse search (check-then-advance):
//   Start at mip 1 (2-pixel cells).  When ray is in front, advance by one cell
//   and go coarser.  When ray is potentially behind, refine (mip-1) without
//   moving t.  Track t_prev = last position confirmed "in front".
//
// Phase 2 — binary search (8 iterations at mip 0):
//   Once mip reaches 0, binary-search between t_prev and t to find the exact
//   depth crossing.  Eliminates the cell-entry thickness failures that caused
//   most false-misses and the blocky banding pattern.
//
// Jitter: IGN per-pixel, rotated by golden-ratio each frame.
//   Range = one minimum-step (2 pixels) so temporal averages it cleanly.

#include "common.slangh"

struct SsrPushConstants
{
    float4x4 invProjection;
    float4x4 projection;
    float4x4 view;
    float2 screenSize;
    float maxDistance;
    float thickness;
    float roughnessCutoff;
    uint maxSteps;
    uint maxMip;
    uint frameIndex;
    uint enabled;
    uint _pad;
};

[[vk::push_constant]]
SsrPushConstants pc;

[[vk::binding(0, 0)]]
Texture2D<float> depthBuffer;
[[vk::binding(1, 0)]]
Texture2D<float2> gbufferNormal;
[[vk::binding(2, 0)]]
Texture2D<float2> gbufferRoughMetal;
[[vk::binding(3, 0)]]
Texture2D<float> hizPyramid;
[[vk::binding(4, 0)]]
Texture2D<float4> hdrTarget;
[[vk::binding(5, 0)]]
RWTexture2D<float4> ssrOutput;

// ─── Helpers ─────────────────────────────────────────────────────────────────

float3 viewSpacePos(uint2 px, float depth)
{
    float2 uv = (float2(px) + 0.5f) / pc.screenSize;
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float4 vh = mul(pc.invProjection, float4(ndc, depth, 1.0f));
    return vh.xyz / vh.w;
}

float3 viewToScreen(float3 P)
{
    float4 clip = mul(pc.projection, float4(P, 1.0f));
    if (clip.w <= 0.0f)
        return float3(-1.0f, -1.0f, -1.0f);
    float2 ndc = clip.xy / clip.w;
    return float3(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f, clip.z / clip.w);
}

float linearZ(float d)
{
    float4 v = mul(pc.invProjection, float4(0.0f, 0.0f, d, 1.0f));
    return v.z / v.w;
}

float ign(float2 px)
{
    return frac(52.9829189f * frac(0.06711056f * px.x + 0.00583715f * px.y));
}

float sampleHiZ(float2 uv, float mipF)
{
    return hizPyramid.SampleLevel(bindlessSamplers[SAMPLER_NEAREST_BORDER], uv, mipF).r;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 px = id.xy;
    if (any(px >= uint2(pc.screenSize)))
        return;

    if (pc.enabled == 0u)
    {
        ssrOutput[int2(px)] = 0;
        return;
    }

    float depth = depthBuffer.Load(int3(px, 0)).r;
    if (depth <= 0.0f)
    {
        ssrOutput[int2(px)] = 0;
        return;
    }

    float roughness = gbufferRoughMetal.Load(int3(px, 0)).r;
    if (roughness > pc.roughnessCutoff)
    {
        ssrOutput[int2(px)] = 0;
        return;
    }

    // ── Geometry ─────────────────────────────────────────────────────────────

    float3 P = viewSpacePos(px, depth);
    float3 N_world = decodeOctahedral(gbufferNormal.Load(int3(px, 0)).rg);
    float3x3 viewRot = float3x3(pc.view[0].xyz, pc.view[1].xyz, pc.view[2].xyz);
    float3 N = normalize(mul(viewRot, N_world));
    float3 V = normalize(-P);
    float3 R = reflect(-V, N);

    // ── Ray setup ─────────────────────────────────────────────────────────────

    float3 ss_start = viewToScreen(P + N * 0.05f);
    float3 ss_end = viewToScreen(P + R * pc.maxDistance);
    if (ss_start.z < 0.0f || ss_end.z < 0.0f)
    {
        ssrOutput[int2(px)] = 0;
        return;
    }

    float2 uv_delta = ss_end.xy - ss_start.xy;
    float ray_px = max(abs(uv_delta.x) * pc.screenSize.x,
                       abs(uv_delta.y) * pc.screenSize.y);
    if (ray_px < 1.0f)
    {
        ssrOutput[int2(px)] = 0;
        return;
    }

    // ── Phase 1: HiZ coarse search ────────────────────────────────────────────

    bool hit = false;
    float2 hit_uv = 0;
    float hit_t = 0;

    float noise = frac(ign(float2(px)) + float(pc.frameIndex) * 0.618033988f);
    int mip = 1;
    float t = noise * 2.0f / ray_px; // jitter: 0..2 pixel offset
    float t_prev = 0.0f;             // last confirmed "ray in front" position

    [loop]
    for (uint step = 0u; step < pc.maxSteps; step++)
    {
        float2 uv = ss_start.xy + uv_delta * t;
        if (any(uv < 0.0f) || any(uv > 1.0f) || t >= 1.0f)
            break;

        float ray_d = lerp(ss_start.z, ss_end.z, t);
        float hiz_d = sampleHiZ(uv, float(mip));

        if (ray_d > hiz_d)
        {
            // Ray in front of this HiZ cell — advance and coarsen.
            t_prev = t;
            t += exp2(float(mip)) / ray_px;
            mip = min(mip + 1, int(pc.maxMip));
        }
        else
        {
            // Potential intersection.
            if (mip <= 1)
            {
                // ── Phase 2: binary search between t_prev and t ───────────────
                // Converges to the exact depth crossing rather than using the
                // coarse cell-entry position — eliminates thickness-test failures.
                float t_lo = t_prev;
                float t_hi = t;
                [loop]
                for (uint b = 0u; b < 8u; b++)
                {
                    float t_mid = (t_lo + t_hi) * 0.5f;
                    float2 uv_mid = ss_start.xy + uv_delta * t_mid;
                    float rd_mid = lerp(ss_start.z, ss_end.z, t_mid);
                    float hz_mid = sampleHiZ(uv_mid, 0.0f);
                    if (rd_mid > hz_mid)
                    {
                        t_lo = t_mid;
                    }
                    else
                    {
                        t_hi = t_mid;
                    }
                }

                float2 f_uv = ss_start.xy + uv_delta * t_hi;
                float f_rd = lerp(ss_start.z, ss_end.z, t_hi);
                float f_hz = sampleHiZ(f_uv, 0.0f);

                if (abs(linearZ(f_rd) - linearZ(f_hz)) < pc.thickness)
                {
                    hit = true;
                    hit_uv = f_uv;
                    hit_t = t_hi;
                }
                break;
            }
            mip = max(mip - 1, 1); // refine, keep mip >= 1
        }
    }

    if (!hit)
    {
        ssrOutput[int2(px)] = 0;
        return;
    }

    // ── Confidence ────────────────────────────────────────────────────────────

    float2 edge_fade = smoothstep(0.0f, 0.05f, hit_uv) * smoothstep(1.0f, 0.95f, hit_uv);
    float conf = saturate(edge_fade.x * edge_fade.y * (1.0f - hit_t * hit_t));

    // ── Colour sample ─────────────────────────────────────────────────────────

    float3 color = hdrTarget.SampleLevel(
                                bindlessSamplers[SAMPLER_LINEAR], hit_uv, roughness * float(pc.maxMip))
                       .rgb;

    ssrOutput[int2(px)] = float4(color, conf);
}
