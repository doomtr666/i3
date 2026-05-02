// Stochastic Screen-Space Reflections — Sample pass (cone approximation)
//
// Traces ONE mirror ray per pixel. On hit, samples HDR_Mips at
// mip = roughness * max_mip_level to approximate the GGX lobe spread,
// coherent with IBL pre-filtering philosophy.
// On miss: falls back to the IBL environment map.
//
// Output: raw radiance (RGB) + alpha=1. The composite pass applies
// the BRDF weight (F0 * brdf_lut.x + brdf_lut.y) on top.

#include "lighting.slangh"
#include "sampling.slangh"

struct SssrSamplePc
{
    float thickness; // max view-space depth gap for hit acceptance
    uint enabled;
    uint downsample_factor; // 1 = full-res, 2 = half-res, etc.
    uint max_mip_level;     // highest mip to sample (roughness=1 → this mip)
    uint debug_mode;        // 0=normal 1=hit/iter/thickness 2=hit_uv 3=confidence 4=depth/hiz 5=hdrMips
};
[[vk::push_constant]]
SssrSamplePc pc;

[[vk::binding(0, 0)]]
Texture2D<float> depthBuffer;
[[vk::binding(1, 0)]]
Texture2D<float> hizBuffer;
[[vk::binding(2, 0)]]
Texture2D<float3> gbufferAlbedo;
[[vk::binding(3, 0)]]
Texture2D<float2> gbufferNormal;
[[vk::binding(4, 0)]]
Texture2D<float2> gbufferRoughMetal;
[[vk::binding(5, 0)]]
Texture2D<float4> hdrMips; // HDR_Mips: mip chain for roughness-dependent lookup
[[vk::binding(6, 0)]]
RWTexture2D<float4> ssrOutput;

#define TRACE_HIZ_MAX_ITER 256

// ─── Helpers ─────────────────────────────────────────────────────────────────

float3 projectViewToScreenSpace(float3 viewPos)
{
    float4 clip = mul(g_common.projection, float4(viewPos, 1.0f));
    clip.w = max(clip.w, 1e-6f);
    float2 ndc = clip.xy / clip.w;
    return float3(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f, clip.z / clip.w);
}

// Hi-Z tracing optimized for Reverse-Z (1.0 = Near, 0.0 = Far).
//
// Hierarchical DDA (cell-skipping quadtree):
//   Miss at mip N → escalate (mipLevel = min(N+1, MAX_MIP)) — advances a full N-cell boundary.
//   Hit at mip N > 0 → refine (mipLevel--).
//   Hit at mip 0 → thickness test: thin = valid hit, thick = reject & escalate.
//
// Thick-wall guard: rays inside solid geometry detect thickness > maxThickness at mip 0
// repeatedly. Each rejection escalates mipLevel by 1 (same as a miss) so the ray
// exponentially skips deeper into the solid. After 16 rejections the ray is clearly
// inside impenetrable geometry and returns false (falls back to IBL).
//
// Self-intersection filter (grazingZThreshold): for nearly horizontal surfaces
// (floor), the reflected ray travels close to the floor depth throughout the
// traversal. The segment test becomes borderline for floor cells → banded hits.
// Cells with cellMaxZ ≤ grazingZThreshold are treated as misses. For non-grazing
// surfaces (walls), grazingZThreshold = 0 (no filtering, preserves far-wall hits).
bool traceHiZ(
    float3 startUVZ,
    float3 rayDirUVZ,
    float4x4 invProj,
    float maxThickness,
    float grazingZThreshold, // skip cells with cellMaxZ ≤ this; 0 = no filter
    out float3 outHitUVZ,
    out uint outIter,
    out float outThicknessRatio)
{
    outIter = 0u;
    outThicknessRatio = 0.0f;

    const int MAX_MIP = 5;
    uint thick_count = 0u;
    float2 screenSize = g_common.GetScreenSize();

    float2 pos = startUVZ.xy * screenSize;
    float z = startUVZ.z;

    float2 dir = rayDirUVZ.xy * screenSize;
    float dz = rayDirUVZ.z;

    float rayLength = length(dir);
    if (rayLength < 0.001f)
        return false;

    dir.x = abs(dir.x) < 1e-5f ? (dir.x >= 0.0f ? 1e-5f : -1e-5f) : dir.x;
    dir.y = abs(dir.y) < 1e-5f ? (dir.y >= 0.0f ? 1e-5f : -1e-5f) : dir.y;
    float2 invDir = 1.0f / dir;

    int2 iStep = int2(dir.x >= 0.0f ? 1 : 0, dir.y >= 0.0f ? 1 : 0);

    const float CELL_EPS = 0.5f / rayLength;

    int mipLevel = MAX_MIP; // start coarse for efficient traversal

    for (uint i = 0; i < TRACE_HIZ_MAX_ITER; i++)
    {
        outIter = i;

        if (pos.x < 0.0f || pos.x >= screenSize.x ||
            pos.y < 0.0f || pos.y >= screenSize.y ||
            z < 0.0f || z > 1.0f)
            return false;

        int cellSize = 1 << mipLevel;
        int2 cellIndex = int2(pos) >> mipLevel;

        float2 nextEdge = float2((cellIndex.x + iStep.x) * cellSize,
                                 (cellIndex.y + iStep.y) * cellSize);
        float2 tToBoundary = (nextEdge - pos) * invDir;
        float tAdvance = min(tToBoundary.x, tToBoundary.y);

        float z1 = z;
        float z2 = z + dz * tAdvance;
        float minZ_segment = min(z1, z2);

        float cellMaxZ = hizBuffer.Load(int3(cellIndex, mipLevel)).r;

        // grazingZThreshold filters floor-depth cells for grazing reflections.
        // For non-grazing (threshold=0), this reduces to the sky-skip (cellMaxZ > 0).
        if (cellMaxZ > grazingZThreshold && minZ_segment <= cellMaxZ)
        {
            if (mipLevel == 0)
            {
                float2 cell_uv = (float2(cellIndex) + 0.5f) / screenSize;
                float ray_view_z = viewSpacePosFromUV(cell_uv, z, invProj).z;
                float geo_view_z = viewSpacePosFromUV(cell_uv, cellMaxZ, invProj).z;
                float thickness = abs(geo_view_z - ray_view_z);
                outThicknessRatio = thickness / maxThickness;

                if (thickness > maxThickness)
                {
                    bool xFirst = tToBoundary.x <= tToBoundary.y;
                    if (xFirst)
                    {
                        pos.x = nextEdge.x;
                        pos.y += dir.y * tAdvance;
                    }
                    else
                    {
                        pos.y = nextEdge.y;
                        pos.x += dir.x * tAdvance;
                    }
                    pos += dir * CELL_EPS;
                    z += dz * (tAdvance + CELL_EPS);
                    // Each rejection escalates by 1 mip level (exponential skip through solid).
                    // After 16 rejections the ray is in impenetrable geometry → return false.
                    thick_count++;
                    if (thick_count >= 16u)
                        return false;
                    mipLevel = min(mipLevel + 1, MAX_MIP);
                    continue;
                }

                outHitUVZ = float3(cell_uv, cellMaxZ);
                return true;
            }
            else
            {
                mipLevel--;
            }
        }
        else
        {
            bool xFirst = tToBoundary.x < tToBoundary.y;
            if (xFirst)
            {
                pos.x = nextEdge.x;
                pos.y += dir.y * tAdvance;
            }
            else
            {
                pos.y = nextEdge.y;
                pos.x += dir.x * tAdvance;
            }
            pos += dir * CELL_EPS;
            z += dz * (tAdvance + CELL_EPS);
            // mipLevel+1 on miss (NOT MAX_MIP): a miss at mip N advances by a full
            // mip-N cell (2^N pixels). Jumping to MAX_MIP instead would advance only
            // CELL_EPS (0.5px), causing 2^N / 0.5 = 64 cycles per cell at mip 4 alone.
            mipLevel = min(mipLevel + 1, MAX_MIP);
        }
    }

    return false;
}

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 px = id.xy;
    uint f = pc.downsample_factor;
    if (any(px >= uint2(g_common.GetScreenWidth() / f, g_common.GetScreenHeight() / f)))
        return;

    uint2 full_px = px * f + f / 2;

    // ── Input-stability debug (modes 4 / 5) — no trace needed ────────────────
    // mode 4: R=depthBuffer[full_px]  G=hizMip2 at same cell.
    //   If R or G flicker → GBuffer depth or HiZ build is non-deterministic.
    // mode 5: hdrMips mip 2 at current pixel (scaled for visibility).
    //   If it flickers → HDR SPD produces non-deterministic mips.
    if (pc.debug_mode == 4u)
    {
        float depth_raw = depthBuffer.Load(int3(full_px, 0)).r;
        float hiz_m2 = hizBuffer.Load(int3(int2(full_px) >> 2, 2)).r;
        // frac(x * 128) maps 1/128 ≈ 0.008 depth range to a full 0→1 band.
        // Any flicker shows as a large color jump rather than a tiny shade change.
        ssrOutput[px] = float4(depth_raw, frac(hiz_m2 * 128.0f), 0.0f, 1.0f);
        return;
    }
    if (pc.debug_mode == 5u)
    {
        float3 hdr_m2 = hdrMips.Load(int3(int2(full_px) >> 2, 2)).rgb * 0.05f;
        ssrOutput[px] = float4(saturate(hdr_m2), 1.0f);
        return;
    }

    if (pc.enabled == 0u)
    {
        ssrOutput[px] = 0;
        return;
    }

    float depth = depthBuffer.Load(int3(full_px, 0)).r;
    if (depth <= 0.0f)
    {
        ssrOutput[int2(px)] = 0;
        return;
    }

    float2 rm = gbufferRoughMetal.Load(int3(full_px, 0)).rg;
    float roughness = rm.r;
    float metallic = rm.g;

    float3 P = viewSpacePosFromPixel(full_px, depth, g_common.GetScreenSize(), g_common.invProjection);
    float3 N_world = decodeOctahedral(gbufferNormal.Load(int3(full_px, 0)).rg);

    float3x3 viewRot = float3x3(g_common.view[0].xyz, g_common.view[1].xyz, g_common.view[2].xyz);
    float3x3 invViewRot = float3x3(g_common.invView[0].xyz, g_common.invView[1].xyz, g_common.invView[2].xyz);
    float3 N = normalize(mul(viewRot, N_world));
    float3 V = normalize(-P);

    // Mirror reflection direction in view space
    float3 D = reflect(-V, N);
    float3 D_world = normalize(mul(invViewRot, D));

    const float MAX_DISTANCE = 200.0f;

    // Default: miss → composite uses its own IBL specular (eff_conf = 0).
    float4 out = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (dot(D, N) > 0.001f)
    {
        float3 P_end = P + D * MAX_DISTANCE;
        float3 screen_P = projectViewToScreenSpace(P);
        float3 screen_P_end = projectViewToScreenSpace(P_end);

        float3 screen_RayDir = normalize(screen_P_end - screen_P);
        float2 screen_RayDir_Pixels = screen_RayDir.xy * g_common.GetScreenSize();
        float step_scale = 1.0f / max(abs(screen_RayDir_Pixels.x), abs(screen_RayDir_Pixels.y));
        float3 ray_step_screen = screen_RayDir * step_scale;

        // Normal bias proportional to view-space distance: lifts the start position
        // above the surface in Reverse-Z. Proportional scaling keeps the bias
        // meaningful at both near (1m → 5mm) and far (10m → 5cm) distances.
        float surface_dist = length(P);
        float3 P_biased = P + N * (surface_dist * 0.005f);
        float3 screen_P_biased = projectViewToScreenSpace(P_biased);
        float3 safe_start = screen_P_biased + ray_step_screen * 4.0f;

        // Grazing self-intersection filter: for nearly horizontal surfaces (floor),
        // the reflected ray travels at ~floor depth for the entire traversal — the
        // segment test is borderline for every floor HiZ cell → banded hits.
        // grazingZThreshold makes those cells permanent misses regardless of ray progress.
        // Gated by N_world.y so walls (N_world.y≈0) keep threshold=0 and correctly
        // reflect objects farther than the wall.
        float dotDN = dot(D, N);                       // > 0.001 guaranteed above
        float grazing = saturate(1.0f - dotDN / 0.2f); // 1 at tangent, 0 at ≥11°
        float horiz = saturate(N_world.y * 4.0f);      // 1 for floor, 0 for wall
        float grazingZThreshold = grazing * horiz * screen_P_biased.z;

        float3 hit_UVZ;
        uint dbg_iter;
        float dbg_thickness_ratio;
        bool valid_hit = traceHiZ(safe_start, ray_step_screen, g_common.invProjection, pc.thickness,
                                  grazingZThreshold, hit_UVZ, dbg_iter, dbg_thickness_ratio);

        // ── Debug viz ─────────────────────────────────────────────────────────
        // mode 1: R=hit  G=iter/96  B=thickness_ratio
        //   Flickering R → trace oscillates. G → iteration count. B~1 → thickness borderline.
        // mode 2: RG = frac(hit_uv * screenSize * 0.5)  B=hit
        //   Flickering RG with stable B → hit UV oscillates between adjacent pixels.
        if (pc.debug_mode == 1u)
        {
            ssrOutput[px] = float4(
                valid_hit ? 1.0f : 0.0f,
                float(dbg_iter) / float(TRACE_HIZ_MAX_ITER),
                saturate(dbg_thickness_ratio),
                1.0f);
            return;
        }
        if (pc.debug_mode == 2u)
        {
            float2 hit_uv_dbg = valid_hit ? hit_UVZ.xy : float2(0.0f);
            float2 px_coord = frac(hit_uv_dbg * g_common.GetScreenSize() * 0.5f);
            ssrOutput[px] = float4(px_coord.x, px_coord.y, valid_hit ? 1.0f : 0.0f, 1.0f);
            return;
        }

        if (valid_hit)
        {
            // Backface validation: reject hits where we'd see the back of a surface
            uint2 hit_px = uint2(hit_UVZ.xy * g_common.GetScreenSize());
            float3 hit_N_world = decodeOctahedral(gbufferNormal.Load(int3(hit_px, 0)).rg);
            float3 hit_N = normalize(mul(viewRot, hit_N_world));

            if (dot(hit_N, D) < -0.001f)
            {
                // View-space distance from surface to hit point.
                float3 hit_P = viewSpacePosFromUV(hit_UVZ.xy, hit_UVZ.z, g_common.invProjection);
                float hit_dist = length(hit_P - P);
                float alpha_sq = roughness * roughness;

                // Mip selection: GGX alpha + distance amplification (over MAX_DISTANCE).
                float dist_norm_mip = saturate(hit_dist / MAX_DISTANCE);
                float mip = clamp(alpha_sq * float(pc.max_mip_level) * (1.0f + dist_norm_mip),
                                  0.0f, float(pc.max_mip_level));

                float3 hit_color = hdrMips.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], hit_UVZ.xy, mip).rgb;

                // Confidence: screen-edge fade × distance fade × cone fade.
                // GGX cone footprint in screen pixels = alpha_sq × screen_dist_px.
                // At contact (screen_dist_px=0): cone_fade=1 for all roughness.
                // When footprint exceeds the mip chain resolution → fall back to IBL.
                float2 screen_delta = hit_UVZ.xy - screen_P.xy;
                float screen_dist_px = length(screen_delta * g_common.GetScreenSize());
                float max_footprint_px = float(1u << pc.max_mip_level);
                float cone_fade = saturate(1.0f - alpha_sq * screen_dist_px / max_footprint_px);

                float2 edge = min(hit_UVZ.xy, 1.0f - hit_UVZ.xy);
                float edge_fade = saturate(min(edge.x, edge.y) / 0.05f);
                float dist_norm_fade = saturate(hit_dist / 50.0f);
                float dist_fade = 1.0f - dist_norm_fade;
                float confidence = edge_fade * dist_fade * cone_fade;

                // mode 3: confidence as grayscale — no color sampling.
                //   Flickering here (but stable mode 1/2) → confidence formula is the source.
                //   Stable here but flickering in normal mode → hdrMips.SampleLevel is the source.
                if (pc.debug_mode == 3u)
                    out = float4(confidence, confidence, confidence, 1.0f);
                else
                    out = float4(hit_color, confidence);
            }
        }
    }

    // Output raw radiance + confidence (alpha).
    // Composite blends SSR vs IBL via eff_conf = saturate(ssr.a * intensity).
    ssrOutput[px] = out;
}
