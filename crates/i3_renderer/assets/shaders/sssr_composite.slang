// SSSR Composite — owns all specular indirect lighting.
//
// deferred_resolve writes only diffuse IBL + direct light.
// This pass writes specular IBL blended with SSR for every surface pixel:
//
//   ibl_spec  = prefiltered(R, roughness) × brdf_weight × ibl_intensity × ao
//   ssr_spec  = ssr_resolved.rgb          × brdf_weight
//   eff_conf  = saturate(ssr.a × intensity)
//   hdr      += lerp(ibl_spec, ssr_spec, eff_conf)
//
//   eff_conf = 0  →  pure IBL specular (SSR disabled or no hit)
//   eff_conf = 1  →  pure SSR specular
//
// The lerp is always non-negative: both ibl_spec and ssr_spec are >= 0.
// No subtraction → no risk of negative HDR values.
//
// Note: ssr.rgb is in scene-radiance units (sampled from HDR_Target).
//   Do NOT multiply by ibl_intensity — that would double-scale it.
//   Do NOT multiply by ao — ao attenuates hemisphere-integrated IBL;
//   a traced ray naturally handles occlusion by hitting the occluder.

#include "common.slangh"

struct SssrCompositePc
{
    float intensity;
};
[[vk::push_constant]]
SssrCompositePc pc;

[[vk::binding(0, 0)]]
Texture2D<float> depthBuffer;
[[vk::binding(1, 0)]]
Texture2D<float2> gbufferNormal;
[[vk::binding(2, 0)]]
Texture2D<float4> gbufferAlbedo;
[[vk::binding(3, 0)]]
Texture2D<float2> gbufferRoughMetal;
[[vk::binding(4, 0)]]
Texture2D<float4> ssrUpsampled;
[[vk::binding(5, 0)]]
Texture2D<float> aoResolved;
[[vk::binding(6, 0)]]
RWTexture2D<float4> hdrTarget;

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 dims;
    hdrTarget.GetDimensions(dims.x, dims.y);
    if (id.x >= dims.x || id.y >= dims.y)
        return;

    int2 coord = int2(id.xy);

    float depth = depthBuffer[coord].r;
    if (depth <= 0.0f)
        return; // sky / background — no specular indirect

    // ── Reconstruct world position and view direction ─────────────────────────

    float2 uv = (float2(id.xy) + 0.5f) / float2(g_common.GetScreenWidth(), g_common.GetScreenHeight());
    float x = uv.x * 2.0f - 1.0f;
    float y = (1.0f - uv.y) * 2.0f - 1.0f;
    float4 wh = mul(g_common.invViewProjection, float4(x, y, depth, 1.0f));
    float3 world_pos = wh.xyz / wh.w;
    float3 V = normalize(g_common.GetCameraPos() - world_pos);

    // ── PBR parameters ────────────────────────────────────────────────────────

    float3 N = decodeOctahedral(gbufferNormal[coord].rg);
    float3 albedo = gbufferAlbedo[coord].rgb;
    float2 rm = gbufferRoughMetal[coord].rg;
    float roughness = rm.r;
    float alpha     = roughness * roughness;
    float metallic  = rm.g;

    float3 F0 = lerp(FRESNEL_F0_DIELECTRIC, albedo, metallic);
    float NdotV = saturate(dot(N, V));
    float ao = aoResolved[coord].r;

    // SSR confidence: 0 when no hit or roughness above cutoff.
    float4 ssr = ssrUpsampled[coord];
    float eff_conf = saturate(ssr.a * pc.intensity);

    // ── Specular indirect ─────────────────────────────────────────────────────

    float4 hdr = hdrTarget[coord];

    if (g_common.GetIblPrefIndex() != 0 && g_common.GetIblLutIndex() != 0)
    {
        float3 R_mirror = reflect(-V, N);
        float mip = alpha * 5.0f;
        float3 prefiltered = bindlessTextures[g_common.GetIblPrefIndex()]
                                 .SampleLevel(bindlessSamplers[SAMPLER_LINEAR], encodeHemiOcta(R_mirror), mip)
                                 .rgb;
        float2 brdf = bindlessTextures[g_common.GetIblLutIndex()]
                          .SampleLevel(bindlessSamplers[SAMPLER_LINEAR], float2(NdotV, roughness), 0)
                          .rg;
        float3 brdf_weight = F0 * brdf.x + brdf.y;

        // ibl_spec includes ao to match deferred_resolve's diffuse term scaling.
        // ssr_spec has no ibl_intensity (ssr.rgb is already in scene-radiance units)
        // and no ao (a traced ray handles occlusion naturally).
        float3 ibl_spec = prefiltered * brdf_weight * g_common.GetIblIntensity() * ao;
        float3 ssr_spec = ssr.rgb * brdf_weight;

        hdrTarget[coord] = float4(hdr.rgb + lerp(ibl_spec, ssr_spec, eff_conf), hdr.a);
    }
    else
    {

        // No IBL: add SSR additively weighted by F0 when there is a hit.
        hdrTarget[coord] = float4(hdr.rgb + eff_conf * ssr.rgb * F0, hdr.a);
    }
}
