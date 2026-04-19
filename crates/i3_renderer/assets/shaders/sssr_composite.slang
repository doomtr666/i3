// SSSR Composite — replaces IBL specular with screen-space reflections.
//
// Correct energy-conserving blend:
//
//   ibl_spec  = prefiltered_env(R_mirror, roughness) × (F0×brdf.x + brdf.y) × ibl_intensity
//   ssr_spec  = ssr_resolved.rgb              × (F0×brdf.x + brdf.y) × ibl_intensity
//   hdr      += conf × (ssr_spec − ibl_spec)
//             = conf × (ssr_resolved.rgb − prefiltered_sample) × brdf_weight × ibl_intensity
//
// When conf=0 : HDR unchanged (IBL stays).
// When conf=1 : SSR fully replaces IBL (energy-neutral swap).
//
// The brdf_weight = (F0×brdf.x + brdf.y) is the same split-sum factor used by
// deferred_resolve.slang, so the two passes are consistent.
//
// If IBL indices are invalid (0xFFFFFFFF), falls back to:
//   hdr += conf × ssr_resolved.rgb × F0 × intensity

#include "common.slangh"

struct SssrCompositePc {
    float4x4 inv_view_proj;     // depth → world pos
    float3   camera_pos;
    float    intensity;         // user-facing multiplier (default 1.0)
    float2   screen_size;
    uint     ibl_lut_index;     // bindless index; 0xFFFFFFFF = no IBL
    uint     ibl_pref_index;    // bindless index; 0xFFFFFFFF = no IBL
    float    ibl_intensity;
    float    _pad;
};
[[vk::push_constant]] SssrCompositePc pc;

[[vk::binding(0, 0)]] Texture2D<float>    depthBuffer;
[[vk::binding(1, 0)]] Texture2D<float2>   gbufferNormal;
[[vk::binding(2, 0)]] Texture2D<float4>   gbufferAlbedo;
[[vk::binding(3, 0)]] Texture2D<float2>   gbufferRoughMetal;
[[vk::binding(4, 0)]] Texture2D<float4>   ssrResolved;
[[vk::binding(5, 0)]] RWTexture2D<float4> hdrTarget;

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID) {
    uint2 dims;
    hdrTarget.GetDimensions(dims.x, dims.y);
    if (id.x >= dims.x || id.y >= dims.y) return;

    int2 coord = int2(id.xy);

    float4 ssr = ssrResolved[coord];
    if (ssr.a <= 0.0f) return;

    float depth = depthBuffer[coord].r;
    if (depth <= 0.0f) return;

    // ── Reconstruct world-space position and view direction ───────────────────

    float2 uv  = (float2(id.xy) + 0.5f) / pc.screen_size;
    float  x   =  uv.x * 2.0f - 1.0f;
    float  y   = (1.0f - uv.y) * 2.0f - 1.0f;
    float4 wh  = mul(pc.inv_view_proj, float4(x, y, depth, 1.0f));
    float3 world_pos = wh.xyz / wh.w;

    float3 V = normalize(pc.camera_pos - world_pos);

    // ── PBR parameters ────────────────────────────────────────────────────────

    float3 N      = decodeOctahedral(gbufferNormal[coord].rg);
    float3 albedo = gbufferAlbedo[coord].rgb;
    float2 rm     = gbufferRoughMetal[coord].rg;
    float  roughness = rm.r;
    float  metallic  = rm.g;

    float3 F0  = lerp(float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float  NdotV = saturate(dot(N, V));

    // ── Compute the IBL specular that deferred_resolve already added ───────────

    float3 ibl_spec = float3(0.0f, 0.0f, 0.0f);
    if (pc.ibl_pref_index != 0xFFFFFFFF && pc.ibl_lut_index != 0xFFFFFFFF) {
        float3 R_mirror  = reflect(-V, N);
        float  mip       = roughness * 5.0f; // MAX_PREFILTER_MIP = 5
        float3 prefiltered = bindlessTextures[pc.ibl_pref_index]
            .SampleLevel(bindlessSamplers[SAMPLER_LINEAR], encodeHemiOcta(R_mirror), mip).rgb;
        float2 brdf = bindlessTextures[pc.ibl_lut_index]
            .SampleLevel(bindlessSamplers[SAMPLER_LINEAR], float2(NdotV, roughness), 0).rg;
        float3 brdf_weight = F0 * brdf.x + brdf.y;
        ibl_spec = prefiltered * brdf_weight * pc.ibl_intensity;

        // Replace IBL with SSR, blended by confidence.
        // hdr += conf × (ssr_spec − ibl_spec)
        float3 ssr_spec = ssr.rgb * brdf_weight * pc.ibl_intensity;
        float4 hdr = hdrTarget[coord];
        hdrTarget[coord] = float4(hdr.rgb + ssr.a * (ssr_spec - ibl_spec) * pc.intensity, hdr.a);
    } else {
        // No IBL: additive fallback weighted by F0.
        float4 hdr = hdrTarget[coord];
        hdrTarget[coord] = float4(hdr.rgb + ssr.a * ssr.rgb * F0 * pc.intensity, hdr.a);
    }
}
