// SSR Composite — blends SSR_Resolved onto HDR_Target (mip 0, in-place).
//
// Weight = F0 × (1 − roughness) × confidence × intensity
//   F0 = lerp(0.04, albedo, metallic)
//
// No Fresnel angle-boost: the quintic Schlick spike at grazing angles
// produces pathological over-reflection on dielectrics that is hard to
// calibrate and visually wrong for SSR (which is an approximation).
// The intensity slider provides artistic control.
//
// Binding layout (set 0):
//   0 : gbufferAlbedo      (sampled)
//   1 : gbufferRoughMetal  (sampled)
//   2 : ssrResolved        (sampled, rgb=colour a=confidence)
//   3 : hdrTarget          (storage, mip 0 in-place read-write)
#include "common.slangh"

struct PushConstants { float intensity; float _pad0; float _pad1; float _pad2; };
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] Texture2D<float4>   gbufferAlbedo;
[[vk::binding(1, 0)]] Texture2D<float4>   gbufferRoughMetal;
[[vk::binding(2, 0)]] Texture2D<float4>   ssrResolved;
[[vk::binding(3, 0)]] RWTexture2D<float4> hdrTarget;

[shader("compute")]
[numthreads(8, 8, 1)]
void compositeMain(uint3 tid : SV_DispatchThreadID)
{
    uint2 dims;
    hdrTarget.GetDimensions(dims.x, dims.y);
    if (tid.x >= dims.x || tid.y >= dims.y) return;

    int2 coord = int2(tid.xy);

    float4 ssr = ssrResolved[coord];
    if (ssr.a <= 0.0f) return;

    float4 rm       = gbufferRoughMetal[coord];
    float3 albedo   = gbufferAlbedo[coord].rgb;
    float3 f0       = lerp(float3(0.04f, 0.04f, 0.04f), albedo, rm.g);
    float3 weight   = f0 * (1.0f - rm.r) * ssr.a * pc.intensity;

    float4 hdr = hdrTarget[coord];
    hdrTarget[coord] = float4(hdr.rgb + ssr.rgb * weight, hdr.a);
}
