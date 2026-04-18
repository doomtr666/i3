// Bloom Composite — additive blend of Bloom_Buffer mip 0 onto HDR_Target mip 0.
//
// Operates in HDR linear space, before tonemapping.
// In-place storage read-write, identical pattern to ssr_composite.slang.
//
// Binding layout (set 0):
//   0 : bloomSrc   (sampled, Bloom_Buffer mip 0)
//   1 : hdrTarget  (storage, HDR_Target mip 0 — in-place read-write)

#include "common.slangh"

struct BloomCompositePc { float intensity; float _pad0; float _pad1; float _pad2; };
[[vk::push_constant]] BloomCompositePc pc;

[[vk::binding(0, 0)]] Texture2D<float4>   bloomSrc;
[[vk::binding(1, 0)]] RWTexture2D<float4> hdrTarget;

[numthreads(8, 8, 1)]
void compositeMain(uint3 tid : SV_DispatchThreadID)
{
    uint2 dims;
    hdrTarget.GetDimensions(dims.x, dims.y);
    if (any(tid.xy >= dims)) return;

    int2 coord = int2(tid.xy);

    float2 uv    = (float2(coord) + 0.5f) / float2(dims);
    float3 bloom = bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv, 0.0f).rgb;

    float4 hdr = hdrTarget[coord];
    hdrTarget[coord] = float4(hdr.rgb + bloom * pc.intensity, hdr.a);
}
