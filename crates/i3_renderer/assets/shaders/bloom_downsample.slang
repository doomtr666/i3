// Bloom Downsample — 13-tap bilinear filter (Jimenez / Call of Duty 2014).
//
// The 13-tap pattern avoids the pulsing / aliasing artefacts that plague
// simple 4-tap bilinear downsampling for bright highlights.
//
// Tap layout (each ● is a bilinear sample, weight shown):
//
//   0.0625  0.125  0.0625
//   0.125   0.25   0.125
//   0.0625  0.125  0.0625
//
// In practice this is decomposed into 5 bilinear samples, each at a
// half-pixel offset, which the hardware blends for free:
//   centre (×0.25), 4 inner-corners (×0.125 each), 4 outer-corners (×0.0625 each)
//
// Binding layout (set 0):
//   0 : bloomSrc  (sampled, Bloom_Buffer mip N)
//   1 : bloomDst  (storage, Bloom_Buffer mip N+1)

#include "common.slangh"

struct BloomMipPc {
    uint2 srcSize;   // dimensions of mip N
    uint  srcMip;
    uint  _pad;
};
[[vk::push_constant]] BloomMipPc pc;

[[vk::binding(0, 0)]] Texture2D<float4>   bloomSrc;
[[vk::binding(1, 0)]] RWTexture2D<float4> bloomDst;

[numthreads(8, 8, 1)]
void downsampleMain(uint3 tid : SV_DispatchThreadID)
{
    uint2 dstSize = max(pc.srcSize / 2u, uint2(1u, 1u));
    if (any(tid.xy >= dstSize)) return;

    float2 texelSize = 1.0f / float2(pc.srcSize);
    float2 uv = (float2(tid.xy) + 0.5f) / float2(dstSize);

    // 13-tap filter — 5 bilinear samples that map to 13 logical taps.
    float3 result = float3(0.0f, 0.0f, 0.0f);

    // Centre tap × 0.25
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv, 0.0f).rgb * 0.25f;

    // Inner 4 taps (±½ texel diagonal) × 0.125 each
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2(-0.5f, -0.5f) * texelSize, 0.0f).rgb * 0.125f;
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2( 0.5f, -0.5f) * texelSize, 0.0f).rgb * 0.125f;
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2(-0.5f,  0.5f) * texelSize, 0.0f).rgb * 0.125f;
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2( 0.5f,  0.5f) * texelSize, 0.0f).rgb * 0.125f;

    // Outer 4 taps (±1 texel cardinal) × 0.0625 each — these together equal 0.25
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2(-1.0f,  0.0f) * texelSize, 0.0f).rgb * 0.0625f;
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2( 1.0f,  0.0f) * texelSize, 0.0f).rgb * 0.0625f;
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2( 0.0f, -1.0f) * texelSize, 0.0f).rgb * 0.0625f;
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2( 0.0f,  1.0f) * texelSize, 0.0f).rgb * 0.0625f;

    bloomDst[int2(tid.xy)] = float4(result, 1.0f);
}
