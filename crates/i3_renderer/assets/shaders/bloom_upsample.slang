// Bloom Upsample — tent filter 9-tap + accumulate onto lower mip.
//
// During the upsample chain each level's output is ADDED to the destination
// mip (which was written by a previous downsample). This accumulation is
// equivalent to a progressive widening of the blur kernel — the bloom spreads
// outward as lower-frequency components merge at coarser mips.
//
// The 9-tap tent kernel (weights sum to 1):
//   1  2  1
//   2  4  2   × (1/16)
//   1  2  1
//
// In Slang storage images, we perform a read-modify-write (Load + Store)
// because there is no atomic float4 add on storage images in standard Vulkan.
// This is safe because only one thread writes each destination pixel.
//
// Binding layout (set 0):
//   0 : bloomSrc  (sampled, Bloom_Buffer mip N)    — the coarser level being upsampled
//   1 : bloomDst  (storage, Bloom_Buffer mip N-1)  — accumulate into

#include "common.slangh"

struct BloomMipPc {
    uint2 srcSize;   // dimensions of the SOURCE (coarser) mip
    uint  srcMip;
    uint  _pad;
};
[[vk::push_constant]] BloomMipPc pc;

[[vk::binding(0, 0)]] Texture2D<float4>   bloomSrc;
[[vk::binding(1, 0)]] RWTexture2D<float4> bloomDst;

[numthreads(8, 8, 1)]
void upsampleMain(uint3 tid : SV_DispatchThreadID)
{
    // Destination is one mip finer (2× each dimension).
    uint2 dstSize = pc.srcSize * 2u;
    if (any(tid.xy >= dstSize)) return;

    float2 texelSize = 1.0f / float2(pc.srcSize);
    float2 uv = (float2(tid.xy) + 0.5f) / float2(dstSize);

    // 9-tap tent filter.
    float3 result = float3(0.0f, 0.0f, 0.0f);
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2(-1.0f, -1.0f) * texelSize, 0.0f).rgb * (1.0f / 16.0f);
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2( 0.0f, -1.0f) * texelSize, 0.0f).rgb * (2.0f / 16.0f);
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2( 1.0f, -1.0f) * texelSize, 0.0f).rgb * (1.0f / 16.0f);
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2(-1.0f,  0.0f) * texelSize, 0.0f).rgb * (2.0f / 16.0f);
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv,                                    0.0f).rgb * (4.0f / 16.0f);
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2( 1.0f,  0.0f) * texelSize, 0.0f).rgb * (2.0f / 16.0f);
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2(-1.0f,  1.0f) * texelSize, 0.0f).rgb * (1.0f / 16.0f);
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2( 0.0f,  1.0f) * texelSize, 0.0f).rgb * (2.0f / 16.0f);
    result += bloomSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv + float2( 1.0f,  1.0f) * texelSize, 0.0f).rgb * (1.0f / 16.0f);

    // Accumulate onto existing content at the destination mip.
    float3 existing = bloomDst[int2(tid.xy)].rgb;
    bloomDst[int2(tid.xy)] = float4(existing + result, 1.0f);
}
