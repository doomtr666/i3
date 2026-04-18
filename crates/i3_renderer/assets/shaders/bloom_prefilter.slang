// Bloom Prefilter — bright-pass + soft-knee at half resolution.
//
// Reads HDR_Target mip 0 at half-res (bilinear = free pre-blur),
// applies a soft-knee threshold curve, writes to Bloom_Buffer mip 0.
//
// Soft-knee curve (Jimenez 2014):
//   rq = clamp(lum - threshold + knee, 0, 2*knee)
//   rq = rq² / (4*knee + ε)
//   weight = max(rq, lum - threshold) / max(lum, ε)
//
// Binding layout (set 0):
//   0 : hdrSrc     (sampled, full-res HDR_Target mip 0)
//   1 : bloomDst   (storage, Bloom_Buffer mip 0)

#include "common.slangh"

struct BloomPrefilterPc {
    uint2  srcSize;     // full-res HDR size
    float  threshold;
    float  knee;
};
[[vk::push_constant]] BloomPrefilterPc pc;

[[vk::binding(0, 0)]] Texture2D<float4>   hdrSrc;
[[vk::binding(1, 0)]] RWTexture2D<float4> bloomDst;

[numthreads(8, 8, 1)]
void prefilterMain(uint3 tid : SV_DispatchThreadID)
{
    uint2 dstSize = max(pc.srcSize / 2u, uint2(1u, 1u));
    if (any(tid.xy >= dstSize)) return;

    // Sample at half-resolution — bilinear gives free 2×2 box pre-blur.
    float2 uv = (float2(tid.xy) + 0.5f) / float2(dstSize);
    float4 col = hdrSrc.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv, 0.0f);

    float lum = dot(col.rgb, float3(0.2126f, 0.7152f, 0.0722f));

    // Soft-knee threshold.
    float rq     = clamp(lum - pc.threshold + pc.knee, 0.0f, 2.0f * pc.knee);
    rq           = (rq * rq) / (4.0f * pc.knee + 1e-6f);
    float weight = max(rq, lum - pc.threshold) / max(lum, 1e-6f);

    bloomDst[int2(tid.xy)] = float4(col.rgb * saturate(weight), 1.0f);
}
