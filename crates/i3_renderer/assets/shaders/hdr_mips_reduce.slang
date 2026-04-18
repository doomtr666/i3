// HDR mip chain reduction — average (2×2 box filter) for R16G16B16A16_SFLOAT.
// Used by HdrMipsPass to generate SceneLit mips for SSR roughness-based sampling.

struct PushConstants {
    uint2 src_size;  // dimensions of the source mip level
    uint  src_mip;   // source mip index to read from
    uint  _pad;
};

[[vk::push_constant]] PushConstants push;

[[vk::binding(0, 0)]] Texture2D<float4>   img_source;
[[vk::binding(1, 0)]] RWTexture2D<float4> output_tex;

[shader("compute")]
[numthreads(8, 8, 1)]
void reduce_mip(uint3 tid : SV_DispatchThreadID)
{
    uint2 dst_size = max(push.src_size >> 1, uint2(1, 1));
    if (any(tid.xy >= dst_size)) return;

    uint2 s = tid.xy * 2;
    uint2 src_max = push.src_size - 1;

    float4 a = img_source.Load(int3(min(s + uint2(0, 0), src_max), 0));
    float4 b = img_source.Load(int3(min(s + uint2(1, 0), src_max), 0));
    float4 c = img_source.Load(int3(min(s + uint2(0, 1), src_max), 0));
    float4 d = img_source.Load(int3(min(s + uint2(1, 1), src_max), 0));

    output_tex[tid.xy] = (a + b + c + d) * 0.25;
}
