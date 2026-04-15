// crates/i3_renderer/assets/shaders/hiz_build.slang

struct PushConstants {
    uint2 src_size;    // original depth buffer size
    uint2 hiz_size;    // Mip 0 size (Pot)
    uint  src_mip;     // src mip for reduction
    uint  _pad;
};

[[vk::push_constant]] PushConstants push;

// Binding 0: Source Image (renamed to avoid cache)
[[vk::binding(0, 0)]] Texture2D<float> img_source;

// Binding 1: Output Mip
[[vk::binding(1, 0)]] RWTexture2D<float> output_tex;

[shader("compute")]
[numthreads(8, 8, 1)]
void blit_depth(uint3 tid : SV_DispatchThreadID) {
    if (any(tid.xy >= push.hiz_size)) return;

    // Nearest-neighbor blit from depth buffer to HiZ mip 0.
    float2 uv = (float2(tid.xy) + 0.5) / float2(push.hiz_size);
    uint2 src_coord = uint2(uv * float2(push.src_size));
    output_tex[tid.xy] = img_source.Load(uint3(src_coord, 0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void reduce_mip(uint3 tid : SV_DispatchThreadID) {
    uint2 dst_coord = tid.xy;
    uint2 src_size = max(push.hiz_size >> push.src_mip, uint2(1, 1));
    // POT image: exact power-of-two division at every level — no odd dimensions.
    uint2 dst_size = max(src_size >> 1, uint2(1, 1));
    
    if (any(dst_coord >= dst_size)) return;

    uint2 s = dst_coord * 2;
    float a = img_source.Load(uint3(min(s + uint2(0,0), src_size - 1), 0));
    float b = img_source.Load(uint3(min(s + uint2(1,0), src_size - 1), 0));
    float c = img_source.Load(uint3(min(s + uint2(0,1), src_size - 1), 0));
    float d = img_source.Load(uint3(min(s + uint2(1,1), src_size - 1), 0));

    // Conserve the largest depth in Reverse-Z (nearest object)
    output_tex[dst_coord] = max(max(a, b), max(c, d));
}
