// crates/i3_renderer/assets/shaders/hiz_reduce.slang
//#include "common.slangh"

// Input: Previous Mip level (R32_SFLOAT)
[[vk::binding(0, 0)]]
Texture2D<float> inputMip;

// Output: Current Mip level (R32_SFLOAT)
[[vk::binding(1, 0)]]
RWTexture2D<float> outputMip;

struct PushConstants {
    uint2 output_size;
} push;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= push.output_size.x || id.y >= push.output_size.y)
        return;

    // Sample 2x2 area from previous mip.
    // Load returns 0.0 for out of bounds, which is "Far" in Reverse-Z.
    // Thus max() remains conservative.
    int2 base = int2(id.xy * 2);
    float d00 = inputMip.Load(int3(base + int2(0, 0), 0));
    float d10 = inputMip.Load(int3(base + int2(1, 0), 0));
    float d01 = inputMip.Load(int3(base + int2(0, 1), 0));
    float d11 = inputMip.Load(int3(base + int2(1, 1), 0));

    // MAX reduction: store the closest depth (largest value in Reverse-Z).
    float maxDepth = max(max(d00, d10), max(d01, d11));

    outputMip[id.xy] = maxDepth;
}
