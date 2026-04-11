// crates/i3_renderer/assets/shaders/hiz_blit.slang
//#include "common.slangh"

// Input: Full resolution Depth Buffer (D32_FLOAT)
[[vk::binding(0, 0)]]
Texture2D<float> depthBuffer;

// Output: Hi-Z Pyramid Mip 0 (R32_SFLOAT)
[[vk::binding(1, 0)]]
RWTexture2D<float> hizMip0;

struct PushConstants {
    uint2 screen_size;
} push;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= push.screen_size.x || id.y >= push.screen_size.y)
        return;

    // Direct copy from DepthBuffer to Hi-Z Mip 0.
    // D32_FLOAT to R32_SFLOAT conversion is implicit.
    float depth = depthBuffer.Load(int3(id.xy, 0));
    hizMip0[id.xy] = depth;
}
