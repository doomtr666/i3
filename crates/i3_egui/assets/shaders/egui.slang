struct PushConstants {
    float4 screen_to_clip; // x=2/w, y=-2/h, z=-1, w=1
};

struct VSInput {
    float2 pos : POSITION;
    float2 uv : TEXCOORD0;
    uint color : COLOR0;
};

struct VSOutput {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
};

[[vk::push_constant]]
PushConstants u_PushConstants;

// Binding 0, Set 0: Font atlas
[[vk::binding(0, 0)]]
Texture2D u_FontAtlas;
[[vk::binding(0, 0)]] 
SamplerState u_FontSampler;

float4 unpackColor(uint color) {
    float4 c;
    c.r = (float)(color & 0xFF) / 255.0;
    c.g = (float)((color >> 8) & 0xFF) / 255.0;
    c.b = (float)((color >> 16) & 0xFF) / 255.0;
    c.a = (float)((color >> 24) & 0xFF) / 255.0;
    
    // Linearize ONLY RGB. Vertex colors are sRGB Premultiplied.
    // Keeping alpha as is for correct blending in linear space.
    return float4(pow(c.rgb, 2.2), c.a);
}

[shader("vertex")]
VSOutput vertexMain(VSInput input) {
    VSOutput output;
    // Map from screen space [0, w]x[0, h] to clip space [-1, 1]x[1, -1]
    // Engine Convention: Top is 1.0, Bottom is -1.0 in clip space before negative viewport flip.
    output.pos = float4(
        input.pos.x * u_PushConstants.screen_to_clip.x + u_PushConstants.screen_to_clip.z,
        input.pos.y * u_PushConstants.screen_to_clip.y + u_PushConstants.screen_to_clip.w,
        0.0,
        1.0
    );
    output.uv = input.uv;
    output.color = unpackColor(input.color);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target {
    return input.color * u_FontAtlas.Sample(u_FontSampler, input.uv);
}
