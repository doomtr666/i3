
struct Uniforms
{
    float time;
    float2 resolution;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> ubo;

[[vk::binding(1, 0)]]
Texture2D sceneColor;
[[vk::binding(1, 0)]]
SamplerState sceneSampler;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    // Generate full-screen triangle
    // uv=(0,0) -> pos=(-1, 1) [Top-Left]
    // uv=(2,0) -> pos=(3, 1)  [Top-Way-Right]
    // uv=(0,2) -> pos=(-1, -3) [Way-Below-Left]
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.uv = uv;
    output.position = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    float2 uv = input.uv;
    
    // Curvature
    float2 dc = abs(0.5 - uv);
    float d = dc.x * dc.x + dc.y * dc.y;
    uv = (uv - 0.5) * (1.0 + d * 0.1) + 0.5;
    
    // --- SDF Sun ---
    float2 sunPos = float2(0.5, 0.35);
    float2 p = uv - sunPos;
    float dist = length(p);
    float radius = 0.25;
    
    float3 sunColorTop = float3(1.0, 0.9, 0.0);    // Yellow
    float3 sunColorBottom = float3(1.0, 0.0, 0.5); // Pink/Purple
    float3 sunColor = lerp(sunColorTop, sunColorBottom, saturate((uv.y - 0.1) * 2.0));
    
    // Base circle
    float sunMask = smoothstep(radius, radius - 0.005, dist);
    
    // Horizontal retro cuts (getting wider at bottom)
    float cutFrequency = 15.0;
    float cutHeight = 0.5; // percent of bar that is cut
    float cut = frac(uv.y * cutFrequency);
    
    // Make cuts wider at the bottom of the sun
    float yFactor = saturate((uv.y - sunPos.y + radius) / (radius * 2.0));
    float threshold = lerp(0.0, 0.8, yFactor);
    if (cut < threshold && uv.y > sunPos.y) {
        sunMask = 0.0;
    }
    
    float3 bgSun = sunColor * sunMask;
    
    // Cutoff
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        return float4(0.0, 0.0, 0.0, 1.0);
        
    // Chromatic Aberration
    float shift = 0.003;
    float r = sceneColor.Sample(sceneSampler, float2(uv.x - shift, uv.y)).r;
    float g = sceneColor.Sample(sceneSampler, float2(uv.x, uv.y)).g;
    float b = sceneColor.Sample(sceneSampler, float2(uv.x + shift, uv.y)).b;
    float3 scene = float3(r, g, b);
    
    // Composite: Sun is behind scene (assuming black background in scene)
    float3 color = scene + bgSun * (1.0 - saturate(length(scene) * 2.0));
    
    // Scanlines
    float scanline = sin(uv.y * ubo.resolution.y * 2.0 * 3.14159) * 0.05 + 0.95;
    color *= scanline;
    
    // Vignette
    float vig = 16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y);
    vig = pow(vig, 0.2);
    color *= vig;
    
    return float4(color, 1.0);
}
