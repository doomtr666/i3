
struct Uniforms
{
    float4x4 viewProj;
    float time;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Uniforms> ubo;

struct VertexInput
{
    float3 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float3 worldPos : POSITION;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    output.worldPos = input.position;
    output.position = mul(ubo.viewProj, float4(input.position, 1.0));
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    // Simple Grid Logic
    float2 grid = abs(frac(input.worldPos.xz) - 0.5) / fwidth(input.worldPos.xz);
    float lineVal = min(grid.x, grid.y);
    float colorVal = 1.0 - min(lineVal, 1.0);
    
    // Fade out
    float dist = length(input.worldPos.xz);
    float alpha = max(0.0, 1.0 - dist * 0.05);

    // Pulse with time
    float pulse = (sin(ubo.time * 2.0) + 1.0) * 0.5;
    float3 color = float3(1.0, 0.0, 1.0) * pulse + float3(0.0, 1.0, 1.0) * (1.0 - pulse);

    return float4(color * colorVal, alpha * colorVal);
}
