
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

    // --- Sun Logic (SDF-ish) ---
    // WorldPos.x is left-right, WorldPos.z is depth.
    // The sun should be "far away" in Z, and centered in X.
    // But we are rendering a floor plane. The sun is usually in the sky.
    // However, for this demo, we can project the sun onto the far background or 
    // just render it as a circle on the floor if we want a reflection, 
    // OR we can't easily render a skybox here without a full quad.
    
    // Actually, to render a sun BEHIND the grid, we need a sky pass or to render it where alpha is low.
    // Let's hack it: The sun is at Z = Far, Y = Up.
    // Since we only have a floor plane, we can't render a sky sun unless we change geometry.
    // BUT! We can fake a "reflection" of the sun on the grid, or just add a sun if we assume the grid fades to transparency.
    
    // Let's add a sun glow based on view direction? 
    // We only have WorldPos of the floor.
    
    // Improved Retro Sun on the backing layer?
    // User asked for "SDF Sun". 
    // Let's assume we want a sun "setting" on the grid horizon.
    // We can compute the view vector.
    
    // Simple approach: Sun is a circle at (0, 0, Far).
    // dist to (0,0) in XZ?
    
    float3 sunColor = float3(1.0, 0.5, 0.0);
    float sunSize = 300.0;
    float sunDist = length(input.worldPos.xz - float2(0.0, 500.0)); // Sun at Z=500
    float sunGlow = exp(-sunDist * 0.01);
    
    // Combine Grid and Sun Glow
    float3 finalColor = color * colorVal;
    finalColor += sunColor * sunGlow * 2.0 * (1.0 - alpha); // Add sun where grid fades
    
    return float4(finalColor, alpha * colorVal + sunGlow * 0.5);
}
