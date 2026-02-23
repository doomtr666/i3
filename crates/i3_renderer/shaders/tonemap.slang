struct PushConstants
{
    uint debug_mode;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]] Texture2D hdrTarget;
[[vk::binding(0, 0)]] SamplerState pointSampler;

struct ExposureData {
    float prev_exposure;
    float new_exposure;
};
[[vk::binding(1, 0)]] StructuredBuffer<ExposureData> exposure_buffer;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.uv = uv;
    output.position = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return output;
}

// ACES Filmic Tonemapping (Approximation by Stephen Hill)
float3 ACESFilm(float3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    // Read HDR Pre-Exposed color
    float3 color = hdrTarget.Sample(pointSampler, input.uv).rgb;
    
    // Remove Pre-Exposure Scale
    float prev_ex = exposure_buffer[0].prev_exposure;
    float new_ex  = exposure_buffer[0].new_exposure;
    
    if (prev_ex == 0.0) { prev_ex = 1.0; }
    if (new_ex == 0.0) { new_ex = 1.0; }
    
    // Exposure Adjustment
    color *= (new_ex / prev_ex);

    // ACES Tonemapping
    color = ACESFilm(color);
    
    return float4(color, 1.0);
}
