// Debug visualization — fullscreen pass for GBuffer inspection
//
// Reads all 4 GBuffer targets and displays the selected channel.
// Channel selection via push constant.

struct PushConstants
{
    uint channel;
    uint3 _pad;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]]
Texture2D gbufferAlbedo;
[[vk::binding(0, 0)]]
SamplerState albedoSampler;

[[vk::binding(1, 0)]]
Texture2D gbufferNormal;
[[vk::binding(1, 0)]]
SamplerState normalSampler;

[[vk::binding(2, 0)]]
Texture2D gbufferRoughMetal;
[[vk::binding(2, 0)]]
SamplerState roughMetalSampler;

[[vk::binding(3, 0)]]
Texture2D gbufferEmissive;
[[vk::binding(3, 0)]]
SamplerState emissiveSampler;

[[vk::binding(4, 0)]]
Texture2D gbufferDepth;
[[vk::binding(4, 0)]]
SamplerState depthSampler;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Decode octahedral normal back to world space
float3 decodeOctahedral(float2 e)
{
    e = e * 2.0 - 1.0;
    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0)
    {
        float2 wrapped = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n = float3(wrapped, n.z);
    }
    return normalize(n);
}

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.uv = uv;
    output.position = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    float2 uv = input.uv;
    float3 color = float3(0.0, 0.0, 0.0);

    // Channel enum: 0=Albedo, 1=Normal, 2=Roughness, 3=Metallic, 4=Emissive, 5=Depth
    switch (pc.channel)
    {
    case 0: // Albedo
        color = gbufferAlbedo.Sample(albedoSampler, uv).rgb;
        break;
    case 1: // World Normal → remap to [0,1] for visualization
        {
            float2 nEnc = gbufferNormal.Sample(normalSampler, uv).rg;
            float3 n = decodeOctahedral(nEnc);
            color = n * 0.5 + 0.5;
        }
        break;
    case 2: // Roughness (R channel of RoughMetal)
        {
            float r = gbufferRoughMetal.Sample(roughMetalSampler, uv).r;
            color = float3(r, r, r);
        }
        break;
    case 3: // Metallic (G channel of RoughMetal)
        {
            float m = gbufferRoughMetal.Sample(roughMetalSampler, uv).g;
            color = float3(m, m, m);
        }
        break;
    case 4: // Emissive
        color = gbufferEmissive.Sample(emissiveSampler, uv).rgb;
        break;
    case 5: // Depth
        {
            float d = gbufferDepth.Sample(depthSampler, uv).r;
            // Basic linearization for visualization (assuming [0,1] range, 
            // focus on the first few units since SPONZA is small-ish)
            float linearD = (2.0 * 0.1) / (100.0 + 0.1 - d * (100.0 - 0.1));
            color = float3(d, d, d); // Just show raw depth for now, or 1-d
            // Most people expect far = black, near = white
            float v = 1.0 - d;
            color = float3(v, v, v);
        }
        break;
    default:
        color = float3(1.0, 0.0, 1.0); // Magenta = unsupported channel
        break;
    }

    return float4(color, 1.0);
}
