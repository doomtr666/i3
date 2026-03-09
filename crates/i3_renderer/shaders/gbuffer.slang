// GBuffer pass — Bindless PBR
//
// Outputs to 4 MRT:
//   RT0 (R8G8B8A8_SRGB):       Albedo.rgb + Per-pixel Emissive Multiplier (alpha)
//   RT1 (R16G16_SFLOAT):       Normal.xy (octahedral encoding)
//   RT2 (R8G8_UNORM):          Roughness + Metallic
//   RT3 (R11G11B10_UFLOAT):    Emissive.rgb

struct MaterialData
{
    float4 baseColorFactor;
    float4 emissiveFactorAndAlphaCutoff;
    float metallicFactor;
    float roughnessFactor;
    int albedoTexIndex;
    int normalTexIndex;
    int rmaoTexIndex;
    int emissiveTexIndex;
    uint2 _pad;
};

// Bindings
layout(set = 1, binding = 0)
StructuredBuffer<MaterialData> materials;

layout(set = 2, binding = 0)
Texture2D bindlessTextures[4096];

layout(set = 2, binding = 1)
SamplerState globalSampler;

struct PushConstants
{
    float4x4 viewProjection;
    float4x4 model;
    uint materialId;
};

[[vk::push_constant]]
PushConstants pc;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(2)]] float2 uv       : TEXCOORD0;
};

struct VertexOutput
{
    float4 clipPosition : SV_Position;
    float3 worldNormal  : NORMAL;
    float2 uv           : TEXCOORD0;
};

struct GBufferOutput
{
    float4 albedo     : SV_Target0;
    float2 normal     : SV_Target1;
    float2 roughMetal : SV_Target2;
    float3 emissive   : SV_Target3;
};

// Octahedral normal encoding [-1,1]^3 -> [0,1]^2
float2 encodeOctahedral(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0)
    {
        float2 wrapped = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n = float3(wrapped, n.z);
    }
    return n.xy * 0.5 + 0.5;
}

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    float4 worldPos = mul(pc.model, float4(input.position, 1.0));
    output.clipPosition = mul(pc.viewProjection, worldPos);

    output.worldNormal = normalize(mul((float3x3)pc.model, input.normal));
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
GBufferOutput fragmentMain(VertexOutput input)
{
    MaterialData mat = materials[pc.materialId];
    
    float4 albedo = float4(mat.baseColorFactor);
    if (mat.albedoTexIndex >= 0)
    {
        albedo *= bindlessTextures[mat.albedoTexIndex].Sample(globalSampler, input.uv);
    }

    // Alpha test
    if (albedo.a < mat.emissiveFactorAndAlphaCutoff.a)
    {
        discard;
    }

    float roughness = mat.roughnessFactor;
    float metallic = mat.metallicFactor;
    if (mat.rmaoTexIndex >= 0)
    {
        float4 rmao = bindlessTextures[mat.rmaoTexIndex].Sample(globalSampler, input.uv);
        // Standard glTF: R=ao (ignored here for gbuffer), G=roughness, B=metallic
        roughness *= rmao.g;
        metallic *= rmao.b;
    }

    float3 emissive = mat.emissiveFactorAndAlphaCutoff.rgb;
    if (mat.emissiveTexIndex >= 0)
    {
        emissive *= bindlessTextures[mat.emissiveTexIndex].Sample(globalSampler, input.uv).rgb;
    }

    if (mat.normalTexIndex >= 0)
    {
        // Normal mapping placeholder
    }

    GBufferOutput output;
    output.albedo     = float4(albedo.rgb, 1.0);
    output.normal     = encodeOctahedral(normalize(input.worldNormal));
    output.roughMetal = float2(roughness, metallic);
    output.emissive   = emissive;
    return output;
}
