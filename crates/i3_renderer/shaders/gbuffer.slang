// GBuffer pass — Bindless PBR
//
// Outputs to 4 MRT:
//   RT0 (R8G8B8A8_SRGB):       Albedo.rgb + Per-pixel Emissive Multiplier (alpha)
//   RT1 (R16G16_SFLOAT):       Normal.xy (octahedral encoding)
//   RT2 (R8G8_UNORM):          Roughness + Metallic
//   RT3 (R11G11B10_UFLOAT):    Emissive.rgb

[layout("std430")]
struct MaterialData
{
    float4 baseColorFactor;
    float4 emissiveFactorAndAlphaCutoff;
    // x: metallic, y: roughness, zw: padding
    float4 pbrFactors;
    // x: albedo, y: normal, z: rmao, w: emissive
    int4 textureIndices;
};

// Bindings
[layout("std430")]
[vk::binding(0, 1)]
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
    uint3 _pad;
};

[[vk::push_constant]]
PushConstants pc;

struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(2)]] float2 uv       : TEXCOORD0;
    [[vk::location(3)]] float4 tangent  : TANGENT;
};

struct VertexOutput
{
    float4 clipPosition : SV_Position;
    float3 worldNormal  : NORMAL;
    float3 worldTangent : TANGENT;
    float3 worldBitangent : BITANGENT;
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

    float3x3 normalMatrix = (float3x3)pc.model;
    float3 worldNormal = normalize(mul(normalMatrix, input.normal));
    float3 worldTangent = normalize(mul(normalMatrix, input.tangent.xyz));
    
    // Gram-Schmidt re-orthogonalization to ensure T is perpendicular to N
    worldTangent = normalize(worldTangent - dot(worldTangent, worldNormal) * worldNormal);
    
    // Calculate bitangent using handness stored in tangent.w
    float3 worldBitangent = cross(worldNormal, worldTangent) * input.tangent.w;

    output.worldNormal = worldNormal;
    output.worldTangent = worldTangent;
    output.worldBitangent = worldBitangent;
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
GBufferOutput fragmentMain(VertexOutput input)
{
    MaterialData mat = materials[pc.materialId];
    
    float4 albedo = float4(mat.baseColorFactor);
    if (mat.textureIndices.x >= 0)
    {
        albedo *= bindlessTextures[mat.textureIndices.x].Sample(globalSampler, input.uv);
    }

    // Alpha test
    if (albedo.a < mat.emissiveFactorAndAlphaCutoff.a)
    {
        discard;
    }

    float roughness = mat.pbrFactors.y;
    float metallic = mat.pbrFactors.x;
    if (mat.textureIndices.z >= 0)
    {
        float4 rmao = bindlessTextures[mat.textureIndices.z].Sample(globalSampler, input.uv);
        // Standard glTF: R=ao (ignored here for gbuffer), G=roughness, B=metallic
        roughness *= rmao.g;
        metallic *= rmao.b;
    }

    float3 emissive = mat.emissiveFactorAndAlphaCutoff.rgb;
    if (mat.textureIndices.w >= 0)
    {
        emissive *= bindlessTextures[mat.textureIndices.w].Sample(globalSampler, input.uv).rgb;
    }

    float3 normal = normalize(input.worldNormal);
    if (mat.textureIndices.y >= 0)
    {
        // Sample normal map (typically BC5: RG channels only)
        float2 normalSample = bindlessTextures[mat.textureIndices.y].Sample(globalSampler, input.uv).rg;
        
        // Transform from [0, 1] to [-1, 1]
        float2 tangentNormal = normalSample * 2.0 - 1.0;
        
        // Reconstruct Z channel
        float z = sqrt(max(0.0, 1.0 - dot(tangentNormal, tangentNormal)));
        float3 normalTS = float3(tangentNormal, z);

        // Transform from tangent space to world space
        float3x3 TBN = float3x3(
            normalize(input.worldTangent),
            normalize(input.worldBitangent),
            normal
        );
        normal = normalize(mul(normalTS, TBN));
    }

    GBufferOutput output;
    output.albedo     = float4(albedo.rgb, 1.0);
    output.normal     = encodeOctahedral(normal);
    output.roughMetal = float2(roughness, metallic);
    output.emissive   = emissive;
    return output;
}
