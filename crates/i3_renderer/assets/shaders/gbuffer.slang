// assets/system/shaders/gbuffer.slang
#include "common.slangh"

// Bindings
[[vk::binding(0, 1)]]
StructuredBuffer<GpuMeshDescriptor> meshDescriptors;

[[vk::binding(1, 1)]]
StructuredBuffer<GpuInstanceData> instances;

[[vk::binding(2, 1)]]
StructuredBuffer<MaterialData> materials;

struct PushConstants
{
    float4x4 viewProjection;
};

[[vk::push_constant]]
PushConstants pc;

struct Vertex
{
    float3 position;
    float3 normal;
    float2 uv;
    float4 tangent;
};

struct VertexOutput
{
    float4 clipPosition : SV_Position;
    float3 worldNormal  : NORMAL;
    float3 worldTangent : TANGENT;
    float3 worldBitangent : BITANGENT;
    float2 uv           : TEXCOORD0;
    [[vk::flat]] uint materialId : MAT_ID;
};

struct GBufferOutput
{
    float4 albedo     : SV_Target0;
    float2 normal     : SV_Target1;
    float2 roughMetal : SV_Target2;
    float3 emissive   : SV_Target3;
};

[shader("vertex")]
VertexOutput vertexMain(uint indexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    GpuInstanceData instance = instances[instanceId];
    GpuMeshDescriptor mesh = meshDescriptors[instance.meshIdx];
    
    // Vertex Pulling via BDA
    uint vertexIdx;
    if (mesh.indexCount > 0) {
        vertexIdx = ((uint*)mesh.indexBufferAddress)[indexId];
    } else {
        vertexIdx = indexId;
    }
    
    Vertex v = ((Vertex*)mesh.vertexBufferAddress)[vertexIdx];

    VertexOutput output;
    float4 worldPos = mul(instance.worldTransform, float4(v.position, 1.0));
    output.clipPosition = mul(pc.viewProjection, worldPos);

    float3x3 normalMatrix = (float3x3)instance.worldTransform;
    float3 worldNormal = normalize(mul(normalMatrix, v.normal));
    float3 worldTangent = normalize(mul(normalMatrix, v.tangent.xyz));
    
    worldTangent = normalize(worldTangent - dot(worldTangent, worldNormal) * worldNormal);
    float3 worldBitangent = cross(worldNormal, worldTangent) * v.tangent.w;

    output.worldNormal = worldNormal;
    output.worldTangent = worldTangent;
    output.worldBitangent = worldBitangent;
    output.uv = v.uv;
    output.materialId = instance.materialId;
    return output;
}

[shader("fragment")]
GBufferOutput fragmentMain(VertexOutput input)
{
    MaterialData mat = materials[input.materialId];
    
    float4 albedo = float4(mat.baseColorFactor);
    if (mat.textureIndices.x >= 0)
    {
        albedo *= bindlessTextures[mat.textureIndices.x].Sample(globalSampler, input.uv);
    }

    if (albedo.a < mat.emissiveFactorAndAlphaCutoff.a)
    {
        discard;
    }

    float roughness = mat.pbrFactors.y;
    float metallic = mat.pbrFactors.x;
    if (mat.textureIndices.z >= 0)
    {
        float4 rmao = bindlessTextures[mat.textureIndices.z].Sample(globalSampler, input.uv);
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
        float2 normalSample = bindlessTextures[mat.textureIndices.y].Sample(globalSampler, input.uv).rg;
        float2 tangentNormal = normalSample * 2.0 - 1.0;
        float z = sqrt(max(0.0, 1.0 - dot(tangentNormal, tangentNormal)));
        float3 normalTS = float3(tangentNormal, z);

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
