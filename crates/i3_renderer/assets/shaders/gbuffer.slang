// assets/system/shaders/gbuffer.slang
#include "common.slangh"

// Bindings
[[vk::binding(0, 0)]]
StructuredBuffer<GpuMeshDescriptor> meshDescriptors;

[[vk::binding(1, 0)]]
StructuredBuffer<GpuInstanceData> instances;

[[vk::binding(2, 0)]]
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
    float3 worldNormal : NORMAL;
    float3 worldTangent : TANGENT;
    float3 worldBitangent : BITANGENT;
    float2 uv : TEXCOORD0;
    [[vk::flat]]
    uint materialId : MAT_ID;
};

struct GBufferOutput
{
    float4 albedo : SV_Target0;
    float2 normal : SV_Target1;
    float2 roughMetal : SV_Target2;
    float3 emissive : SV_Target3;
};

[shader("vertex")]
VertexOutput vertexMain(uint indexId: SV_VertexID, 
                         uint instanceId : SV_InstanceID,
                         [[vk::builtin("BaseInstance")]] uint baseInstance : SV_StartInstanceLocation)
{
    uint absoluteInstanceId = baseInstance + instanceId;
    GpuInstanceData instance = instances[absoluteInstanceId];
    GpuMeshDescriptor mesh = meshDescriptors[instance.meshIdx];

    // Vertex Pulling via BDA — index format aware
    // mesh.indexStride: 2 = U16, 4 = U32 (matches GpuMeshDescriptor::index_stride in scene.rs)
    uint vertexIdx;
    if (mesh.indexCount > 0u && mesh.indexBufferAddress != 0)
    {
        uint indexIdWithOffset = indexId + mesh.firstIndex;
        if (mesh.indexStride == 2u)
        {
            // U16 indices: read safely as uint16_t via BDA.
            vertexIdx = (uint)((uint16_t *)mesh.indexBufferAddress)[indexIdWithOffset];
        }
        else
        {
            // U32 indices: read directly as uint.
            vertexIdx = ((uint *)mesh.indexBufferAddress)[indexIdWithOffset];
        }
    }
    else
    {
        vertexIdx = indexId;
    }

    // Fetch vertex attributes using explicit byte offsets + mesh.vertexStride.
    // This bypasses ambiguity in Slang's BDA struct stride:
    //   std430 layout → float3 aligned to 16 → Vertex = 64B (WRONG, actual data = 48B)
    //   scalar layout → float3 aligned to 4  → Vertex = 48B (correct)
    // Using byte arithmetic with the actual stride from the mesh descriptor is always correct.
    // POSITION_NORMAL_UV_TANGENT packed layout (48B):
    //   [+0]  float3 position (12B)
    //   [+12] float3 normal   (12B)
    //   [+24] float2 uv       (8B)
    //   [+32] float4 tangent  (16B)
    // Using mesh.vertexOffset to support base vertex offset in packed buffers.
    uint64_t vBase = mesh.vertexBufferAddress + uint64_t(vertexIdx + mesh.vertexOffset) * uint64_t(mesh.vertexStride);
    float3 vPos = *(float3 *)(vBase);
    float3 vNormal = *(float3 *)(vBase + uint64_t(12));
    float2 vUV = *(float2 *)(vBase + uint64_t(24));
    float4 vTangent = *(float4 *)(vBase + uint64_t(32));

    VertexOutput output;
    float4 worldPos = mul(instance.worldTransform, float4(vPos, 1.0));
    output.clipPosition = mul(pc.viewProjection, worldPos);

    float3x3 normalMatrix = (float3x3)instance.worldTransform;
    float3 worldNormal = normalize(mul(normalMatrix, vNormal));
    float3 worldTangent = normalize(mul(normalMatrix, vTangent.xyz));

    worldTangent = normalize(worldTangent - dot(worldTangent, worldNormal) * worldNormal);
    float3 worldBitangent = cross(worldNormal, worldTangent) * vTangent.w;

    output.worldNormal = worldNormal;
    output.worldTangent = worldTangent;
    output.worldBitangent = worldBitangent;
    output.uv = vUV;
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
        albedo *= bindlessTextures[mat.textureIndices.x].Sample(bindlessSamplers[SAMPLER_ANISO], input.uv);
    }

    if (albedo.a < mat.emissiveFactorAndAlphaCutoff.a)
    {
        discard;
    }

    float roughness = mat.pbrFactors.y;
    float metallic = mat.pbrFactors.x;
    if (mat.textureIndices.z >= 0)
    {
        float4 rmao = bindlessTextures[mat.textureIndices.z].Sample(bindlessSamplers[SAMPLER_ANISO], input.uv);
        roughness *= rmao.g;
        metallic *= rmao.b;
    }

    float3 emissive = mat.emissiveFactorAndAlphaCutoff.rgb;
    if (mat.textureIndices.w >= 0)
    {
        emissive *= bindlessTextures[mat.textureIndices.w].Sample(bindlessSamplers[SAMPLER_ANISO], input.uv).rgb;
    }

    float3 normal = normalize(input.worldNormal);
    if (mat.textureIndices.y >= 0)
    {
        float2 normalSample = bindlessTextures[mat.textureIndices.y].Sample(bindlessSamplers[SAMPLER_ANISO], input.uv).rg;
        float2 tangentNormal = normalSample * 2.0 - 1.0;
        float z = sqrt(max(0.0, 1.0 - dot(tangentNormal, tangentNormal)));
        float3 normalTS = float3(tangentNormal, z);

        float3x3 TBN = float3x3(
            normalize(input.worldTangent),
            normalize(input.worldBitangent),
            normal);
        normal = normalize(mul(normalTS, TBN));
    }

    GBufferOutput output;
    output.albedo = float4(albedo.rgb, 1.0);
    output.normal = encodeOctahedral(normal);
    output.roughMetal = float2(roughness, metallic);
    output.emissive = emissive;
    return output;
}
