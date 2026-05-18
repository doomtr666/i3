// examples/voxel/assets/shaders/voxel_gbuffer.slang
#include "common.slangh"

// Bindings
[[vk::binding(0, 0)]]
StructuredBuffer<GpuMeshDescriptor> meshDescriptors;

[[vk::binding(1, 0)]]
StructuredBuffer<GpuInstanceData> instances;

[[vk::binding(2, 0)]]
StructuredBuffer<MaterialData> materials;

struct VertexOutput
{
    float4 clipPosition : SV_Position;
    float3 worldNormal : NORMAL;
    float3 worldTangent : TANGENT;
    float3 worldBitangent : BITANGENT;
    float2 uv : TEXCOORD0;
    float3 worldPos : WORLDPOS;
    [[vk::flat]]
    uint materialId : MAT_ID;
    [[vk::flat]]
    uint voxelFlags : VOXEL_FLAGS;
    [[vk::flat]]
    uint voxelPad : VOXEL_PAD;
};

struct GBufferOutput
{
    float4 albedo : SV_Target0;
    float2 normal : SV_Target1;
    float2 roughMetal : SV_Target2;
    float3 emissive : SV_Target3;
    float hiz_depth : SV_Target4;
};

[shader("vertex")]
VertexOutput vertexMain(uint indexId: SV_VertexID,
                        uint instanceId: SV_InstanceID,
                        [[vk::builtin("BaseInstance")]] uint baseInstance: SV_StartInstanceLocation)
{
    uint absoluteInstanceId = baseInstance + instanceId;
    GpuInstanceData instance = instances[absoluteInstanceId];
    GpuMeshDescriptor mesh = meshDescriptors[instance.meshIdx];

    uint vertexIdx;
    if (mesh.indexCount > 0u && mesh.indexBufferAddress != 0)
    {
        uint indexIdWithOffset = indexId + mesh.firstIndex;
        if (mesh.indexStride == 2u)
        {
            vertexIdx = (uint)((uint16_t *)mesh.indexBufferAddress)[indexIdWithOffset];
        }
        else
        {
            vertexIdx = ((uint *)mesh.indexBufferAddress)[indexIdWithOffset];
        }
    }
    else
    {
        vertexIdx = indexId;
    }

    uint64_t vBase = mesh.vertexBufferAddress + uint64_t(vertexIdx + mesh.vertexOffset) * uint64_t(mesh.vertexStride);
    float3 vPos = *(float3 *)(vBase);
    float3 vNormal = *(float3 *)(vBase + uint64_t(12));
    float2 vUV = *(float2 *)(vBase + uint64_t(24));
    float4 vTangent = *(float4 *)(vBase + uint64_t(32));

    VertexOutput output;
    float4 worldPos = mul(instance.worldTransform, float4(vPos, 1.0));
    output.clipPosition = mul(g_common.viewProjection, worldPos);

    float3x3 normalMatrix = (float3x3)instance.worldTransform;
    float3 worldNormal = normalize(mul(normalMatrix, vNormal));
    float3 worldTangent = normalize(mul(normalMatrix, vTangent.xyz));

    worldTangent = normalize(worldTangent - dot(worldTangent, worldNormal) * worldNormal);
    float3 worldBitangent = cross(worldNormal, worldTangent) * vTangent.w;

    output.worldNormal = worldNormal;
    output.worldTangent = worldTangent;
    output.worldBitangent = worldBitangent;
    output.uv = vUV;
    output.worldPos = worldPos.xyz;
    output.materialId = instance.materialId;
    output.voxelFlags = instance.flags;
    output.voxelPad = instance._pad;
    return output;
}

// ------------------------------------------------------------------------------------------------
// Hex-Tiling Implementation
// ------------------------------------------------------------------------------------------------

float2 hash22(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.xx + p3.yz) * p3.zy);
}

float4 SampleHexTiled(int texIndex, float2 uv)
{
    if (texIndex < 0) return float4(1.0);
    
    float2 grid = float2(1.0, sqrt(3.0));
    float2 uv2 = uv / grid;
    
    float2 center1 = floor(uv2) + 0.5;
    float2 center2 = floor(uv2 + 0.5);
    
    float2 offset1 = uv - center1 * grid;
    float2 offset2 = uv - center2 * grid;
    
    float2 center = dot(offset1, offset1) < dot(offset2, offset2) ? center1 : center2;
    float2 offset = uv - center * grid;
    
    float2 h = hash22(center) * 2.0 - 1.0;
    float angle = h.x * PI;
    float2 rotUv = float2(
        offset.x * cos(angle) - offset.y * sin(angle),
        offset.x * sin(angle) + offset.y * cos(angle)
    );
    
    float2 finalUv = rotUv + h * 0.5;
    float2 dx = ddx(uv);
    float2 dy = ddy(uv);
    
    return bindlessTextures[texIndex].SampleGrad(bindlessSamplers[SAMPLER_ANISO], finalUv, dx, dy);
}

// ------------------------------------------------------------------------------------------------
// Triplanar Splatting
// ------------------------------------------------------------------------------------------------

void EvaluateVoxelMaterial(
    MaterialData mat, float3 p, float3 w, float3 worldNormal,
    out float4 albedo, out float3 normal, out float4 rmao)
{
    float texScale = 0.05; 
    float2 uvX = p.zy * texScale;
    float2 uvY = p.xz * texScale;
    float2 uvZ = p.xy * texScale;

    albedo = float4(1.0);
    if (mat.textureIndices.x >= 0) {
        float4 cx = SampleHexTiled(mat.textureIndices.x, uvX);
        float4 cy = SampleHexTiled(mat.textureIndices.x, uvY);
        float4 cz = SampleHexTiled(mat.textureIndices.x, uvZ);
        albedo = cx * w.x + cy * w.y + cz * w.z;
    }

    rmao = float4(0.0, 1.0, 0.0, 1.0); 
    if (mat.textureIndices.z >= 0) {
        float4 cx = SampleHexTiled(mat.textureIndices.z, uvX);
        float4 cy = SampleHexTiled(mat.textureIndices.z, uvY);
        float4 cz = SampleHexTiled(mat.textureIndices.z, uvZ);
        rmao = cx * w.x + cy * w.y + cz * w.z;
    }

    normal = worldNormal;
    if (mat.textureIndices.y >= 0) {
        float2 nx = SampleHexTiled(mat.textureIndices.y, uvX).rg * 2.0 - 1.0;
        float2 ny = SampleHexTiled(mat.textureIndices.y, uvY).rg * 2.0 - 1.0;
        float2 nz = SampleHexTiled(mat.textureIndices.y, uvZ).rg * 2.0 - 1.0;
        
        float3 tx = float3(nx, sqrt(max(0.0, 1.0 - dot(nx, nx))));
        float3 ty = float3(ny, sqrt(max(0.0, 1.0 - dot(ny, ny))));
        float3 tz = float3(nz, sqrt(max(0.0, 1.0 - dot(nz, nz))));
        
        float3 blendNormalX = float3(worldNormal.x, worldNormal.y + tx.y, worldNormal.z + tx.x);
        float3 blendNormalY = float3(worldNormal.x + ty.x, worldNormal.y, worldNormal.z + ty.y);
        float3 blendNormalZ = float3(worldNormal.x + tz.x, worldNormal.y + tz.y, worldNormal.z);
        
        normal = normalize(blendNormalX * w.x + blendNormalY * w.y + blendNormalZ * w.z);
    }
}

[shader("fragment")]
GBufferOutput fragmentMain(VertexOutput input)
{
    // Important: Only render voxel chunks!
    if (input.materialId != 0xFFFFFFFEu)
    {
        discard;
    }

    float4 albedo = float4(1.0);
    float roughness = 1.0;
    float metallic = 0.0;
    float3 emissive = float3(0.0);
    float3 normal = normalize(input.worldNormal);

    uint grassId = input.voxelFlags & 0xFFFFu;
    uint rockId = (input.voxelFlags >> 16) & 0xFFFFu;
    uint dirtId = input.voxelPad;

    MaterialData mGrass = materials[grassId];
    MaterialData mRock = materials[rockId];
    MaterialData mDirt = materials[dirtId];

    float3 blendW = abs(input.worldNormal);
    blendW = max(blendW - 0.2, 0.0);
    blendW /= (blendW.x + blendW.y + blendW.z + 0.0001);

    float slope = abs(input.worldNormal.y);
    float wGrass = smoothstep(0.75, 0.9, slope);
    float wRock = smoothstep(0.7, 0.5, slope);
    float wDirt = max(0.0, 1.0 - wGrass - wRock);

    float sum = wGrass + wRock + wDirt + 0.0001;
    wGrass /= sum;
    wRock /= sum;
    wDirt /= sum;

    float4 aGrass, aRock, aDirt;
    float3 nGrass, nRock, nDirt;
    float4 rmaoGrass, rmaoRock, rmaoDirt;

    if (wGrass > 0.01) EvaluateVoxelMaterial(mGrass, input.worldPos, blendW, input.worldNormal, aGrass, nGrass, rmaoGrass);
    else { aGrass = float4(0); nGrass = float3(0); rmaoGrass = float4(0); }

    if (wRock > 0.01) EvaluateVoxelMaterial(mRock, input.worldPos, blendW, input.worldNormal, aRock, nRock, rmaoRock);
    else { aRock = float4(0); nRock = float3(0); rmaoRock = float4(0); }

    if (wDirt > 0.01) EvaluateVoxelMaterial(mDirt, input.worldPos, blendW, input.worldNormal, aDirt, nDirt, rmaoDirt);
    else { aDirt = float4(0); nDirt = float3(0); rmaoDirt = float4(0); }

    albedo = aGrass * wGrass + aRock * wRock + aDirt * wDirt;
    normal = normalize(nGrass * wGrass + nRock * wRock + nDirt * wDirt);
    
    float4 rmao = rmaoGrass * wGrass + rmaoRock * wRock + rmaoDirt * wDirt;
    roughness = rmao.g;
    metallic = rmao.b;

    GBufferOutput output;
    output.albedo = float4(albedo.rgb, 1.0);
    output.normal = encodeOctahedral(normal);
    output.roughMetal = float2(roughness, metallic);
    output.emissive = emissive;
    output.hiz_depth = input.clipPosition.z;
    return output;
}
