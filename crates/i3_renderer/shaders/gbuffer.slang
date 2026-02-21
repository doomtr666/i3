// GBuffer pass — Phase 1: hardcoded cubes
//
// Outputs to 4 MRT:
//   RT0 (R8G8B8A8_SRGB):       Albedo.rgb + unused alpha
//   RT1 (R16G16_SFLOAT):       Normal.xy (octahedral encoding)
//   RT2 (R8G8_UNORM):          Roughness + Metallic
//   RT3 (R11G11B10_UFLOAT):    Emissive.rgb

struct PushConstants
{
    float4x4 viewProjection;
    float4x4 model;
};

[[vk::push_constant]]
PushConstants pc;

struct VertexInput
{
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float3 color    : COLOR;
};

struct VertexOutput
{
    float4 clipPosition : SV_Position;
    float3 worldNormal  : NORMAL;
    float3 color        : COLOR;
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

    // Transform normal (no non-uniform scale in Phase 1, so model matrix is fine)
    output.worldNormal = normalize(mul((float3x3)pc.model, input.normal));
    output.color = input.color;
    return output;
}

[shader("fragment")]
GBufferOutput fragmentMain(VertexOutput input)
{
    GBufferOutput output;
    output.albedo     = float4(input.color, 1.0);
    output.normal     = encodeOctahedral(normalize(input.worldNormal));
    output.roughMetal = float2(0.5, 0.0); // Hardcoded: roughness=0.5, metallic=0.0
    output.emissive   = float3(0.0, 0.0, 0.0);
    return output;
}
