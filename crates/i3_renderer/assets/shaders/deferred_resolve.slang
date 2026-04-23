// assets/system/shaders/deferred_resolve.slang
#include "lighting.slangh"

#define USE_RT
#define USE_IBL

struct PushConstants
{
    uint3 grid_size;
    float ao_strength; // 0.0 = no AO, 1.0 = full AO
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]]
Texture2D gbufferAlbedo;
[[vk::binding(1, 0)]]
Texture2D gbufferNormal;
[[vk::binding(2, 0)]]
Texture2D gbufferRoughMetal;
[[vk::binding(3, 0)]]
Texture2D gbufferEmissive;
[[vk::binding(4, 0)]]
Texture2D depthBuffer;

[[vk::binding(5, 0)]]
StructuredBuffer<LightData> lights;
[[vk::binding(6, 0)]]
StructuredBuffer<ClusterGridData> cluster_grid;
[[vk::binding(7, 0)]]
StructuredBuffer<uint> cluster_light_indices;

[[vk::binding(8, 0)]]
StructuredBuffer<ExposureData> exposure_buffer;

[[vk::binding(9, 0)]]
RaytracingAccelerationStructure tlas;

[[vk::binding(10, 0)]]
Texture2D<float> aoTexture;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

#ifdef USE_RT

bool traceShadow(RayDesc ray)
{
    // Use optimized ray query for shadow testing - terminates at first hit
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);

    // Process intersections (simplified for shadow rays)
    while (q.Proceed())
    {
        // For non-opaque triangles, force them to be opaque (simplified alpha handling)
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit(); // Accept intersection as solid
    }

    // Return true if any intersection was found (surface is in shadow)
    return (q.CommittedStatus() != COMMITTED_NOTHING);
}

#endif

[shader("vertex")]
VertexOutput vertexMain(uint vertexId: SV_VertexID)
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

    float depth = depthBuffer.Sample(bindlessSamplers[SAMPLER_NEAREST], uv).r;
    if (depth <= 0.0)
        discard;

    // Read GBuffer
    float3 albedo = gbufferAlbedo.Sample(bindlessSamplers[SAMPLER_LINEAR], uv).rgb;
    float2 nEnc = gbufferNormal.Sample(bindlessSamplers[SAMPLER_LINEAR], uv).rg;
    float3 normal = decodeOctahedral(nEnc);

    float2 roughMetal = gbufferRoughMetal.Sample(bindlessSamplers[SAMPLER_LINEAR], uv).rg;
    float roughness = roughMetal.r;
    float metallic = roughMetal.g;
    float3 emissive = gbufferEmissive.Sample(bindlessSamplers[SAMPLER_LINEAR], uv).rgb;

    // World & View space reconstruct
    float x = uv.x * 2.0 - 1.0;
    float y = (1.0 - uv.y) * 2.0 - 1.0;
    float4 clip_pos = float4(x, y, depth, 1.0);

    float4 world_pos_h = mul(g_common.invViewProjection, clip_pos);
    float3 world_pos = world_pos_h.xyz / world_pos_h.w;

    float4 view_pos_h = mul(g_common.invProjection, clip_pos);
    float view_z = -(view_pos_h.z / view_pos_h.w);

    // Calculate cluster index
    uint tile_x = clamp(uint(uv.x * float(pc.grid_size.x)), 0u, pc.grid_size.x - 1);
    uint tile_y = clamp(uint(uv.y * float(pc.grid_size.y)), 0u, pc.grid_size.y - 1);

    float z_slice = 0.0;
    if (view_z > g_common.GetNearPlane())
        z_slice = log(view_z / g_common.GetNearPlane()) * float(pc.grid_size.z) / log(g_common.GetFarPlane() / g_common.GetNearPlane());

    uint tile_z = clamp(uint(max(0.0, z_slice)), 0u, pc.grid_size.z - 1);

    uint cluster_index = tile_z * pc.grid_size.x * pc.grid_size.y +
                         tile_y * pc.grid_size.x +
                         tile_x;

    // Fetch lights
    ClusterGridData grid_item = cluster_grid[cluster_index];

    float3 V = normalize(g_common.GetCameraPos() - world_pos);
#ifdef USE_IBL
    // IBL ambient
    float3 F0 = lerp(float3(0.04), albedo, metallic);
    float NdotV = saturate(dot(normal, V));

    float3 kS = F0 + (max(float3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
    float3 kD = (1.0 - kS) * (1.0 - metallic);

    float3 irradiance = float3(0.0);
    if (g_common.GetIblIrrIndex() != 0)
    {
        irradiance = bindlessTextures[g_common.GetIblIrrIndex()].Sample(bindlessSamplers[SAMPLER_LINEAR], encodeHemiOcta(normal)).rgb;
    }
    float3 diffuse_ambient = kD * albedo * irradiance;

    float ao = lerp(1.0f, aoTexture.Sample(bindlessSamplers[SAMPLER_NEAREST], uv).r, pc.ao_strength);
    float3 color = emissive + diffuse_ambient * g_common.GetIblIntensity() * ao;
#else
    float3 color = emissive + 0.005 * albedo;
#endif

    for (uint i = 0; i < grid_item.count; ++i)
    {
        uint light_idx = cluster_light_indices[1 + grid_item.offset + i];
        LightData light = lights[light_idx];

        float3 diff = light.position - world_pos;
        float distance = length(diff);
        float3 L = diff / max(distance, 0.0001);

        float3 radiance = float3(0.0, 0.0, 0.0);

        if (light.light_type == 1) // directional
        {
            L = normalize(-light.direction);

#ifdef USE_RT
            // visibility test
            RayDesc rd;
            rd.Origin = world_pos;
            rd.Direction = L;
            rd.TMin = 0.001f;
            rd.TMax = 1e6;
            if (!traceShadow(rd))
#endif
            {
                radiance = light.color * light.intensity;
            }
        }
        else // point light
        {
#ifdef USE_RT
            // visibility test
            RayDesc rd;
            rd.Origin = world_pos;
            rd.Direction = L;
            rd.TMin = 0.001f;
            rd.TMax = distance;

            if (!traceShadow(rd))
#endif
            {
                float attenuation = 1.0 / max(distance * distance, 0.01);
                float r = distance / light.radius;
                float window = clamp(1.0 - pow(r, 4.0), 0.0, 1.0);
                attenuation *= window * window;

                radiance = light.color * light.intensity * attenuation;
            }
        }

        color += computeBRDF(L, V, normal, albedo, metallic, roughness, radiance);
    }

    if (g_common.GetDebugChannel() == 12) // DebugChannel::ClusterGrid
    {
        float3 debug_color = float3(((tile_x % 2) ^ (tile_y % 2) ^ (tile_z % 2)) != 0 ? 0.2 : 0.05);
        color = debug_color;
    }
    else if (g_common.GetDebugChannel() == 11) // DebugChannel::LightDensity
    {
        float density = float(grid_item.count) / 64.0;
        color = float3(density, 0.0, 1.0 - density);
    }


    float prev_ex = exposure_buffer[0].new_exposure;
    if (prev_ex == 0.0)
        prev_ex = 1.0;
    color *= prev_ex;

    return float4(color, 1.0);
}
