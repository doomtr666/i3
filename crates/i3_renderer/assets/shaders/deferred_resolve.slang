// assets/system/shaders/deferred_resolve.slang
#include "lighting.slangh"

#define USE_RT
#define USE_IBL

struct PushConstants
{
    float4x4 inv_view_proj;
    float4x4 inv_projection;
    float3 camera_pos;
    float near_plane;

    uint3 grid_size;
    float far_plane;

    float2 screen_dimensions;
    uint debug_mode;

    uint ibl_lut_index;
    uint ibl_irr_index;
    uint ibl_pref_index;
    float ibl_intensity;
    uint pad1;
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

float3 offsetRay(in float3 p, in float3 n)
{
    // Smallest epsilon that can be added without losing precision is 1.19209e-07, but we play safe
    const float epsilon = 1.0f / 256.0f; // Safe epsilon

    float magnitude = length(p);
    float offset = epsilon * magnitude;
    // multiply the direction vector by the smallest offset
    float3 offsetVector = n * offset;
    // add the offset vector to the starting point
    float3 offsetPoint = p + offsetVector;

    return offsetPoint;
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
    {
        discard;
    }

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

    float4 world_pos_h = mul(pc.inv_view_proj, clip_pos);
    float3 world_pos = world_pos_h.xyz / world_pos_h.w;

    float4 view_pos_h = mul(pc.inv_projection, clip_pos);
    float view_z = -(view_pos_h.z / view_pos_h.w);

    // Calculate cluster index
    uint tile_x = clamp(uint(uv.x * float(pc.grid_size.x)), 0u, pc.grid_size.x - 1);
    uint tile_y = clamp(uint(uv.y * float(pc.grid_size.y)), 0u, pc.grid_size.y - 1);

    float z_slice = 0.0;
    if (view_z > pc.near_plane)
    {
        z_slice = log(view_z / pc.near_plane) * float(pc.grid_size.z) / log(pc.far_plane / pc.near_plane);
    }
    uint tile_z = clamp(uint(max(0.0, z_slice)), 0u, pc.grid_size.z - 1);

    uint cluster_index = tile_z * pc.grid_size.x * pc.grid_size.y +
                         tile_y * pc.grid_size.x +
                         tile_x;

    // Fetch lights
    ClusterGridData grid_item = cluster_grid[cluster_index];

    float3 V = normalize(pc.camera_pos - world_pos);
#ifdef USE_IBL
    // IBL ambient
    float3 F0 = lerp(float3(0.04), albedo, metallic);
    float NdotV = saturate(dot(normal, V));

    float3 kS = F0 + (max(float3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
    float3 kD = (1.0 - kS) * (1.0 - metallic);

    float3 irradiance = float3(0.0);
    if (pc.ibl_irr_index != 0xFFFFFFFF)
    {
        irradiance = bindlessTextures[pc.ibl_irr_index].Sample(bindlessSamplers[SAMPLER_LINEAR], encodeHemiOcta(normal)).rgb;
    }
    float3 diffuse_ambient = kD * albedo * irradiance;

    float3 spec_ambient = float3(0.0);
    if (pc.ibl_pref_index != 0xFFFFFFFF && pc.ibl_lut_index != 0xFFFFFFFF)
    {
        float3 R = reflect(-V, normal);
        float mip_level = roughness * 5.0; // MAX_PREFILTER_MIP = 5
        float3 prefiltered = bindlessTextures[pc.ibl_pref_index].SampleLevel(bindlessSamplers[SAMPLER_LINEAR], encodeHemiOcta(R), mip_level).rgb;
        float2 brdf = bindlessTextures[pc.ibl_lut_index].Sample(bindlessSamplers[SAMPLER_LINEAR], float2(NdotV, roughness)).rg;
        spec_ambient = prefiltered * (F0 * brdf.x + brdf.y);
    }

    float3 color = emissive + (diffuse_ambient + spec_ambient) * pc.ibl_intensity;
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
            rd.Origin = offsetRay(world_pos, normal);
            rd.Direction = L;
            rd.TMin = 0;
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
            rd.Origin = offsetRay(world_pos, normal);
            rd.Direction = L;
            rd.TMin = 0;
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

    if (pc.debug_mode == 2)
    {
        float3 debug_color = float3(((tile_x % 2) ^ (tile_y % 2) ^ (tile_z % 2)) != 0 ? 0.2 : 0.05);
        color = debug_color;
    }
    else if (pc.debug_mode == 1)
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
