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
    uint pad;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]] Texture2D gbufferAlbedo;
[[vk::binding(0, 0)]] SamplerState albedoSampler;

[[vk::binding(1, 0)]] Texture2D gbufferNormal;
[[vk::binding(1, 0)]] SamplerState normalSampler;

[[vk::binding(2, 0)]] Texture2D gbufferRoughMetal;
[[vk::binding(2, 0)]] SamplerState roughMetalSampler;

[[vk::binding(3, 0)]] Texture2D gbufferEmissive;
[[vk::binding(3, 0)]] SamplerState emissiveSampler;

[[vk::binding(4, 0)]] Texture2D depthBuffer;
[[vk::binding(4, 0)]] SamplerState depthSampler;

struct LightData {
    float3 position;
    float radius;
    float3 color;
    float intensity;
    float3 direction;
    uint light_type;
};

[[vk::binding(5, 0)]] StructuredBuffer<LightData> lights;

struct ClusterGridData {
    uint offset;
    uint count;
};

[[vk::binding(6, 0)]] StructuredBuffer<ClusterGridData> cluster_grid;
[[vk::binding(7, 0)]] StructuredBuffer<uint> cluster_light_indices;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

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

// Minimal PBR (Cook-Torrance)
#define PI 3.14159265359

float DistributionGGX(float3 N, float3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / max(denom, 0.0000001); // Prevent divide by zero
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float3 computeBRDF(float3 L, float3 V, float3 N, float3 albedo, float metallic, float roughness, float3 radiance) {
    float3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    
    if (NdotL <= 0.0) return float3(0.0, 0.0, 0.0);

    float3 F0 = float3(0.04, 0.04, 0.04); 
    F0 = lerp(F0, albedo, metallic);

    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    float3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    float3 specular = numerator / denominator;

    float3 kS = F;
    float3 kD = float3(1.0, 1.0, 1.0) - kS;
    kD *= 1.0 - metallic;

    return (kD * albedo / PI + specular) * radiance * NdotL;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    float2 uv = input.uv;
    
    float depth = depthBuffer.Sample(depthSampler, uv).r;
    if (depth >= 1.0) { // Background
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    // Read GBuffer
    float3 albedo = gbufferAlbedo.Sample(albedoSampler, uv).rgb;
    float2 nEnc = gbufferNormal.Sample(normalSampler, uv).rg;
    float3 normal = decodeOctahedral(nEnc);
    
    // Convert to SRGB? Assuming albedo is already linear from SRGB read, but wait, if it's stored as SRGB texture, sampling returns linear.
    // RoughMetal is UNORM.
    float2 roughMetal = gbufferRoughMetal.Sample(roughMetalSampler, uv).rg;
    float roughness = roughMetal.r;
    float metallic = roughMetal.g;
    float3 emissive = gbufferEmissive.Sample(emissiveSampler, uv).rgb;

    // World & View space reconstruct
    float x = uv.x * 2.0 - 1.0;
    float y = (1.0 - uv.y) * 2.0 - 1.0;
    float4 clip_pos = float4(x, y, depth, 1.0);
    
    float4 world_pos_h = mul(pc.inv_view_proj, clip_pos);
    float3 world_pos = world_pos_h.xyz / world_pos_h.w;

    float4 view_pos_h = mul(pc.inv_projection, clip_pos);
    float view_z = -(view_pos_h.z / view_pos_h.w); // Right-handed, view-space Z points towards camera

    // Calculate cluster index
    uint tile_x = uint(uv.x * float(pc.grid_size.x));
    uint tile_y = uint(uv.y * float(pc.grid_size.y));
    
    float z_slice = 0.0;
    if (view_z > pc.near_plane) {
        z_slice = log(view_z / pc.near_plane) * float(pc.grid_size.z) / log(pc.far_plane / pc.near_plane);
    }
    uint tile_z = clamp(uint(z_slice), 0, pc.grid_size.z - 1);

    uint cluster_index = tile_z * pc.grid_size.x * pc.grid_size.y + 
                         tile_y * pc.grid_size.x + 
                         tile_x;

    // Fetch lights
    ClusterGridData grid_item = cluster_grid[cluster_index];
    
    float3 V = normalize(pc.camera_pos - world_pos);
    float3 color = emissive;

    for (uint i = 0; i < grid_item.count; ++i) {
        uint light_idx = cluster_light_indices[1 + grid_item.offset + i];
        LightData light = lights[light_idx];
        
        float3 L = normalize(light.position - world_pos);
        float distance = length(light.position - world_pos);

        float3 radiance = float3(0.0, 0.0, 0.0);

        if (light.light_type == 1) { // Directional
            L = normalize(-light.direction);
            radiance = light.color * light.intensity;
        } else { // Point
            float attenuation = 1.0 / (distance * distance);
            
            // Simple windowing function to smoothly fade out at radius
            float r = distance / light.radius;
            float window = clamp(1.0 - pow(r, 4.0), 0.0, 1.0);
            attenuation *= window * window;

            radiance = light.color * light.intensity * attenuation;
        }

        color += computeBRDF(L, V, normal, albedo, metallic, roughness, radiance);
    }
    
    if (pc.debug_mode == 2) {
        // Cluster Grid (Checkerboard based on cluster_index)
        float3 debug_color = float3(((tile_x % 2) ^ (tile_y % 2) ^ (tile_z % 2)) != 0 ? 0.2 : 0.05);
        color = debug_color;
    } else if (pc.debug_mode == 1) {
        // Light Density (Heatmap based on light count)
        float density = float(grid_item.count) / 64.0; // Assume max 64 lights is hot red
        color = float3(density, 0.0, 1.0 - density);
    } else {
        // Tonemapping (Reinhard)
        color = color / (color + float3(1.0, 1.0, 1.0));
        // Gamma correction
        color = pow(color, float3(1.0/2.2, 1.0/2.2, 1.0/2.2));
    }

    return float4(color, 1.0);
}
