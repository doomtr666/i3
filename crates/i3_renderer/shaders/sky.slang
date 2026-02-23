struct PushConstants
{
    float4x4 inv_view_proj;
    float3 camera_pos;
    float pad0;
    float3 sun_direction;
    float sun_intensity;
    float3 sun_color;
    float pad1;
};

[[vk::push_constant]]
PushConstants pc;

struct VertexOutput
{
    float4 position : SV_Position;
    float3 view_dir : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    VertexOutput output;
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    float4 pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0); // Depth = 1.0
    
    // Unproject to find world direction
    float4 world_pos_h = mul(pc.inv_view_proj, pos);
    float3 world_pos = world_pos_h.xyz / world_pos_h.w;
    
    output.position = pos;
    output.view_dir = normalize(world_pos - pc.camera_pos);
    
    return output;
}

// Simplified Atmospheric Scattering (Rayleigh + Mie approximation)
float3 get_sky_color(float3 dir, float3 sun_dir, float3 sun_color)
{
    float cos_theta = dot(dir, sun_dir);
    float zenith_cos = clamp(dir.y, 0.05, 1.0);
    
    // Rayleigh scattering (blue sky)
    float3 rayleigh = float3(0.05, 0.2, 0.5) * (1.0 / pow(zenith_cos, 0.5));
    
    // Mie scattering (haze near sun)
    float mie = pow(max(cos_theta, 0.0), 32.0) * 0.5;
    
    // Sun disk
    float sun_disk = smoothstep(0.998, 0.999, cos_theta) * 50.0;
    
    float3 sky = lerp(float3(0.1, 0.1, 0.2), float3(0.4, 0.6, 1.0), zenith_cos);
    sky += (mie + sun_disk) * sun_color;
    
    // Horizon glow
    float horizon = pow(1.0 - zenith_cos, 4.0);
    sky += float3(0.4, 0.5, 0.6) * horizon;
    
    return sky;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    float3 dir = normalize(input.view_dir);
    
    // sun_direction is the direction the light is pointing (downwards).
    // The sun is in the opposite direction.
    float3 color = get_sky_color(dir, -pc.sun_direction, pc.sun_color * pc.sun_intensity);
    
    return float4(color, 1.0);
}
