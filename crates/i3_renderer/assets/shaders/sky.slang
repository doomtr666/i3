// assets/system/shaders/sky.slang
#include "common.slangh"
struct VertexOutput
{
    float4 position : SV_Position;
    float3 view_dir : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId: SV_VertexID)
{
    VertexOutput output;
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    // Depth = 0.0 to render behind everything (reverse-Z: far = 0)
    float4 pos = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);

    float4 world_pos_h = mul(g_common.invViewProjection, pos);
    float3 world_pos = world_pos_h.xyz / world_pos_h.w;

    output.position = pos;
    output.view_dir = normalize(world_pos - g_common.GetCameraPos());

    return output;
}

float3 get_sky_color(float3 dir, float3 sun_dir, float3 sun_color)
{
    float cos_theta = dot(dir, sun_dir);
    float zenith_cos = clamp(dir.y, 0.05, 1.0);

    float3 rayleigh = float3(0.05, 0.2, 0.5) * (1.0 / pow(zenith_cos, 0.5));
    float mie = pow(max(cos_theta, 0.0), 32.0) * 0.5;
    float sun_disk = smoothstep(0.998, 0.999, cos_theta) * 50.0;

    float3 sky = lerp(float3(0.1, 0.1, 0.2), float3(0.4, 0.6, 1.0), zenith_cos);
    sky += (mie + sun_disk) * sun_color;

    float horizon = pow(1.0 - zenith_cos, 4.0);
    sky += float3(0.4, 0.5, 0.6) * horizon;

    return sky;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    float3 dir = normalize(input.view_dir);
    float3 color;

    if (g_common.GetIblEnvIndex() != 0)
    {
        // IBL sky — sample equirect from bindless array
        color = bindlessTextures[g_common.GetIblEnvIndex()].SampleLevel(bindlessSamplers[SAMPLER_LINEAR], dirToEquirect(dir), 0).rgb;
    }
    else
    {
        // Procedural sky fallback
        color = get_sky_color(dir, -g_common.GetSunDir(), g_common.GetSunColor() * g_common.GetSunIntensity());
    }

    return float4(color, 1.0);
}

