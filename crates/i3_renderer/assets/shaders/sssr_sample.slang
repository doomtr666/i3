// Stochastic Screen-Space Reflections — Sample pass
//
// One ray per pixel per frame, direction importance-sampled from GGX.

#include "lighting.slangh"
#include "sampling.slangh"

struct SssrSamplePc
{
    float thickness; // max view-space depth gap for hit acceptance
    uint enabled;
    uint downsample_factor; // 1 = full-res, 2 = half-res, etc.
    uint num_samples;       // 1..16, clamped on CPU side
};
[[vk::push_constant]]
SssrSamplePc pc;

[[vk::binding(0, 0)]]
Texture2D<float> depthBuffer;
[[vk::binding(1, 0)]]
Texture2D<float> hizBuffer;
[[vk::binding(2, 0)]]
Texture2D<float3> gbufferAlbedo;
[[vk::binding(3, 0)]]
Texture2D<float2> gbufferNormal;
[[vk::binding(4, 0)]]
Texture2D<float2> gbufferRoughMetal;
[[vk::binding(5, 0)]]
Texture2D<float4> hdrLit;
[[vk::binding(6, 0)]]
RWTexture2D<float4> ssrOutput;

// ─── Helpers ─────────────────────────────────────────────────────────────────

float3 projectViewToScreenSpace(float3 viewPos)
{
    float4 clip = mul(g_common.projection, float4(viewPos, 1.0f));
    clip.w = max(clip.w, 1e-6f);
    float2 ndc = clip.xy / clip.w;
    return float3(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f, clip.z / clip.w);
}

// Hi-Z tracing optimized for Reverse-Z (1.0 = Near, 0.0 = Far)
// Returns true if a valid intersection is found, and stores the hit's UVZ position.
bool traceHiZ(
    float3 startUVZ,     // Starting coordinates: UV [0,1] + Depth [0,1]
    float3 rayDirUVZ,    // Direction vector (projected P_end - P_start)
    float4x4 invProj,    // Inverse matrix for thickness testing
    float maxThickness,  // Thickness tolerance in meters
    out float3 outHitUVZ // Result in UVZ (if hit)
)
{
    const int MaxMipLevel = 5;
    float2 screenSize = g_common.GetScreenSize();

    // 1. SCALE TO PIXEL SPACE
    float3 rayPos = float3(startUVZ.xy * screenSize, startUVZ.z);
    float3 rayDir = float3(rayDirUVZ.xy * screenSize, rayDirUVZ.z);

    float rayLength = length(rayDir.xy);
    if (rayLength < 0.001f)
        return false;

    // DDA algorithm pre-computations (NaN protection)
    float2 dir = rayDir.xy;
    dir.x = abs(dir.x) < 1e-5f ? (dir.x >= 0.0f ? 1e-5f : -1e-5f) : dir.x;
    dir.y = abs(dir.y) < 1e-5f ? (dir.y >= 0.0f ? 1e-5f : -1e-5f) : dir.y;
    float2 invDir = 1.0f / dir;

    int2 stepDir = int2(rayDir.x >= 0.0f ? 1 : -1, rayDir.y >= 0.0f ? 1 : -1);
    float2 crossStep = float2(stepDir.x > 0 ? 1.0f : 0.0f, stepDir.y > 0 ? 1.0f : 0.0f);

    float tEpsilon = 0.01f / rayLength;

    float t = 0.0f;
    int mipLevel = MaxMipLevel;  // Fix 1: start coarse, descend on intersection
    const uint MAX_ITER = 64;    // Fix 4: was 100

    for (uint i = 0; i < MAX_ITER; i++)
    {
        float3 currentPos = rayPos + rayDir * t;

        // Stop condition: Off-screen
        if (currentPos.x < 0.0f || currentPos.x >= screenSize.x ||
            currentPos.y < 0.0f || currentPos.y >= screenSize.y ||
            currentPos.z < 0.0f || currentPos.z > 1.0f)
        {
            return false;
        }

        float cellSize = float(1 << mipLevel);  // Fix 5: integer shift instead of exp2
        float2 cellIndex = floor((currentPos.xy + rayDir.xy * 0.01f) / cellSize);

        float2 distanceToBoundary = (cellIndex + crossStep) * cellSize - currentPos.xy;
        float2 tToBoundary = distanceToBoundary * invDir;

        float tAdvance = min(tToBoundary.x, tToBoundary.y);
        float nextT = t + tAdvance;

        // Heuristic test (Reverse-Z)
        float z1 = currentPos.z;
        float z2 = rayPos.z + rayDir.z * nextT;
        float minZ_segment = min(z1, z2);

        // Pre-compute UV for thickness test — independent of texture fetch (Fix 2 + Fix 3)
        float2 hit_uv = currentPos.xy / screenSize;

        int2 mipCoords = int2(cellIndex);
        float cellMaxZ = hizBuffer.Load(int3(mipCoords, mipLevel)).r;

        // Independent of cellMaxZ — hides texture fetch latency (Fix 2)
        float t_next = nextT + tEpsilon;
        int mip_next = min(mipLevel + 1, MaxMipLevel);

        if (minZ_segment <= cellMaxZ)
        {
            if (mipLevel == 0)
            {
                // --- INTEGRATED THICKNESS TEST (Fix 3: avoids 2x mat4-vec4 multiply) ---
                float ray_view_z = viewSpacePosFromUV(hit_uv, currentPos.z, invProj).z;
                float geo_view_z = viewSpacePosFromUV(hit_uv, cellMaxZ, invProj).z;

                float thickness = abs(geo_view_z - ray_view_z);

                if (thickness > maxThickness)
                {
                    // False positive: passing behind a thin object (e.g., a garland).
                    t = nextT;
                    mipLevel = mip_next;
                    continue;
                }

                // Valid hit
                outHitUVZ = float3(hit_uv, currentPos.z);
                return true;
            }
            else
            {
                mipLevel--;
            }
        }
        else
        {
            t = t_next;
            mipLevel = mip_next;
        }
    }

    return false;
}

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 px = id.xy;
    uint f = pc.downsample_factor;
    // Bounds check against low-res dispatch dimensions
    if (any(px >= uint2(g_common.GetScreenWidth() / f, g_common.GetScreenHeight() / f)))
        return;

    // Full-res pixel at the center of the block this low-res pixel represents
    uint2 full_px = px * f + f / 2;

    if (pc.enabled == 0u)
    {
        ssrOutput[px] = 0;
        return;
    }

    float depth = depthBuffer.Load(int3(full_px, 0)).r;
    if (depth <= 0.0f)
    {
        ssrOutput[int2(px)] = 0;
        return;
    }

    float2 rm = gbufferRoughMetal.Load(int3(full_px, 0)).rg;
    float metallic = rm.g;

    float roughness = rm.r;
    float alpha = roughness * roughness;

    float3 P = viewSpacePosFromPixel(full_px, depth, g_common.GetScreenSize(), g_common.invProjection);
    float3 N_world = decodeOctahedral(gbufferNormal.Load(int3(full_px, 0)).rg);
    // Albedo read from GBuffer (required for metal Fresnel)
    float3 albedo = gbufferAlbedo.Load(int3(full_px, 0));
    float3 F0 = lerp(FRESNEL_F0_DIELECTRIC, albedo, metallic);

    float3x3 viewRot = float3x3(g_common.view[0].xyz, g_common.view[1].xyz, g_common.view[2].xyz);
    float3 N = normalize(mul(viewRot, N_world));
    float3 V = normalize(-P);

    uint num_samples = pc.num_samples;

    float3 accum_color = float3(0, 0, 0);

    for (uint i = 0; i < num_samples; i++)
    {
        float3 noise = pcg3f(uint3(px, g_common.GetFrameIndex() + i));

        // 1. Build basis
        float3 T, B;
        buildBasis(N, T, B);

        // 2. Transform V to tangent space
        float3 V_tangent = float3(dot(V, T), dot(V, B), dot(V, N));

        // 3. Sample with tangential V
        float3 H_local = sampleGGXVNDF(V_tangent, alpha, alpha, noise.x, noise.y);

        // 4. Transform H back to View Space
        float3 H = normalize(T * H_local.x + B * H_local.y + N * H_local.z);

        float3 D = reflect(-V, H);

        // Inverse matrix to transform back to World Space (adjust to your specific structure)
        float3x3 invViewRot = float3x3(g_common.invView[0].xyz, g_common.invView[1].xyz, g_common.invView[2].xyz);
        float3 D_world = normalize(mul(invViewRot, D));

        float3 output_color = bindlessTextures[g_common.GetIblEnvIndex()].SampleLevel(bindlessSamplers[SAMPLER_LINEAR], dirToEquirect(D_world), 0).rgb;

        // Reject if ray plunges below geometry
        if (dot(D, N) > 0.001f)
        {

            // Calculate ray end point (e.g., 100 meters)
            float3 P_end = P + D * 1000.0f;

            float3 screen_P = projectViewToScreenSpace(P);
            float3 screen_P_end = projectViewToScreenSpace(P_end);

            float3 screen_RayDir = normalize(screen_P_end - screen_P);

            float2 screen_RayDir_Pixels = screen_RayDir.xy * g_common.GetScreenSize();
            float step_scale = 1.0f / max(abs(screen_RayDir_Pixels.x), abs(screen_RayDir_Pixels.y));
            float3 ray_step_screen = screen_RayDir * step_scale;

            // --- THE FIX (Ray Bump) ---
            // Offset to avoid self-intersection
            float start_offset = 2.0f; // + noise.z;
            float3 safe_start_P = screen_P + ray_step_screen * start_offset;

            // Trace with integrated thickness test
            float3 hit_UVZ;
            bool valid_hit = traceHiZ(safe_start_P, ray_step_screen, g_common.invProjection, pc.thickness, hit_UVZ);

            if (valid_hit)
            {

                // 3. Backface validation
                uint2 hit_px = uint2(hit_UVZ.xy * g_common.GetScreenSize());
                float3 hit_N_world = decodeOctahedral(gbufferNormal.Load(int3(hit_px, 0)).rg);
                float3 hit_N = normalize(mul(viewRot, hit_N_world)); // Transform back to View Space!
                float N_dot_D = dot(hit_N, D);
                if (N_dot_D < -0.001f)
                {
                    output_color = hdrLit.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], hit_UVZ.xy, 0).rgb;
                }
            }
        }

        float3 brdf_weight = fresnelSchlick(max(dot(V, H), 0.0f), F0) * GGXVNDFWeight(V, D, N, alpha);
        accum_color += output_color * brdf_weight;
    }
    ssrOutput[px] = float4(accum_color / float(num_samples), 1.0f);
}
