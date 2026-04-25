// Ground-Truth Ambient Occlusion — Main compute pass
// Robust implementation with absolute view-vector angles and strict 3D distance attenuation.

#include "common.slangh"
#include "sampling.slangh"

struct GtaoPushConstants
{
    float4x4 invProjection;
    float4x4 view;
    float2 screenSize;
    float radius;          // World-space sampling radius (metres)
    float falloff;         // Distance falloff power (e.g. 2.0)
    uint sliceCount;       // Azimuthal directions (2 = perf, 3 = quality)
    uint stepCount;        // Steps per direction (4–6)
    uint frameIndex;       // For golden-ratio temporal rotation
    uint enabled;          // 0 = write 1.0 (disabled), 1 = run GTAO
    uint blue_noise_index; // Bindless index of the 128×128 RG8 blue-noise texture
    uint _pad;
};

[[vk::push_constant]]
GtaoPushConstants pc;

[[vk::binding(0, 0)]]
Texture2D<float> depthBuffer;
[[vk::binding(1, 0)]]
Texture2D<float2> gbufferNormal;
[[vk::binding(2, 0)]]
RWTexture2D<float> aoOutput;

// ─── Position reconstruction ──────────────────────────────────────────────────

float3 sampleViewPos(float2 screenPos)
{
    // round() restores symmetry so +dir and -dir sample at identical pixel offsets
    int2 ipx = int2(round(clamp(screenPos, float2(0.0f), float2(pc.screenSize) - 1.0f)));
    return viewSpacePosFromPixel(uint2(ipx), depthBuffer[ipx].r, pc.screenSize, pc.invProjection);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

[numthreads(8, 8, 1)]
void computeMain(uint3 dispatchId: SV_DispatchThreadID)
{
    uint2 px = dispatchId.xy;
    if (any(px >= uint2(pc.screenSize)))
        return;

    if (pc.enabled == 0)
    {
        aoOutput[int2(px)] = 1.0f;
        return;
    }

    float depth = depthBuffer[int2(px)].r;
    if (depth <= 0.0f) // reverse-Z: 0 = sky / far plane
    {
        aoOutput[int2(px)] = 1.0f;
        return;
    }

    // 1. Setup base vectors
    float3 P = viewSpacePosFromPixel(px, depth, pc.screenSize, pc.invProjection);
    float3 V = normalize(-P); // Vector pointing to the camera

    // Decode and transform normal to view space
    float2 nEnc = gbufferNormal[int2(px)].rg;
    float3 N_world = decodeOctahedral(nEnc);
    float3x3 viewRot = float3x3(pc.view[0].xyz, pc.view[1].xyz, pc.view[2].xyz);
    float3 N = normalize(mul(viewRot, N_world));

    // Screen-space radius clamping
    float inv_tan_fov = abs(pc.invProjection[0][0]);
    float ss_radius = clamp(
        pc.radius * inv_tan_fov * (pc.screenSize.x * 0.5f) / max(-P.z, 0.01f),
        2.0f, 256.0f);
    float step_size = ss_radius / float(pc.stepCount + 1);

    // Dither setup
    uint3 h = pcg3(uint3(px, pc.frameIndex));
    float angle_base = float(h.x) * (1.0f / 4294967296.0f);
    float step_noise = float(h.y) * (1.0f / 4294967296.0f);

    float ao = 0.0f;
    float radius2 = pc.radius * pc.radius;        // Squared radius for fast 3D distance checks
    float thickness_threshold = pc.radius * 0.5f; // Heuristic to reject foreground objects

    // 2. Integration loop
    for (uint slice = 0; slice < pc.sliceCount; slice++)
    {
        float slice_angle = (float(slice) / float(pc.sliceCount)) * PI + angle_base;
        float2 dir_ss = float2(cos(slice_angle), sin(slice_angle));

        // 3D direction of the slice plane in view space
        float3 dir_view = normalize(float3(
            dir_ss.x * inv_tan_fov,
            -dir_ss.y * abs(pc.invProjection[1][1]),
            0.0f));

        // Project normal onto the slice plane
        float3 plane_n = cross(V, dir_view);
        float3 N_proj = N - dot(N, plane_n) * plane_n;
        float n_len = length(N_proj);

        // Calculate absolute angle of the projected normal relative to V
        float sign_n = sign(dot(N_proj, dir_view));
        float cos_n = clamp(dot(N_proj, V) / max(n_len, 1e-4f), -1.0f, 1.0f);
        float n_gamma = sign_n * acos(cos_n);

        // Track max cosines (starts at -1.0, which means 180 degrees away from V)
        float cos_h1 = -1.0f;
        float cos_h2 = -1.0f;

        // Raymarching
        for (uint step = 1; step <= pc.stepCount; step++)
        {
            float dist_ss = (float(step - 1u) + step_noise) * step_size;

            // --- Positive Direction (S1) ---
            float3 S1 = sampleViewPos(float2(px) + dir_ss * dist_ss);
            float3 d1 = S1 - P;
            float dist2_1 = dot(d1, d1);

            // CORRECTION 1 : Attenuation based on actual 3D distance instead of pixels
            float w1 = saturate(1.0f - dist2_1 / radius2);
            w1 = pow(w1, pc.falloff);

            // CORRECTION 2 : Strict rejection of foreground objects
            // If sample is closer to camera than P by distance > thickness, ignore it.
            if ((S1.z - P.z) > thickness_threshold)
                w1 = 0.0f;

            float len_d1 = sqrt(dist2_1);
            float c1 = dot(d1, V) / max(len_d1, 1e-4f);
            cos_h1 = max(cos_h1, lerp(cos_h1, c1, w1));

            // --- Negative Direction (S2) ---
            float3 S2 = sampleViewPos(float2(px) - dir_ss * dist_ss);
            float3 d2 = S2 - P;
            float dist2_2 = dot(d2, d2);

            float w2 = saturate(1.0f - dist2_2 / radius2);
            w2 = pow(w2, pc.falloff);

            if ((S2.z - P.z) > thickness_threshold)
                w2 = 0.0f;

            float len_d2 = sqrt(dist2_2);
            float c2 = dot(d2, V) / max(len_d2, 1e-4f);
            cos_h2 = max(cos_h2, lerp(cos_h2, c2, w2));
        }

        // Reconvert to angles relative to V
        float h1 = acos(clamp(cos_h1, -1.0f, 1.0f));
        float h2 = -acos(clamp(cos_h2, -1.0f, 1.0f)); // Negative side of the slice

        // Clamp horizons to the tangent plane to prevent occlusion from behind the surface
        h1 = n_gamma + max(h1 - n_gamma, -HALF_PI);
        h2 = n_gamma + min(h2 - n_gamma, HALF_PI);

        // Analytical integration over the arc
        float term1 = sin(h1) - sin(h2);
        float term2 = cos(h1) - cos(h2);
        float ao_slice = n_len * 0.25f * (term1 * cos(n_gamma) + term2 * sin(n_gamma));

        ao += saturate(ao_slice);
    }

    ao /= float(pc.sliceCount);

    // Fade out AO when screen-space radius gets too small (avoids aliasing on distant objects)
    // float ao_fade = smoothstep(2.0f, float(pc.stepCount) * 3.0f, ss_radius);
    aoOutput[int2(px)] = saturate(ao);
}
