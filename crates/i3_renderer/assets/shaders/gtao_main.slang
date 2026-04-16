// Ground-Truth Ambient Occlusion — Main compute pass
// Based on: "Practical Real-Time Strategies for Accurate Indirect Occlusion" (Jimenez 2016)
//
// Algorithm: horizon-based integral.
// For each pixel, in S azimuthal directions:
//   1. March T steps in +dir and -dir in screen-space
//   2. Track the maximum horizon angle h1 (+) and h2 (-) in view-space
//   3. Compute gamma = projected normal angle in the slice plane
//   4. AO_slice = 1 - 0.5*(cos(h1 - gamma) + cos(h2 - gamma))
// AO = average(AO_slice) over all slices

#include "common.slangh"

static const float HALF_PI = 1.5707963267948966f;

struct GtaoPushConstants {
    float4x4 invProjection;
    float4x4 view;
    float2   screenSize;
    float    radius;      // world-space sampling radius (metres)
    float    falloff;     // distance falloff power (e.g. 2.0)
    uint     sliceCount;  // azimuthal directions (2 = perf, 3 = quality)
    uint     stepCount;   // steps per direction (4–6)
    uint     frameIndex;  // for golden-ratio temporal rotation
    uint     enabled;     // 0 = write 1.0 (disabled), 1 = run GTAO
};

[[vk::push_constant]]
GtaoPushConstants pc;

[[vk::binding(0, 0)]] Texture2D<float>   depthBuffer;
[[vk::binding(1, 0)]] Texture2D<float2>  gbufferNormal;
[[vk::binding(2, 0)]] RWTexture2D<float> aoOutput;

// ─── Position reconstruction ──────────────────────────────────────────────────

float3 viewSpacePos(uint2 px, float depth) {
    float2 uv  = (float2(px) + 0.5f) / pc.screenSize;
    // Vulkan viewport: Y-down in screen space, Y-up in NDC/view space → flip Y
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float4 clip = float4(ndc, depth, 1.0f);
    float4 vh   = mul(pc.invProjection, clip);
    return vh.xyz / vh.w;
}

float3 sampleViewPos(float2 screenPos) {
    uint2 px = uint2(clamp(screenPos, float2(0.5f), pc.screenSize - 0.5f));
    return viewSpacePos(px, depthBuffer[int2(px)].r);
}

// ─── Horizon angle in view-space ──────────────────────────────────────────────
// Returns the elevation angle from the horizontal plane through P to the P→S vector.
// Positive = S is "above" P (closer to camera along Z).
// In view-space, camera is at origin, scene is at negative Z, so +delta.z = toward camera.

float elevationAngle(float3 P, float3 S) {
    float3 d  = S - P;
    float hz  = length(d.xy);
    if (hz < 1e-5f) return -HALF_PI;
    return atan2(d.z, hz);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

[numthreads(8, 8, 1)]
void computeMain(uint3 dispatchId: SV_DispatchThreadID) {
    uint2 px = dispatchId.xy;
    if (any(px >= uint2(pc.screenSize))) return;

    if (pc.enabled == 0) {
        aoOutput[int2(px)] = 1.0f;
        return;
    }

    float depth = depthBuffer[int2(px)].r;
    if (depth <= 0.0f) {               // reverse-Z: 0 = sky / far plane
        aoOutput[int2(px)] = 1.0f;
        return;
    }

    // View-space position at this pixel
    float3 P = viewSpacePos(px, depth);

    // View-space normal — decode world-space octahedral, rotate into view
    float2 nEnc   = gbufferNormal[int2(px)].rg;
    float3 N_world = decodeOctahedral(nEnc);
    float3x3 viewRot = float3x3(pc.view[0].xyz, pc.view[1].xyz, pc.view[2].xyz);
    float3 N = normalize(mul(viewRot, N_world));

    // Screen-space radius of the sampling kernel (in pixels)
    // invProjection[0][0] ≈ 1 / tan(fovX/2)  →  focal-length-like quantity
    float inv_tan_fov = abs(pc.invProjection[0][0]);
    float ss_radius   = clamp(
        pc.radius * inv_tan_fov * (pc.screenSize.x * 0.5f) / max(-P.z, 0.01f),
        2.0f, 256.0f
    );
    float step_size = ss_radius / float(pc.stepCount + 1);

    // View direction (toward camera from P)
    float3 V = normalize(-P);

    // Temporal angle rotation via golden ratio
    float angle_offset = frac(float(pc.frameIndex) * 0.6180339887f) * PI;

    float ao = 0.0f;

    for (uint slice = 0; slice < pc.sliceCount; slice++) {
        float slice_angle = (float(slice) / float(pc.sliceCount)) * PI + angle_offset;
        float2 dir_ss = float2(cos(slice_angle), sin(slice_angle));

        // Approximate view-space 3D slice direction (in XY plane of view space)
        float3 dir_view = normalize(float3(
            dir_ss.x * abs(pc.invProjection[0][0]),
           -dir_ss.y * abs(pc.invProjection[1][1]),  // screen Y-down → view Y-flip
            0.0f
        ));

        // Project N onto the slice plane: remove the component along the plane normal
        float3 plane_n = normalize(cross(dir_view, V));
        float3 N_proj  = N - dot(N, plane_n) * plane_n;
        float  n_len   = length(N_proj);

        // gamma = angle of projected normal in the slice plane from horizontal
        float gamma;
        if (n_len < 1e-4f) {
            gamma = 0.0f;
        } else {
            N_proj /= n_len;
            // atan2(component-along-V, component-along-dir)
            gamma = atan2(dot(N_proj, V), dot(N_proj, dir_view));
        }

        // Find max horizon angles in +dir and -dir
        float h1 = -HALF_PI;  // horizon in +dir
        float h2 = -HALF_PI;  // horizon in -dir

        for (uint step = 1; step <= pc.stepCount; step++) {
            float dist_ss = float(step) * step_size;

            // Distance falloff weight [1 → 0] over the sampling radius
            float t = saturate(dist_ss / ss_radius);
            float w = 1.0f - pow(t, pc.falloff);
            if (w < 1e-3f) break;

            float3 S_pos = sampleViewPos(float2(px) + dir_ss * dist_ss);
            float  h     = elevationAngle(P, S_pos);
            h1 = max(h1, h * w);

            float3 S_neg = sampleViewPos(float2(px) - dir_ss * dist_ss);
            h = elevationAngle(P, S_neg);
            h2 = max(h2, h * w);
        }

        // Clamp to the visible hemisphere defined by the projected normal
        h1 = gamma + min(h1 - gamma,  HALF_PI);
        h2 = gamma + min(h2 - gamma, -HALF_PI);

        // Jimenez 2016: AO_slice = 1 - 0.5*(cos(h1 - gamma) + cos(h2 - gamma))
        float ao_slice = 1.0f - 0.5f * (cos(h1 - gamma) + cos(-h2 + gamma));
        ao += saturate(ao_slice);
    }

    ao /= float(pc.sliceCount);
    aoOutput[int2(px)] = saturate(ao);
}
