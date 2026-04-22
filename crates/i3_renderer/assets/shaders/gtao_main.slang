// Ground-Truth Ambient Occlusion — Main compute pass
// Based on: "Practical Real-Time Strategies for Accurate Indirect Occlusion" (Jimenez 2016)
//
// Algorithm: horizon-based integral.
// For each pixel, in S azimuthal directions:
//   1. March T steps in +dir and -dir in screen-space
//   2. Track the maximum horizon angle h1 (+) and h2 (-) in view-space
//   3. Compute gamma = projected normal angle in the slice plane
//   4. Integrate visibility over the two horizon arcs
// AO = average(AO_slice) over all slices
//

#include "common.slangh"
#include "noise.slangh"

static const float HALF_PI = 1.5707963267948966f;

struct GtaoPushConstants
{
    float4x4 invProjection;
    float4x4 view;
    float2 screenSize;
    float radius;          // world-space sampling radius (metres)
    float falloff;         // distance falloff power (e.g. 2.0)
    uint sliceCount;       // azimuthal directions (2 = perf, 3 = quality)
    uint stepCount;        // steps per direction (4–6)
    uint frameIndex;       // for golden-ratio temporal rotation
    uint enabled;          // 0 = write 1.0 (disabled), 1 = run GTAO
    uint blue_noise_index; // bindless index of the 128×128 RG8 blue-noise texture
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

float3 viewSpacePos(uint2 px, float depth)
{
    float2 uv = (float2(px) + 0.5f) / pc.screenSize;
    // Vulkan viewport: Y-down in screen space, Y-up in NDC/view space → flip Y
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float4 clip = float4(ndc, depth, 1.0f);
    float4 vh = mul(pc.invProjection, clip);
    return vh.xyz / vh.w;
}

float3 sampleViewPos(float2 screenPos)
{
    // round() is critical: uint2(float) truncates toward zero (floor for positives).
    // floor(px + 1.06) = px+1  but  floor(px - 1.06) = px-2, not px-1.
    // That asymmetry makes +dir and -dir sample different distances from center.
    // round() restores symmetry: both land on the nearest pixel. (cf. XeGTAO)
    int2 ipx = int2(round(clamp(screenPos, float2(0.0f), float2(pc.screenSize) - 1.0f)));
    return viewSpacePos(uint2(ipx), depthBuffer[ipx].r);
}

// ─── Elevation angle ──────────────────────────────────────────────────────────
// Elevation of P→S above the plane perpendicular to V.
// Using length(d - dot(d,V)*V) as the horizontal distance (component perp to V)
// is exact for any screen position — unlike length(d.xy) which approximates V≈(0,0,1).

float elevationAngle(float3 P, float3 S, float3 V)
{
    float3 d = S - P;
    float dv = dot(d, V);
    float dh2 = max(dot(d, d) - dv * dv, 0.0f); // |d|² - (d·V)²
    float dh = sqrt(dh2);
    if (dh < 1e-5f)
        return dv > 0.0f ? HALF_PI : -HALF_PI;
    return atan2(dv, dh);
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
    if (depth <= 0.0f)
    { // reverse-Z: 0 = sky / far plane
        aoOutput[int2(px)] = 1.0f;
        return;
    }

    // View-space position at this pixel
    float3 P = viewSpacePos(px, depth);

    // View-space normal — decode world-space octahedral, rotate into view
    float2 nEnc = gbufferNormal[int2(px)].rg;
    float3 N_world = decodeOctahedral(nEnc);
    float3x3 viewRot = float3x3(pc.view[0].xyz, pc.view[1].xyz, pc.view[2].xyz);
    float3 N = normalize(mul(viewRot, N_world));

    // Screen-space radius of the sampling kernel (in pixels)
    // invProjection[0][0] = tan(fovX/2); projecting world radius to pixels:
    //   ss_radius = world_radius / (tan(fovX/2) * depth) * (screen_width/2)
    float inv_tan_fov = abs(pc.invProjection[0][0]);
    float ss_radius = clamp(
        pc.radius * inv_tan_fov * (pc.screenSize.x * 0.5f) / max(-P.z, 0.01f),
        2.0f, 256.0f);
    float step_size = ss_radius / float(pc.stepCount + 1);

    // View direction (toward camera from P)
    float3 V = normalize(-P);

    // get noise sample depends on position + frameIndex
    uint3 h = pcg3(uint3(px, pc.frameIndex));
    float angle_base = float(h.x) * (1.0f / 4294967296.0f);
    float step_noise = float(h.y) * (1.0f / 4294967296.0f);

    // World size of one pixel at P's depth — used for per-step z discontinuity threshold.
    float pixel_world = max(-P.z, 0.01f) * 2.0f * inv_tan_fov / pc.screenSize.x;

    float ao = 0.0f;

    for (uint slice = 0; slice < pc.sliceCount; slice++)
    {
        float slice_angle = (float(slice) / float(pc.sliceCount)) * PI + angle_base;
        float2 dir_ss = float2(cos(slice_angle), sin(slice_angle));

        // Approximate view-space 3D slice direction (in XY plane of view space)
        float3 dir_view = normalize(float3(
            dir_ss.x * abs(pc.invProjection[0][0]),
            -dir_ss.y * abs(pc.invProjection[1][1]), // screen Y-down → view Y-flip
            0.0f));

        // Project N onto the slice plane: remove component along plane normal
        float3 plane_n = normalize(cross(dir_view, V));
        float3 N_proj = N - dot(N, plane_n) * plane_n;
        float n_len = length(N_proj);

        // Normalized version (safe — fallback to dir_view if degenerate)
        float3 N_proj_n = (n_len > 1e-4f) ? N_proj / n_len : dir_view;

        // ── Approximated integral (Jimenez 2016 simplified) ───────────────────
        //
        // Angle convention: elevation from XY-horizontal plane.
        //   gamma = 0   → N_proj along dir_view (sideways)
        //   gamma = π/2 → N_proj along V (facing camera)
        //
        // The "no occlusion" level for both arcs is (gamma - π/2).
        // Falloff lerps toward this level (not toward 0) to avoid directional bias.

        // asin(dot(N_proj_n, V)) gives elevation of N from the plane perp to V — always in
        // [-π/2, π/2].  atan2(dot(N,V), dot(N,dir_view)) would return ±π for faces whose
        // normal points in -dir_view, breaking the formula for those surfaces.
        float gamma = asin(clamp(dot(N_proj_n, V), -1.0f, 1.0f));

        // Initialize horizons to the base of the visible hemisphere.
        // cos(h - gamma) = cos(-π/2) = 0 → no AO contribution with no occluders.
        float h1 = gamma - HALF_PI;
        float h2 = gamma - HALF_PI;
        float h_neutral = gamma - HALF_PI;

        for (uint step = 1; step <= pc.stepCount; step++)
        {
            float dist_ss = (float(step - 1u) + step_noise) * step_size;
            float t = saturate(dist_ss / ss_radius);
            float w = 1.0f - pow(t, pc.falloff);
            if (w < 1e-3f)
                break;

            // z discontinuity threshold: scales with step world-distance.
            // tan(72°) ≈ 3 → allows surfaces up to 72° without suppression.
            // True edges (perpendicular geometry) have z_diff >> world_step → z_w ≈ 0.
            float z_threshold = max(dist_ss * pixel_world * 3.0f, 0.001f);

            float3 S_pos = sampleViewPos(float2(px) + dir_ss * dist_ss);
            float h = elevationAngle(P, S_pos, V);
            float z_w = saturate(1.0f - max(0.0f, S_pos.z - P.z) / z_threshold);
            h1 = max(h1, lerp(h_neutral, h, w * z_w));

            float3 S_neg = sampleViewPos(float2(px) - dir_ss * dist_ss);
            h = elevationAngle(P, S_neg, V);
            z_w = saturate(1.0f - max(0.0f, S_neg.z - P.z) / z_threshold);
            h2 = max(h2, lerp(h_neutral, h, w * z_w));
        }

        // Clamp both horizons to the visible hemisphere [gamma − π/2, gamma + π/2]
        h1 = gamma + clamp(h1 - gamma, -HALF_PI, HALF_PI);
        h2 = gamma + clamp(h2 - gamma, -HALF_PI, HALF_PI);

        // 1 − 0.5·(cos(h1−γ) + cos(h2−γ))
        float ao_slice = 1.0f - 0.5f * (cos(h1 - gamma) + cos(gamma - h2));
        ao += saturate(ao_slice);
    }

    ao /= float(pc.sliceCount);

    // Fade to no-AO as ss_radius shrinks toward its minimum clamp.
    // step_size = ss_radius/(stepCount+1); below ~stepCount px every step
    // rounds to the same pixel → constant output.  Use a wide smoothstep so
    // the transition is invisible rather than a sharp band at a specific depth.
    float ao_fade = smoothstep(2.0f, float(pc.stepCount) * 3.0f, ss_radius);
    aoOutput[int2(px)] = lerp(1.0f, saturate(ao), ao_fade);
}
