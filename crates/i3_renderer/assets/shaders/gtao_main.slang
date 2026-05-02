// Ground-Truth Ambient Occlusion — Main compute pass
// Algorithm follows XeGTAO (Intel, Jimenez et al.) closely:
//   - Per-slice horizon initialized to N-tangent plane (not V-tangent)
//   - Correct arc integral: (cosN + 2h·sin(n) - cos(2h-n)) / 4
//   - Final power curve for contrast control

#include "common.slangh"
#include "sampling.slangh"

struct GtaoPushConstants
{
    float4x4 invProjection;
    float4x4 view;
    float2 screenSize;
    float radius;       // World-space sampling radius (metres)
    float finalPower;   // Final visibility power curve (e.g. 2.0); was "falloff"
    uint sliceCount;    // Azimuthal directions (2 = perf, 3 = quality)
    uint stepCount;     // Steps per direction (4–6)
    uint frameIndex;    // For temporal dithering
    uint enabled;       // 0 = write 1.0 (disabled), 1 = run GTAO
    uint blue_noise_index;
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
    float3 V = normalize(-P);

    // Decode and transform normal to view space
    float2 nEnc = gbufferNormal[int2(px)].rg;
    float3 N_world = decodeOctahedral(nEnc);
    float3x3 viewRot = float3x3(pc.view[0].xyz, pc.view[1].xyz, pc.view[2].xyz);
    float3 N = normalize(mul(viewRot, N_world));

    // Screen-space radius: invProjection[0][0] = tan(fovx/2)
    float tan_half_fov = abs(pc.invProjection[0][0]);
    float ss_radius = clamp(
        pc.radius / tan_half_fov * (pc.screenSize.x * 0.5f) / max(-P.z, 0.01f),
        2.0f, 256.0f);
    float step_size = ss_radius / float(pc.stepCount + 1);

    // Small base visibility for tiny screen-space radii (avoids aliasing at distance)
    float ao = saturate((10.0f - ss_radius) / 100.0f) * 0.5f;

    // Linear distance falloff matching XeGTAO: full weight below 38.5% radius, zero at radius
    float falloff_mul = -1.0f / (pc.radius * 0.615f);
    float falloff_add =  1.0f + (0.385f / 0.615f);   // = falloffFrom/falloffRange + 1

    // Thickness heuristic: ignore samples that are in front of P by more than this
    float thickness_threshold = pc.radius * 0.5f;

    // Dither
    uint3 rng = pcg3(uint3(px, pc.frameIndex));
    float angle_base = float(rng.x) * (1.0f / 4294967296.0f);
    float step_noise  = float(rng.y) * (1.0f / 4294967296.0f);

    // 2. Integration loop — one per azimuthal slice
    for (uint slice = 0; slice < pc.sliceCount; slice++)
    {
        float slice_angle = (float(slice) / float(pc.sliceCount)) * PI + angle_base;
        float2 dir_ss = float2(cos(slice_angle), sin(slice_angle));

        // 3D slice direction in view space (screen y is flipped)
        float3 dir_view = normalize(float3(
            dir_ss.x * tan_half_fov,
            -dir_ss.y * abs(pc.invProjection[1][1]),
            0.0f));

        // Project N onto the slice plane (remove axisVec component)
        float3 axis_vec = normalize(cross(dir_view - dot(dir_view, V) * V, V));
        float3 N_proj   = N - axis_vec * dot(N, axis_vec);
        float  n_len    = length(N_proj);

        // Projected normal angle relative to V (signed: positive toward dir_view)
        float sign_n  = sign(dot(N_proj, dir_view));
        float cos_n   = clamp(dot(N_proj, V) / max(n_len, 1e-4f), -1.0f, 1.0f);
        float n_gamma = sign_n * acos(cos_n);

        // XeGTAO-style horizon init: N-tangent plane, not V-tangent plane.
        // cos(n + π/2) = -sin(n), cos(n - π/2) = +sin(n).
        // Using these as the minimum horizon cosine means the integration region
        // starts at the plane tangent to N, correctly handling tilted surfaces.
        float low_h_pos = cos(n_gamma + HALF_PI); // = -sin(n_gamma)
        float low_h_neg = cos(n_gamma - HALF_PI); // = +sin(n_gamma)
        float cos_h_pos = low_h_pos;
        float cos_h_neg = low_h_neg;

        // 3. Horizon march — positive and negative directions simultaneously
        for (uint step = 1; step <= pc.stepCount; step++)
        {
            float dist_ss = (float(step - 1u) + step_noise) * step_size;

            // --- Positive direction ---
            float3 S1   = sampleViewPos(float2(px) + dir_ss * dist_ss);
            float3 d1   = S1 - P;
            float  len1 = length(d1);
            float  w1   = saturate(len1 * falloff_mul + falloff_add);
            if ((S1.z - P.z) > thickness_threshold) w1 = 0.0f;
            float  c1   = dot(d1 / max(len1, 1e-4f), V);
            // lerp to low_h_pos when out-of-radius: sample has no effect on horizon
            cos_h_pos = max(cos_h_pos, lerp(low_h_pos, c1, w1));

            // --- Negative direction ---
            float3 S2   = sampleViewPos(float2(px) - dir_ss * dist_ss);
            float3 d2   = S2 - P;
            float  len2 = length(d2);
            float  w2   = saturate(len2 * falloff_mul + falloff_add);
            if ((S2.z - P.z) > thickness_threshold) w2 = 0.0f;
            float  c2   = dot(d2 / max(len2, 1e-4f), V);
            cos_h_neg = max(cos_h_neg, lerp(low_h_neg, c2, w2));
        }

        // 4. Arc integral (XeGTAO Algorithm 1, lines 27–28)
        // Antiderivative of sin(h)·cos(h-n) w.r.t. h is (cosN + 2h·sin(n) - cos(2h-n)) / 4
        float h_pos =  acos(clamp(cos_h_pos, -1.0f, 1.0f)); // positive side angle
        float h_neg = -acos(clamp(cos_h_neg, -1.0f, 1.0f)); // negative side angle
        float sin_n = sin(n_gamma);
        float iarc_pos = (cos_n + 2.0f * h_pos * sin_n - cos(2.0f * h_pos - n_gamma)) * 0.25f;
        float iarc_neg = (cos_n + 2.0f * h_neg * sin_n - cos(2.0f * h_neg - n_gamma)) * 0.25f;

        // Slight inflation of n_len toward 1 reduces overdarkening on steep slopes (XeGTAO fudge)
        n_len = lerp(n_len, 1.0f, 0.05f);

        ao += n_len * (iarc_pos + iarc_neg);
    }

    ao /= float(pc.sliceCount);

    // Final power curve tightens AO contrast; max(0.03) avoids total black (XeGTAO convention)
    ao = pow(saturate(ao), pc.finalPower);
    ao = max(0.03f, ao);

    aoOutput[int2(px)] = ao;
}
