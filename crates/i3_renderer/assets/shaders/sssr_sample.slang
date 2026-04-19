// Stochastic Screen-Space Reflections — Sample pass
//
// One ray per pixel per frame, direction importance-sampled from GGX.
//
// Algorithm:
//   1. Build a TBN frame from the view-space normal.
//   2. Draw two uniform samples (IGN + golden-ratio rotation per frame).
//   3. Map samples to a half-vector H in the GGX distribution.
//      For roughness=0 H degenerates to N → deterministic mirror ray.
//   4. Reflect -V around H to get the reflection direction R.
//   5. Frustum-clip the ray endpoint if R points toward the camera.
//   6. March linearly in screen space (max_steps equal steps + 8-step binary
//      search at the first intersection found).
//   7. Reject hits closer than 4 px (self-intersection guard) or deeper
//      than pc.thickness in view-space.
//
// Output: float4(hit_color, confidence)
//   - hit_color  : HDR colour at the intersection point.
//   - confidence : edge_fade × (1 − hit_t²) ∈ [0, 1].
//   The composite pass applies the BRDF weight (split-sum factor) that
//   matches the IBL term computed by deferred_resolve.

#include "common.slangh"

struct SssrSamplePc {
    float4x4 inv_projection;    // view-space position reconstruction
    float4x4 projection;        // viewToClip
    float4x4 view;              // world normal → view normal
    float2   screen_size;
    float    max_distance;
    float    thickness;         // max view-space depth gap for hit acceptance
    float    roughness_cutoff;
    uint     max_steps;
    uint     frame_index;
    uint     enabled;
    uint     _pad;
};
[[vk::push_constant]] SssrSamplePc pc;

[[vk::binding(0, 0)]] Texture2D<float>    depthBuffer;
[[vk::binding(1, 0)]] Texture2D<float2>   gbufferNormal;
[[vk::binding(2, 0)]] Texture2D<float2>   gbufferRoughMetal;
[[vk::binding(3, 0)]] Texture2D<float4>   hdrTarget;
[[vk::binding(4, 0)]] RWTexture2D<float4> ssrOutput;

// ─── Helpers ─────────────────────────────────────────────────────────────────

float3 viewSpacePos(uint2 px, float depth) {
    float2 uv  = (float2(px) + 0.5f) / pc.screen_size;
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float4 vh  = mul(pc.inv_projection, float4(ndc, depth, 1.0f));
    return vh.xyz / vh.w;
}

float4 viewToClip(float3 P) {
    return mul(pc.projection, float4(P, 1.0f));
}

// Y-flip: uv.y = 0.5 - ndc.y * 0.5  (negative-viewport, §5 engine_conventions.md).
float4 clipToScreenFull(float4 clip) {
    float2 ndc = clip.xy / clip.w;
    return float4(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f, clip.z, clip.w);
}

float linearZ(float ndc_d) {
    float4 v = mul(pc.inv_projection, float4(0.0f, 0.0f, ndc_d, 1.0f));
    return v.z / v.w;
}

// PCG hash — no linear isocontours, fully independent samples.
// IGN was replaced because its linear isocontours (bands of ~15 px) caused
// all pixels on a band to sample the same azimuthal angle → hatching.
uint pcgHash(uint v) {
    uint s = v * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float toF01(uint h) { return float(h) * (1.0f / 4294967296.0f); }

// Three independent noise values per pixel per frame.
// Golden-ratio offsets shift each dimension across frames (low-discrepancy).
void pixelNoise(uint2 px, uint frame, out float u1, out float u2, out float jitter) {
    uint seed = pcgHash(px.x ^ pcgHash(px.y ^ pcgHash(frame)));
    u1      = frac(toF01(seed)               + float(frame) * 0.618033988f);
    seed    = pcgHash(seed);
    u2      = frac(toF01(seed)               + float(frame) * 0.381966011f);
    seed    = pcgHash(seed);
    jitter  = frac(toF01(seed)               + float(frame) * 0.149161f);
}

float rayDepth(float cz_s, float cw_s, float cz_e, float cw_e, float t) {
    return lerp(cz_s, cz_e, t) / lerp(cw_s, cw_e, t);
}

// ─── GGX importance sampling ─────────────────────────────────────────────────
//
// Given two uniform samples u1, u2 ∈ [0,1) and GGX roughness α = roughness²,
// returns a half-vector H in the local tangent frame (Z = up).
// PDF(H) = α² * cosθ / (π * ((cosθ² * (α²−1) + 1)²))
//
// For α→0 the formula degenerates; clamp to 1e-3 → deterministic mirror.

float3 sampleGGX_H(float u1, float u2, float alpha) {
    float a2   = alpha * alpha;
    float phi  = 2.0f * PI * u1;
    float denom = max(1.0f + (a2 - 1.0f) * u2, 1e-6f);
    float cos2  = (1.0f - u2) / denom;
    float cosT  = sqrt(max(cos2, 0.0f));
    float sinT  = sqrt(max(1.0f - cos2, 0.0f));
    return float3(sinT * cos(phi), sinT * sin(phi), cosT);
}

// Build an orthonormal TBN with Z = N (view-space).
void buildTBN(float3 N, out float3 T, out float3 B) {
    float3 up = abs(N.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

// ─── Main ────────────────────────────────────────────────────────────────────

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID) {
    uint2 px = id.xy;
    if (any(px >= uint2(pc.screen_size))) return;

    if (pc.enabled == 0u) { ssrOutput[int2(px)] = 0; return; }

    float depth = depthBuffer.Load(int3(px, 0)).r;
    if (depth <= 0.0f) { ssrOutput[int2(px)] = 0; return; }

    float roughness = gbufferRoughMetal.Load(int3(px, 0)).r;
    if (roughness > pc.roughness_cutoff) { ssrOutput[int2(px)] = 0; return; }

    // ── Geometry ──────────────────────────────────────────────────────────────

    float3 P       = viewSpacePos(px, depth);
    float3 N_world = decodeOctahedral(gbufferNormal.Load(int3(px, 0)).rg);
    float3x3 viewRot = float3x3(pc.view[0].xyz, pc.view[1].xyz, pc.view[2].xyz);
    float3 N = normalize(mul(viewRot, N_world));
    float3 V = normalize(-P);

    // ── GGX importance-sample the reflection direction ─────────────────────────
    //
    // Two independent noise samples, rotated each frame.
    // u1 controls the azimuthal angle φ; u2 controls the polar angle θ.
    // A golden-ratio offset on u2 breaks temporal coherence between frames
    // without destroying the spatial stratification of IGN on u1.

    float alpha = max(roughness * roughness, 1e-3f);

    float u1, u2, marchNoise;
    pixelNoise(px, pc.frame_index, u1, u2, marchNoise);

    float3 H_local = sampleGGX_H(u1, u2, alpha);

    float3 T, B;
    buildTBN(N, T, B);
    float3 H = normalize(T * H_local.x + B * H_local.y + N * H_local.z);

    // Guard: if H is below the surface, flip to mirror side.
    if (dot(H, N) < 0.0f) H = reflect(H, N);

    float3 R = reflect(-V, H);

    // Reject rays going into the surface (possible for very rough materials).
    if (dot(R, N) <= 0.0f) { ssrOutput[int2(px)] = 0; return; }

    // ── Ray endpoints ─────────────────────────────────────────────────────────

    float3 P_start = P + N * 0.05f;
    float3 P_end   = P + R * pc.max_distance;

    float4 clip_s = viewToClip(P_start);
    float4 clip_e = viewToClip(P_end);

    if (clip_s.w <= 0.0f) { ssrOutput[int2(px)] = 0; return; }

    // Frustum clip: if the endpoint is behind the camera, clip to near plane.
    if (clip_e.w <= 0.0f) {
        float t_near = clip_s.w / (clip_s.w - clip_e.w) * 0.99f;
        if (t_near < 0.02f) { ssrOutput[int2(px)] = 0; return; }
        P_end  = lerp(P_start, P_end, t_near);
        clip_e = viewToClip(P_end);
        if (clip_e.w <= 0.0f) { ssrOutput[int2(px)] = 0; return; }
    }

    float4 ss_s = clipToScreenFull(clip_s);
    float4 ss_e = clipToScreenFull(clip_e);

    float2 uv_start = ss_s.xy;
    float2 uv_delta = ss_e.xy - ss_s.xy;

    float cz_s = ss_s.z, cw_s = ss_s.w;
    float cz_e = ss_e.z, cw_e = ss_e.w;

    float start_ndc = cz_s / cw_s;
    if (start_ndc < 0.0f || start_ndc > 1.0f) { ssrOutput[int2(px)] = 0; return; }

    float ray_px = max(abs(uv_delta.x) * pc.screen_size.x,
                       abs(uv_delta.y) * pc.screen_size.y);
    if (ray_px < 1.0f) { ssrOutput[int2(px)] = 0; return; }

    // ── Phase 1: linear march ─────────────────────────────────────────────────
    //
    // Fixed step count — no HiZ needed for SSSR (temporal denoises noise).
    // dt = 1/max_steps in t-space; jitter within the first step to avoid
    // repeating the same pattern every frame.

    float dt = 1.0f / float(pc.max_steps);
    float t  = marchNoise * dt;
    float t_prev = 0.0f;
    bool  bracketed = false;

    [loop]
    for (uint step = 0u; step < pc.max_steps; ++step) {
        float2 uv = uv_start + uv_delta * t;
        if (any(uv < 0.0f) || any(uv > 1.0f) || t >= 1.0f) break;

        float ray_d = rayDepth(cz_s, cw_s, cz_e, cw_e, t);
        float buf_d = depthBuffer.SampleLevel(bindlessSamplers[SAMPLER_NEAREST_BORDER], uv, 0.0f).r;

        // Reverse-Z: ray_d > buf_d → ray is closer to camera (in front).
        if (ray_d > buf_d) {
            t_prev = t;
            t     += dt;
        } else {
            // Ray crossed behind the surface — bracket found.
            bracketed = true;
            break;
        }
    }

    if (!bracketed) { ssrOutput[int2(px)] = 0; return; }

    // ── Phase 2: binary search within [t_prev, t] ────────────────────────────

    float t_lo = t_prev;
    float t_hi = t;

    [loop]
    for (uint b = 0u; b < 8u; ++b) {
        float  t_mid  = (t_lo + t_hi) * 0.5f;
        float  rd_mid = rayDepth(cz_s, cw_s, cz_e, cw_e, t_mid);
        float2 uv_mid = uv_start + uv_delta * t_mid;
        float  bd_mid = depthBuffer.SampleLevel(bindlessSamplers[SAMPLER_NEAREST_BORDER], uv_mid, 0.0f).r;
        if (rd_mid > bd_mid) t_lo = t_mid;
        else                  t_hi = t_mid;
    }

    float2 hit_uv = uv_start + uv_delta * t_hi;
    float  hit_rd = rayDepth(cz_s, cw_s, cz_e, cw_e, t_hi);
    float  hit_bd = depthBuffer.SampleLevel(bindlessSamplers[SAMPLER_NEAREST_BORDER], hit_uv, 0.0f).r;

    // Reject: depth gap too large (false intersection) or too close to origin
    // (residual self-intersection from the starting surface).
    if (abs(linearZ(hit_rd) - linearZ(hit_bd)) >= pc.thickness) { ssrOutput[int2(px)] = 0; return; }
    if (t_hi < 4.0f / ray_px)                                    { ssrOutput[int2(px)] = 0; return; }

    // ── Sample reflected colour ───────────────────────────────────────────────

    // The composite will apply the split-sum BRDF weight; store raw colour here.
    float3 color = hdrTarget.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], hit_uv, 0.0f).rgb;

    // ── Confidence ────────────────────────────────────────────────────────────

    float2 edge_fade = smoothstep(0.0f, 0.05f, hit_uv)
                     * smoothstep(1.0f, 0.95f, hit_uv);
    float  conf = saturate(edge_fade.x * edge_fade.y * (1.0f - t_hi * t_hi));

    ssrOutput[int2(px)] = float4(color, conf);
}
