#include "common.slangh"

// ─── Clipmap storage buffers (Set 0) — all 4 levels concatenated ──────────────
[[vk::binding(0, 0)]]
StructuredBuffer<uint> page_table;
[[vk::binding(1, 0)]]
StructuredBuffer<float> sdf_atlas; // world-space metres
[[vk::binding(2, 0)]]
StructuredBuffer<uint> mat_atlas;

// ─── Push constants ───────────────────────────────────────────────────────────
struct LevelPC
{
    float3 world_origin;
    float voxel_size;
};
struct ClipmapPC
{
    LevelPC levels[10];
    uint num_levels;
    uint debug_flags; // bit 0 = show level colours
    uint2 _pad;
};
[[vk::push_constant]]
ClipmapPC pc;

// ─── Constants ────────────────────────────────────────────────────────────────
static const uint BRICK_DIM = 8u;
static const uint BRICK_VOXELS = 729u; // 9³ — BRICK_DIM+1 per axis, 1-voxel positive overlap
static const uint GRID_X = 32u;
static const uint GRID_Y = 32u;
static const uint GRID_Z = 32u;
static const uint GRID_VOL = 32768u;
static const uint MAX_BRICKS = 8192u;
static const uint PAGE_EMPTY = 0xFFFFFFFFu;
// Fraction of each level's half-extent used for cross-level SDF blending.
// 0.25 = 25%, giving ~1.6 m of transition at L2, ~3.2 m at L3, etc.
static const float LEVEL_BLEND = 0.25;

// ─── Helpers ──────────────────────────────────────────────────────────────────

bool brickInGrid(int3 b)
{
    return all(b >= int3(0, 0, 0)) &&
           uint(b.x) < GRID_X && uint(b.y) < GRID_Y && uint(b.z) < GRID_Z;
}

uint brickFlatIdx(int3 b)
{
    return uint(b.x) + GRID_X * (uint(b.y) + GRID_Y * uint(b.z));
}

float3 safeRcp(float3 rd)
{
    return float3(
        abs(rd.x) > 1e-7 ? 1.0 / rd.x : (rd.x >= 0.0 ? 1e30 : -1e30),
        abs(rd.y) > 1e-7 ? 1.0 / rd.y : (rd.y >= 0.0 ? 1e30 : -1e30),
        abs(rd.z) > 1e-7 ? 1.0 / rd.z : (rd.z >= 0.0 ? 1e30 : -1e30));
}

// ─── Trilinear SDF + Smooth Gradient ─────────────────────────────────────────
// Returns the SDF value at p for the given level.
// The analytic gradient of a trilinear interpolant is piecewise-constant along
// its axes, causing blocky/faceted normals. To achieve genuinely smooth normals,
// we must use central finite differences on the continuous trilinear field itself.

float sampleSdfLevel(float3 p, uint lev)
{
    float3 org = pc.levels[lev].world_origin;
    float vs = pc.levels[lev].voxel_size;
    float bws = float(BRICK_DIM) * vs;

#if 0
    float3 gp = (p - org) / bws;
    int3   b  = int3(floor(gp));          // offset from origin ∈ [0, grid) for valid p
    if (!brickInGrid(b)) return 1e10;
#else
    // Shift the evaluation point by half a voxel to perfectly align the 0.5..8.5
    // valid interpolation range with the brick boundaries. This avoids negative
    // local positions that would be clamped to 0, causing flat normals at boundaries.
    float3 gp = (p - (org + 0.5 * vs)) / bws;
    int3 b = clamp(int3(floor(gp)), int3(0), int3(GRID_X - 1, GRID_Y - 1, GRID_Z - 1));

#endif

    // Toroidal page-table lookup.
    // b.x is the offset from the origin, so the world-brick is (ob + b).
    // Its cell in the page table is (ob + b) % grid.
    int3 ob = int3(round(org / bws)); // origin world-brick (integer)
    int3 grid = int3(GRID_X, GRID_Y, GRID_Z);
    int3 cell = ((ob + b) % grid + grid) % grid;

    uint page = page_table[lev * GRID_VOL + brickFlatIdx(cell)];
    if (page == PAGE_EMPTY)
        return 1e10;

    uint base = lev * MAX_BRICKS * BRICK_VOXELS + page * BRICK_VOXELS;
    // Atlas stride is BRICK_DIM+1 (9) — one extra voxel of positive overlap per axis.
    uint bd = BRICK_DIM + 1u;

    // Local position inside the brick, in voxel units.
    // Clamped to [0, BRICK_DIM-ε] so vi ∈ [0, BRICK_DIM-1] and vi+1 ∈ [1, BRICK_DIM].
    // The overlap voxel at index BRICK_DIM is the neighbouring brick's voxel 0,
    // letting the trilinear use real cross-brick data instead of clamped edge values.
    float3 lp = (p - (org + float3(b) * bws)) / vs - 0.5;
    lp = clamp(lp, float3(0.0), float3(float(BRICK_DIM) - 1e-4));
    int3 vi = int3(floor(lp));
    float3 fr = lp - float3(vi);
    vi = min(vi, int3(int(BRICK_DIM) - 1, int(BRICK_DIM) - 1, int(BRICK_DIM) - 1));

    // Fetch 8 corners (world-space SDF values in metres)
    float s000 = sdf_atlas[base + uint(vi.x) + bd * (uint(vi.y) + bd * uint(vi.z))];
    float s100 = sdf_atlas[base + uint(vi.x + 1) + bd * (uint(vi.y) + bd * uint(vi.z))];
    float s010 = sdf_atlas[base + uint(vi.x) + bd * (uint(vi.y + 1) + bd * uint(vi.z))];
    float s110 = sdf_atlas[base + uint(vi.x + 1) + bd * (uint(vi.y + 1) + bd * uint(vi.z))];
    float s001 = sdf_atlas[base + uint(vi.x) + bd * (uint(vi.y) + bd * uint(vi.z + 1))];
    float s101 = sdf_atlas[base + uint(vi.x + 1) + bd * (uint(vi.y) + bd * uint(vi.z + 1))];
    float s011 = sdf_atlas[base + uint(vi.x) + bd * (uint(vi.y + 1) + bd * uint(vi.z + 1))];
    float s111 = sdf_atlas[base + uint(vi.x + 1) + bd * (uint(vi.y + 1) + bd * uint(vi.z + 1))];

    float fx = fr.x, fy = fr.y, fz = fr.z;

    // Trilinear value
    return lerp(
        lerp(lerp(s000, s100, fx), lerp(s010, s110, fx), fy),
        lerp(lerp(s001, s101, fx), lerp(s011, s111, fx), fy),
        fz);
}

float sampleSdfLevelGrad(float3 p, uint lev, out float3 grad)
{
    float val = sampleSdfLevel(p, lev);
    
    // Smooth normal via Central Finite Differences.
    // An epsilon of 0.5 to 1.0 voxel crosses the C0 cell boundaries and smooths
    // the piecewise-constant trilinear derivative into a continuous gradient.
    float e = pc.levels[lev].voxel_size * 0.5;
    
    float dx = sampleSdfLevel(p + float3(e, 0, 0), lev) - sampleSdfLevel(p - float3(e, 0, 0), lev);
    float dy = sampleSdfLevel(p + float3(0, e, 0), lev) - sampleSdfLevel(p - float3(0, e, 0), lev);
    float dz = sampleSdfLevel(p + float3(0, 0, e), lev) - sampleSdfLevel(p - float3(0, 0, e), lev);
    
    grad = float3(dx, dy, dz) / (2.0 * e);
    return val;
}

// ─── Level sampling with transition blending ──────────────────────────────────
// Near each level's outer boundary (LEVEL_BLEND zone), lerps to the next coarser
// level so transitions are seamless instead of a hard cut.

float sampleFinest(float3 p, out uint lev_out)
{

    for (uint lev = 0u; lev < pc.num_levels; lev++)
    {
        float bws = float(BRICK_DIM) * pc.levels[lev].voxel_size;
        float3 org = pc.levels[lev].world_origin;
        float3 half_e = float3(GRID_X, GRID_Y, GRID_Z) * bws * 0.5;
        float3 ctr = org + half_e;
        float3 gmax = org + float3(GRID_X, GRID_Y, GRID_Z) * bws;
        if (!all(p >= org) || !all(p < gmax))
            continue;

        float d = sampleSdfLevel(p, lev);
        if (d >= 9e9)
            continue;

        lev_out = lev;

        if (lev + 1u < pc.num_levels)
        {
            float3 rel = abs(p - ctr) / half_e;
            float t = max(rel.x, max(rel.y, rel.z));
            float alpha = smoothstep(1.0 - LEVEL_BLEND, 1.0, t);
            if (alpha > 0.0)
            {
                float d_next = sampleSdfLevel(p, lev + 1u);
                if (d_next < 9e9)
                    d = lerp(d, d_next, alpha);
            }
        }
        return d;
    }
    lev_out = pc.num_levels;
    return 1e10;
}

// ─── Smooth normal (Söderström / JCGT 2022) ────────────────────────────────────
// Analytic gradient of the trilinear SDF interpolant — C0 within each brick.
// Mirrors sampleFinest's blend so the normal direction is consistent with the
// blended SDF that drove the sphere tracer.  Without the blend, the tracer can
// converge in the transition zone via the coarse level's SDF, but normalFinest
// would return the fine level's gradient (pointing toward a different surface),
// producing catastrophically wrong normals near level edges.

float3 normalFinest(float3 p, uint lev)
{
    if (lev >= pc.num_levels)
        return float3(0, 1, 0);
    float3 grad;
    float d = sampleSdfLevelGrad(p, lev, grad);
    if (d >= 9e9)
        return float3(0, 1, 0);

    // Blend gradient at the transition zone.
    // The blended distance is D = lerp(d, d_next, alpha), so the true gradient is
    //   ∇D = (1-α)∇d + α∇d_next + (∇α)(d_next - d)
    // The last term (∇α)(d_next - d) is non-zero whenever the two levels disagree
    // near the level boundary and must be included to avoid a shading discontinuity.
    if (lev + 1u < pc.num_levels)
    {
        float bws = float(BRICK_DIM) * pc.levels[lev].voxel_size;
        float3 org = pc.levels[lev].world_origin;
        float3 half_e = float3(GRID_X, GRID_Y, GRID_Z) * bws * 0.5;
        float3 ctr = org + half_e;
        float3 rel = abs(p - ctr) / half_e;
        float t = max(rel.x, max(rel.y, rel.z));
        float alpha = smoothstep(1.0 - LEVEL_BLEND, 1.0, t);
        if (alpha > 0.0)
        {
            float3 grad_next;
            float d_next = sampleSdfLevelGrad(p, lev + 1u, grad_next);
            if (d_next < 9e9)
            {
                // ∇α = dα/dt · ∇t  where t = L∞ norm of rel.
                // dα/dt = 6·t_n·(1-t_n)/BLEND  (derivative of smoothstep).
                float t_n = clamp((t - (1.0 - LEVEL_BLEND)) / LEVEL_BLEND, 0.0, 1.0);
                float da_dt = 6.0 * t_n * (1.0 - t_n) / LEVEL_BLEND;
                // ∇t selects the axis (or axes) achieving the max.
                float3 grad_alpha = float3(0.0, 0.0, 0.0);
                if (abs(rel.x - t) < 1e-5)
                    grad_alpha.x = da_dt * sign(p.x - ctr.x) / half_e.x;
                if (abs(rel.y - t) < 1e-5)
                    grad_alpha.y = da_dt * sign(p.y - ctr.y) / half_e.y;
                if (abs(rel.z - t) < 1e-5)
                    grad_alpha.z = da_dt * sign(p.z - ctr.z) / half_e.z;

                grad = (1.0 - alpha) * grad + alpha * grad_next + grad_alpha * (d_next - d);
            }
        }
    }

    float g = length(grad);
    return g > 1e-6 ? grad / g : float3(0, 1, 0);
}

// ─── Sphere-trace through clipmap ─────────────────────────────────────────────

bool traceClipmap(float3 ro, float3 rd, float far,
                  out float t_hit, out float3 hit_normal, out uint hit_lev, out int steps)
{
    float t = 0.0;
    steps = 0;
    for (int i = 0; i < 512; i++)
    {
        steps = i + 1;
        float3 p = ro + t * rd;
        uint lev;
        float d = sampleFinest(p, lev);

        if (d < 9e9)
        {
            float vs = pc.levels[lev].voxel_size;
            if (d < vs * 0.15)
            {
                t_hit = t;
                hit_lev = lev;
                hit_normal = normalFinest(p, lev);
                return true;
            }
            t += max(d * 0.85, vs * 0.05);
        }
        else
        {
            // Outside all levels — advance by the coarsest brick size.
            t += float(BRICK_DIM) * pc.levels[pc.num_levels - 1u].voxel_size;
        }

        if (t >= far)
            break;
    }

    t_hit = far;
    hit_normal = float3(0, 1, 0);
    hit_lev = 0u;
    return false;
}

// ─── Fullscreen vertex ────────────────────────────────────────────────────────

[shader("vertex")]
float4 vsMain(uint id: SV_VertexID) : SV_Position
{
    float2 uv = float2(float((id << 1u) & 2u), float(id & 2u));
    return float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

// ─── GBuffer output ───────────────────────────────────────────────────────────

struct GBufferOut
{
    float4 albedo : SV_Target0;
    float2 normal : SV_Target1;
    float2 roughMetal : SV_Target2;
    float3 emissive : SV_Target3;
    float hizDepth : SV_Target4;
    float depth : SV_Depth;
};

[shader("fragment")]
GBufferOut fsMain(float4 svPos: SV_Position)
{
    if (pc.num_levels == 0u)
        discard;

    float2 screenSize = g_common.GetScreenSize();
    float2 uv = svPos.xy / screenSize;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float4 wFar = mul(g_common.invViewProjection, float4(ndc, 0.0, 1.0));
    wFar /= wFar.w;

    float3 ro = g_common.GetCameraPos();
    float3 rd = normalize(wFar.xyz - ro);

    float t_hit;
    float3 hit_normal;
    uint hit_lev;
    int steps;
    if (!traceClipmap(ro, rd, g_common.GetFarPlane(), t_hit, hit_normal, hit_lev, steps))
        discard;

    float3 hitPos = ro + t_hit * rd;
    float4 clip = mul(g_common.viewProjection, float4(hitPos, 1.0));
    float ndcDepth = clip.z / clip.w;

    GBufferOut o;
    // Default albedo
    o.albedo = float4(0.75, 0.73, 0.70, 1.0);

    // bit 0 – level colors
    if ((pc.debug_flags & 1u) != 0u)
    {
        float4 level_colors[10] = {
            float4(0.05, 0.20, 1.00, 1.0), // L0  0.0125 m — deep blue
            float4(0.10, 0.50, 1.00, 1.0), // L1  0.025 m
            float4(0.10, 0.75, 0.90, 1.0), // L2  0.05 m
            float4(0.15, 0.90, 0.55, 1.0), // L3  0.10 m
            float4(0.20, 0.85, 0.20, 1.0), // L4  0.20 m
            float4(0.85, 0.85, 0.10, 1.0), // L5  0.40 m — yellow
            float4(0.95, 0.55, 0.05, 1.0), // L6  0.80 m
            float4(0.90, 0.25, 0.05, 1.0), // L7  1.60 m
            float4(0.80, 0.05, 0.20, 1.0), // L8  4.00 m
            float4(0.60, 0.00, 0.40, 1.0), // L9  8.00 m — purple
        };
        o.albedo = level_colors[min(hit_lev, 9u)];
    }

    // bit 1 – brick grid overlay (black lines at brick boundaries)
    if ((pc.debug_flags & 2u) != 0u)
    {
        float bws = float(BRICK_DIM) * pc.levels[hit_lev].voxel_size;
        float3 bf = frac((hitPos - pc.levels[hit_lev].world_origin) / bws);
        float3 be = min(bf, 1.0 - bf);
        float edge = min(be.x, min(be.y, be.z));
        float t = 1.0 - smoothstep(0.0, 0.04, edge);
        o.albedo = lerp(o.albedo, float4(0.0, 0.0, 0.0, 1.0), t);
    }

    // bit 2 – world-Y gradient (hue cycles every 2 m)
    if ((pc.debug_flags & 4u) != 0u)
    {
        float h = frac(hitPos.y * 0.5);
        // Simple HSV→RGB with S=0.8 V=0.9
        float3 c = clamp(abs(fmod(h * 6.0 + float3(0, 4, 2), 6.0) - 3.0) - 1.0, 0.0, 1.0);
        o.albedo = float4(lerp(float3(1, 1, 1), c, 0.8) * 0.9, 1.0);
    }

    // bit 3 – step-count heat map (green=cheap → red=expensive, full at 256 steps)
    if ((pc.debug_flags & 8u) != 0u)
    {
        float heat = saturate(float(steps) / 256.0);
        o.albedo = float4(heat, 1.0 - heat, 0.0, 1.0);
    }

    o.normal = encodeOctahedral(hit_normal);
    o.roughMetal = float2(0.6, 0.0);
    o.emissive = float3(0.0, 0.0, 0.0);
    o.hizDepth = ndcDepth;
    o.depth = ndcDepth;
    return o;
}
