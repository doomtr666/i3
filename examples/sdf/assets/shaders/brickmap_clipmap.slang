#include "common.slangh"

// ─── Clipmap storage buffers (Set 0) — all levels concatenated ────────────────
[[vk::binding(0, 0)]] StructuredBuffer<uint>  page_table;
[[vk::binding(1, 0)]] ByteAddressBuffer       sdf_atlas; // u8-packed voxels
[[vk::binding(2, 0)]] ByteAddressBuffer       mat_atlas; // u8-packed material IDs

// ─── Push constants ───────────────────────────────────────────────────────────
struct LevelPC
{
    float3 world_origin;
    float  voxel_size;
};
struct ClipmapPC
{
    LevelPC levels[10];
    uint    num_levels;
    uint    debug_flags; // bit 0=level colours  1=brick grid  2=world-Y  3=step heat
    uint2   _pad;
};
[[vk::push_constant]] ClipmapPC pc;

// ─── Constants ────────────────────────────────────────────────────────────────
static const uint  BRICK_DIM    = 8u;
static const uint  BRICK_VOXELS = 729u;   // 9³
static const uint  BRICK_DWORDS = 183u;   // ceil(729/4)
static const uint  GRID_X       = 32u;
static const uint  GRID_Y       = 32u;
static const uint  GRID_Z       = 32u;
static const uint  GRID_VOL     = 32768u;
static const uint  MAX_BRICKS   = 8192u;
static const uint  PAGE_EMPTY   = 0xFFFFFFFFu;
static const float LEVEL_BLEND  = 0.25;

// ─── Materials ────────────────────────────────────────────────────────────────
struct Material { float3 albedo; float roughness; float metalness; };
static const Material kMaterials[8] = {
    { float3(0.75, 0.73, 0.70), 0.60, 0.00 }, // 0: grey stone (default)
    { float3(0.80, 0.30, 0.15), 0.80, 0.00 }, // 1: terracotta
    { float3(0.95, 0.95, 0.95), 0.10, 1.00 }, // 2: polished silver
    { float3(0.85, 0.65, 0.10), 0.20, 1.00 }, // 3: gold
    { float3(0.20, 0.35, 0.70), 0.60, 0.00 }, // 4: blue ceramic
    { float3(0.15, 0.55, 0.25), 0.90, 0.00 }, // 5: matte green
    { float3(0.70, 0.15, 0.15), 0.40, 0.00 }, // 6: red plastic
    { float3(0.05, 0.05, 0.05), 0.80, 1.00 }, // 7: black metal
};

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

// ─── Atlas u8 accessors ───────────────────────────────────────────────────────

// Decode one SDF byte at vox_flat index within a brick.
// base_bytes: byte offset of brick in ByteAddressBuffer.
// half_diag:  brick half-diagonal (decode scale).
float loadSdfVoxel(uint base_bytes, uint vox_flat, float half_diag)
{
    uint dword_off = vox_flat / 4u;
    uint byte_pos  = vox_flat % 4u;
    uint packed    = sdf_atlas.Load(base_bytes + dword_off * 4u);
    uint byte_val  = (packed >> (byte_pos * 8u)) & 0xFFu;
    return (float(byte_val) / 127.5 - 1.0) * half_diag;
}

// Load a material ID byte at vox_flat index within a brick.
uint loadMatVoxel(uint base_bytes, uint vox_flat)
{
    uint dword_off = vox_flat / 4u;
    uint byte_pos  = vox_flat % 4u;
    uint packed    = mat_atlas.Load(base_bytes + dword_off * 4u);
    return (packed >> (byte_pos * 8u)) & 0xFFu;
}

// Common brick lookup shared by SDF and material samplers.
// Returns false if p is outside the level or in an empty brick.
// Fills: base_bytes, vi (corner voxel index), fr (trilinear fractions), half_diag.
bool brickLookup(float3 p, uint lev,
                 out uint base_bytes, out int3 vi, out float3 fr, out float half_diag)
{
    float3 org = pc.levels[lev].world_origin;
    float  vs  = pc.levels[lev].voxel_size;
    float  bws = float(BRICK_DIM) * vs;
    half_diag  = float(BRICK_DIM) * vs * 0.8660254; // sqrt(3)/2

    float3 gp = (p - (org + 0.5 * vs)) / bws;
    int3 b = clamp(int3(floor(gp)), int3(0), int3(GRID_X - 1, GRID_Y - 1, GRID_Z - 1));

    int3 ob   = int3(round(org / bws));
    int3 grid = int3(GRID_X, GRID_Y, GRID_Z);
    int3 cell = ((ob + b) % grid + grid) % grid;

    uint page = page_table[lev * GRID_VOL + brickFlatIdx(cell)];
    if (page == PAGE_EMPTY)
        return false;

    base_bytes = (lev * MAX_BRICKS + page) * BRICK_DWORDS * 4u;

    float3 lp = (p - (org + float3(b) * bws)) / vs - 0.5;
    lp = clamp(lp, float3(0.0), float3(float(BRICK_DIM) - 1e-4));
    vi = int3(floor(lp));
    fr = lp - float3(vi);
    vi = min(vi, int3(int(BRICK_DIM) - 1, int(BRICK_DIM) - 1, int(BRICK_DIM) - 1));
    return true;
}

// ─── Trilinear SDF + Smooth Gradient ─────────────────────────────────────────

float sampleSdfLevel(float3 p, uint lev)
{
    uint   base_bytes;
    int3   vi;
    float3 fr;
    float  half_diag;
    if (!brickLookup(p, lev, base_bytes, vi, fr, half_diag))
        return 1e10;

    uint bd = BRICK_DIM + 1u;
    float fx = fr.x, fy = fr.y, fz = fr.z;

    // Fetch 8 corners via u8 atlas
    float s000 = loadSdfVoxel(base_bytes, uint(vi.x  ) + bd*(uint(vi.y  ) + bd*uint(vi.z  )), half_diag);
    float s100 = loadSdfVoxel(base_bytes, uint(vi.x+1) + bd*(uint(vi.y  ) + bd*uint(vi.z  )), half_diag);
    float s010 = loadSdfVoxel(base_bytes, uint(vi.x  ) + bd*(uint(vi.y+1) + bd*uint(vi.z  )), half_diag);
    float s110 = loadSdfVoxel(base_bytes, uint(vi.x+1) + bd*(uint(vi.y+1) + bd*uint(vi.z  )), half_diag);
    float s001 = loadSdfVoxel(base_bytes, uint(vi.x  ) + bd*(uint(vi.y  ) + bd*uint(vi.z+1)), half_diag);
    float s101 = loadSdfVoxel(base_bytes, uint(vi.x+1) + bd*(uint(vi.y  ) + bd*uint(vi.z+1)), half_diag);
    float s011 = loadSdfVoxel(base_bytes, uint(vi.x  ) + bd*(uint(vi.y+1) + bd*uint(vi.z+1)), half_diag);
    float s111 = loadSdfVoxel(base_bytes, uint(vi.x+1) + bd*(uint(vi.y+1) + bd*uint(vi.z+1)), half_diag);

    return lerp(
        lerp(lerp(s000, s100, fx), lerp(s010, s110, fx), fy),
        lerp(lerp(s001, s101, fx), lerp(s011, s111, fx), fy),
        fz);
}

float sampleSdfLevelGrad(float3 p, uint lev, out float3 grad)
{
    float val = sampleSdfLevel(p, lev);

    float e = pc.levels[lev].voxel_size * 0.5;

    float dx = sampleSdfLevel(p + float3(e, 0, 0), lev) - sampleSdfLevel(p - float3(e, 0, 0), lev);
    float dy = sampleSdfLevel(p + float3(0, e, 0), lev) - sampleSdfLevel(p - float3(0, e, 0), lev);
    float dz = sampleSdfLevel(p + float3(0, 0, e), lev) - sampleSdfLevel(p - float3(0, 0, e), lev);

    grad = float3(dx, dy, dz) / (2.0 * e);
    return val;
}

// ─── Material sampling ────────────────────────────────────────────────────────
// Trilinear blend of material properties weighted by surface proximity.
// Corners with sdf < 0 (inside solid) contribute fully;
// corners outside attenuate with exp(-sdf / vs) over ~1 voxel width.

Material sampleMatBlended(float3 p, uint lev)
{
    uint   base_bytes;
    int3   vi;
    float3 fr;
    float  half_diag;
    if (!brickLookup(p, lev, base_bytes, vi, fr, half_diag))
        return kMaterials[0];

    float vs = pc.levels[lev].voxel_size;
    uint  bd = BRICK_DIM + 1u;

    Material result;
    result.albedo    = float3(0.0, 0.0, 0.0);
    result.roughness = 0.0;
    result.metalness = 0.0;
    float total_w = 0.0;

    for (uint iz = 0u; iz <= 1u; iz++)
    for (uint iy = 0u; iy <= 1u; iy++)
    for (uint ix = 0u; ix <= 1u; ix++)
    {
        uint  vf  = uint(vi.x + ix) + bd * (uint(vi.y + iy) + bd * uint(vi.z + iz));
        float sd  = loadSdfVoxel(base_bytes, vf, half_diag);
        uint  mid = loadMatVoxel(base_bytes, vf);

        float wx = (ix == 0u) ? (1.0 - fr.x) : fr.x;
        float wy = (iy == 0u) ? (1.0 - fr.y) : fr.y;
        float wz = (iz == 0u) ? (1.0 - fr.z) : fr.z;
        float w  = wx * wy * wz * exp(-max(sd, 0.0) / vs);

        Material m = kMaterials[min(mid, 7u)];
        result.albedo    += m.albedo    * w;
        result.roughness += m.roughness * w;
        result.metalness += m.metalness * w;
        total_w          += w;
    }

    if (total_w > 1e-6)
    {
        result.albedo    /= total_w;
        result.roughness /= total_w;
        result.metalness /= total_w;
    }
    else
        result = kMaterials[0];

    return result;
}

// ─── Level sampling with transition blending ──────────────────────────────────

float sampleFinest(float3 p, out uint lev_out)
{
    for (uint lev = 0u; lev < pc.num_levels; lev++)
    {
        float bws  = float(BRICK_DIM) * pc.levels[lev].voxel_size;
        float3 org = pc.levels[lev].world_origin;
        float3 gmax = org + float3(GRID_X, GRID_Y, GRID_Z) * bws;
        if (!all(p >= org) || !all(p < gmax))
            continue;

        float d = sampleSdfLevel(p, lev);
        if (d >= 9e9)
            continue;

        lev_out = lev;

        if (lev + 1u < pc.num_levels)
        {
            float3 half_e = float3(GRID_X, GRID_Y, GRID_Z) * bws * 0.5;
            float3 ctr    = org + half_e;
            float3 rel    = abs(p - ctr) / half_e;
            float  t      = max(rel.x, max(rel.y, rel.z));
            float  alpha  = smoothstep(1.0 - LEVEL_BLEND, 1.0, t);
            if (alpha > 0.0)
            {
                float d_next = sampleSdfLevel(p, lev + 1u);
                if (d_next < 9e9)
                    d = lerp(d, min(d, d_next), alpha);
            }
        }
        return d;
    }
    lev_out = pc.num_levels;
    return 1e10;
}

// ─── Smooth normal (Söderström / JCGT 2022) ───────────────────────────────────

float3 normalFinest(float3 p, uint lev)
{
    if (lev >= pc.num_levels)
        return float3(0, 1, 0);
    float3 grad;
    float d = sampleSdfLevelGrad(p, lev, grad);
    if (d >= 9e9)
        return float3(0, 1, 0);

    if (lev + 1u < pc.num_levels)
    {
        float  bws    = float(BRICK_DIM) * pc.levels[lev].voxel_size;
        float3 org    = pc.levels[lev].world_origin;
        float3 half_e = float3(GRID_X, GRID_Y, GRID_Z) * bws * 0.5;
        float3 ctr    = org + half_e;
        float3 rel    = abs(p - ctr) / half_e;
        float  t      = max(rel.x, max(rel.y, rel.z));
        float  alpha  = smoothstep(1.0 - LEVEL_BLEND, 1.0, t);
        if (alpha > 0.0)
        {
            float3 grad_next;
            float d_next = sampleSdfLevelGrad(p, lev + 1u, grad_next);
            float vs_next = pc.levels[lev + 1u].voxel_size;
            if (d_next < 9e9 && d_next < vs_next * 2.0)
            {
                float  t_n    = clamp((t - (1.0 - LEVEL_BLEND)) / LEVEL_BLEND, 0.0, 1.0);
                float  da_dt  = 6.0 * t_n * (1.0 - t_n) / LEVEL_BLEND;
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
                t_hit      = t;
                hit_lev    = lev;
                hit_normal = normalFinest(p, lev);
                return true;
            }
            t += max(d * 0.85, vs * 0.05);
        }
        else
        {
            // No SDF data at p — advance by the finest brick size that covers this point.
            // Using the coarsest-level step (64 m) overshoots all geometry in a sparse
            // GPU-culled clipmap where free-space cells are PAGE_EMPTY.
            float step = float(BRICK_DIM) * pc.levels[pc.num_levels - 1u].voxel_size;
            for (uint fb = 0u; fb < pc.num_levels; fb++)
            {
                float  bws_fb  = float(BRICK_DIM) * pc.levels[fb].voxel_size;
                float3 org_fb  = pc.levels[fb].world_origin;
                float3 end_fb  = org_fb + float3(GRID_X, GRID_Y, GRID_Z) * bws_fb;
                if (all(p >= org_fb) && all(p < end_fb))
                {
                    step = bws_fb;
                    break;
                }
            }
            t += step;
        }

        if (t >= far)
            break;
    }

    t_hit      = far;
    hit_normal = float3(0, 1, 0);
    hit_lev    = 0u;
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
    float4 albedo     : SV_Target0;
    float2 normal     : SV_Target1;
    float2 roughMetal : SV_Target2;
    float3 emissive   : SV_Target3;
    float  hizDepth   : SV_Target4;
    float  depth      : SV_Depth;
};

[shader("fragment")]
GBufferOut fsMain(float4 svPos: SV_Position)
{
    if (pc.num_levels == 0u)
        discard;

    float2 screenSize = g_common.GetScreenSize();
    float2 uv  = svPos.xy / screenSize;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float4 wFar = mul(g_common.invViewProjection, float4(ndc, 0.0, 1.0));
    wFar /= wFar.w;

    float3 ro = g_common.GetCameraPos();
    float3 rd = normalize(wFar.xyz - ro);

    float  t_hit;
    float3 hit_normal;
    uint   hit_lev;
    int    steps;
    if (!traceClipmap(ro, rd, g_common.GetFarPlane(), t_hit, hit_normal, hit_lev, steps))
        discard;

    float3 hitPos  = ro + t_hit * rd;
    float4 clip    = mul(g_common.viewProjection, float4(hitPos, 1.0));
    float  ndcDepth = clip.z / clip.w;

    // Material lookup — trilinear blend weighted by surface proximity
    Material mat = sampleMatBlended(hitPos, hit_lev);

    GBufferOut o;
    o.albedo = float4(mat.albedo, 1.0);

    // bit 0 – level colors
    if ((pc.debug_flags & 1u) != 0u)
    {
        float4 level_colors[10] = {
            float4(0.05, 0.20, 1.00, 1.0), // L0  0.0125 m
            float4(0.10, 0.50, 1.00, 1.0), // L1  0.025 m
            float4(0.10, 0.75, 0.90, 1.0), // L2  0.05 m
            float4(0.15, 0.90, 0.55, 1.0), // L3  0.10 m
            float4(0.20, 0.85, 0.20, 1.0), // L4  0.20 m
            float4(0.85, 0.85, 0.10, 1.0), // L5  0.40 m
            float4(0.95, 0.55, 0.05, 1.0), // L6  0.80 m
            float4(0.90, 0.25, 0.05, 1.0), // L7  1.60 m
            float4(0.80, 0.05, 0.20, 1.0), // L8  4.00 m
            float4(0.60, 0.00, 0.40, 1.0), // L9  8.00 m
        };
        o.albedo = level_colors[min(hit_lev, 9u)];
    }

    // bit 1 – brick grid overlay
    if ((pc.debug_flags & 2u) != 0u)
    {
        float  bws = float(BRICK_DIM) * pc.levels[hit_lev].voxel_size;
        float3 bf  = frac((hitPos - pc.levels[hit_lev].world_origin) / bws);
        float3 be  = min(bf, 1.0 - bf);
        float  edge = min(be.x, min(be.y, be.z));
        float  tt   = 1.0 - smoothstep(0.0, 0.04, edge);
        o.albedo = lerp(o.albedo, float4(0.0, 0.0, 0.0, 1.0), tt);
    }

    // bit 2 – world-Y gradient
    if ((pc.debug_flags & 4u) != 0u)
    {
        float  h = frac(hitPos.y * 0.5);
        float3 c = clamp(abs(fmod(h * 6.0 + float3(0, 4, 2), 6.0) - 3.0) - 1.0, 0.0, 1.0);
        o.albedo = float4(lerp(float3(1, 1, 1), c, 0.8) * 0.9, 1.0);
    }

    // bit 3 – step-count heat map
    if ((pc.debug_flags & 8u) != 0u)
    {
        float heat = saturate(float(steps) / 256.0);
        o.albedo = float4(heat, 1.0 - heat, 0.0, 1.0);
    }

    o.normal     = encodeOctahedral(hit_normal);
    o.roughMetal = float2(mat.roughness, mat.metalness);
    o.emissive   = float3(0.0, 0.0, 0.0);
    o.hizDepth   = ndcDepth;
    o.depth      = ndcDepth;
    return o;
}
