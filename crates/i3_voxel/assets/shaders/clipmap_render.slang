#include "common.slangh"
#include "noise.slangh" // terrainDensity — for analytic debug (bit 4/5) parity with the bake

// ─── Set 0: clipmap buffers ───────────────────────────────────────────────────
// page_table: NUM_LEVELS·GRID³ toroidal slot indices (PAGE_EMPTY = no brick).
// clip_levels: per-level window origin + voxel size — the lookup needs no descent.
struct LevelPC
{
    float3 world_origin; // window corner = origin_brick · (BRICK_DIM·voxel_size)
    float  voxel_size;
};

struct GpuPrimitive
{
    float4 row0;
    float4 row1;
    float4 row2;
    float inv_scale;
    float data0;
    float data1;
    float data2;
    uint prim_type;
    uint subtract;
    uint material_id;
    uint _pad;
};

[[vk::binding(0, 0)]]
StructuredBuffer<uint> page_table;
[[vk::binding(1, 0)]]
ByteAddressBuffer geom_atlas; // 8 bytes/voxel
[[vk::binding(2, 0)]]
StructuredBuffer<GpuPrimitive> prims;
[[vk::binding(4, 0)]]
ByteAddressBuffer weight_atlas; // 4 bytes/voxel: baked terrain layer weights (RGBA8)
[[vk::binding(5, 0)]]
StructuredBuffer<LevelPC> clip_levels;

// ─── Push constants ───────────────────────────────────────────────────────────
struct ClipmapPC
{
    uint num_levels;
    uint debug_flags;
    uint prim_count;
    uint terrain_on;
    // Terrain layers (0=grass 1=rock 2=snow 3=dirt): bindless texture indices (-1 = none)
    int4 idxAlbedo;
    int4 idxNormal;
    int4 idxOrm;
    float4 thr0;       // snowLo, snowHi, slopeLo, slopeHi
    float4 thr1;       // dirtLo, dirtHi, tiling, jitterAmp
};
[[vk::push_constant]]
ClipmapPC pc;

// Compile-time gate for the analytic ground-truth DEBUG paths (bits 4/5/6). They call
// gtScene (full prim CSG + terrain noise VM), which costs ~40 live registers — allocated
// STATICALLY for the whole shader, so it caps occupancy even though it's never taken in
// production (the SASS register peak was here). Flip to 1 only to debug brick vs analytic.
#define SVO_DEBUG 0

#define SVO_NO_MATERIAL 0

#if SVO_DEBUG
// ─── Analytic SDF (ground-truth, bypasses bricks) ─────────────────────────────
float gtSphere(float3 p, float r)
{
    return length(p) - r;
}
float gtBox(float3 p, float3 he)
{
    float3 q = abs(p) - he;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}
float gtCapsule(float3 p, float hh, float r)
{
    float cy = clamp(p.y, -hh, hh);
    return length(p - float3(0, cy, 0)) - r;
}
float gtCylinder(float3 p, float hh, float r)
{
    float dxz = length(p.xz) - r;
    float dy = abs(p.y) - hh;
    return length(max(float2(dxz, dy), 0.0)) + min(max(dxz, dy), 0.0);
}
float gtTorus(float3 p, float R, float r)
{
    float2 q = float2(length(p.xz) - R, p.y);
    return length(q) - r;
}

float gtEvalPrim(float3 pw, GpuPrimitive prim)
{
    float3 pc2 = pw - float3(prim.row0.w, prim.row1.w, prim.row2.w);
    float3 pr = float3(dot(prim.row0.xyz, pc2), dot(prim.row1.xyz, pc2), dot(prim.row2.xyz, pc2));
    float3 pl = pr * prim.inv_scale;
    float d;
    if (prim.prim_type == 0u)
        d = gtSphere(pl, prim.data0);
    else if (prim.prim_type == 1u)
        d = gtBox(pl, float3(prim.data0, prim.data1, prim.data2));
    else if (prim.prim_type == 2u)
        d = gtCapsule(pl, prim.data0, prim.data1);
    else if (prim.prim_type == 3u)
        d = gtCylinder(pl, prim.data0, prim.data1);
    else
        d = gtTorus(pl, prim.data0, prim.data1);
    return d / prim.inv_scale;
}

// CSG union of solids minus subtracts; returns distance and nearest material.
float gtScene(float3 p, out uint mat_id)
{
    float solid = 1e10, empty = 1e10, mind = 1e10;
    mat_id = 0u;
    for (uint i = 0u; i < pc.prim_count; i++)
    {
        float d = gtEvalPrim(p, prims[i]);
        if (prims[i].subtract != 0u)
        {
            empty = min(empty, d);
        }
        else
        {
            solid = min(solid, d);
            if (d < mind)
            {
                mind = d;
                mat_id = prims[i].material_id;
            }
        }
    }
    if (pc.terrain_on != 0u)
    {
        float td = terrainDensity(p, float3(0, 0, 0)).v;
        if (td < solid)
        {
            solid = td;
            mat_id = 5u;
        } // TERRAIN_MAT
    }
    return max(solid, -empty);
}
#endif // SVO_DEBUG

// ─── Constants ────────────────────────────────────────────────────────────────
static const uint BRICK_DIM = 8u;
static const float INF = 1e10;

// Clipmap cascade — MUST match crate::clipmap (NUM_LEVELS comes from pc.num_levels).
static const int  CLIP_GRID  = 32;
static const uint GRID_VOL   = 32768u;          // CLIP_GRID³
static const uint MAX_BRICKS = 6144u;           // MAX_BRICKS_PER_LEVEL
static const uint BRICK_BYTES = 5832u;          // BRICK_VOXELS(729) · 8
static const uint PAGE_EMPTY = 0xFFFFFFFFu;
// Kept for the depth-colour debug overlay only.
static const uint MAX_SVO_DEPTH = 18u;

// ─── Materials (same as clipmap) ─────────────────────────────────────────────
struct Material
{
    float3 albedo;
    float roughness;
    float metalness;
};
static const Material kMaterials[8] = {
    { float3(0.75, 0.73, 0.70), 0.60, 0.00 },
    { float3(0.80, 0.30, 0.15), 0.80, 0.00 },
    { float3(0.95, 0.95, 0.95), 0.10, 1.00 },
    { float3(0.85, 0.65, 0.10), 0.20, 1.00 },
    { float3(0.20, 0.35, 0.70), 0.60, 0.00 },
    { float3(0.15, 0.55, 0.25), 0.90, 0.00 },
    { float3(0.70, 0.15, 0.15), 0.40, 0.00 },
    { float3(0.05, 0.05, 0.05), 0.80, 1.00 },
};

// ─── Voxel decode (8 bytes: snorm16 dist over ±BAND·vs + oct snorm16 normal +
//     u8 mat + u8 flags). BAND must match svo_bake. ────────────────────────────
static const float BAND = 4.0;
struct Voxel
{
    float dist;
    float3 normal;
    uint mat;
};

float snorm16decode(uint u)
{
    int s = (int(u) << 16) >> 16;
    return max(float(s) / 32767.0, -1.0);
}

float3 octDecode(float2 e)
{
    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0)
        n.xy = (1.0 - abs(float2(n.y, n.x))) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

Voxel loadVoxel(uint base_bytes, uint vox, float band)
{
    uint2 w = geom_atlas.Load2(base_bytes + vox * 8u);
    Voxel v;
    v.dist = snorm16decode(w.x & 0xFFFFu) * band; // ±BAND·vs world units
    v.normal = octDecode(float2(snorm16decode((w.x >> 16u) & 0xFFFFu),
                                snorm16decode(w.y & 0xFFFFu)));
    v.mat = (w.y >> 16u) & 0xFFu;
    return v;
}

// Distance-ONLY voxel load: reads just word0 (4 B) and decodes the distance — no
// normal, no material. The sphere-march only needs distance per step; decoding the
// oct-normal (2 extra snorm16 + octDecode) ×8 corners ×every step was the hottest
// cost (snorm16decode/loadVoxel dominated the samples). Full normal+mat is decoded
// ONCE at the hit (reconstructBrick).
float loadVoxelDist(uint base_bytes, uint vox, float band)
{
    uint w0 = geom_atlas.Load(base_bytes + vox * 8u);
    return snorm16decode(w0 & 0xFFFFu) * band;
}

// ─── SVO tree descent ─────────────────────────────────────────────────────────

// found  = p is in a baked (surface) leaf, sdf/mat/vs valid.
// inside = p is inside the tree at all (false ⇒ outside root ⇒ ray left the scene).
// cmin/cmax = bounds of the cell p is in (empty leaf or outside), for exact skipping.
struct BrickSample
{
    float sdf;
    uint mat_id;
    float voxel_size;
    float half_diag;
    bool found;
    bool inside;
    float3 cmin;
    float3 cmax;
    float3 grad;
};

// A located leaf: its bounds + brick handle. CACHED across sphere-trace steps so the
// tree descent (root→leaf) — the renderer's #1 hotspot — runs only when the ray
// crosses into a NEW cell, not at every march step.
struct LeafRef
{
    float3 aabb_min;
    float3 aabb_max;
    float vs;
    uint brick_offset; // ~0u = empty (no surface)
    uint level;        // cascade level this brick is from (for inter-level blend)
    bool found;        // surface leaf (brick_offset valid)
    bool inside;       // p was inside the tree (false ⇒ ray left the scene)
};

int  posmod(int a, int m)   { return ((a % m) + m) % m; }
uint brickFlat(int3 c)      { return uint(c.x) + uint(CLIP_GRID) * (uint(c.y) + uint(CLIP_GRID) * uint(c.z)); }

// O(1) leaf lookup (replaces the SVO root→leaf descent — the old #1 hotspot). Picks the
// FINEST cascade level whose toroidal window contains p, then a single page_table read.
// Returns the same LeafRef the sphere-march consumes (brick bounds + atlas handle), so the
// march loop, reconstruct, and texturing are identical to the SVO path. `found=false` for
// an empty cell at the finest containing level (the march `boxExit`s it); `inside=false`
// when p is outside every window (ray left the clipmap).
// Look up the brick at a SPECIFIC cascade level (no level search). `inside` = p is within
// this level's window; `found` = that toroidal cell holds a baked surface brick.
LeafRef lookupAtLevel(float3 p, uint lev)
{
    LeafRef r;
    r.aabb_min = float3(0, 0, 0);
    r.aabb_max = float3(0, 0, 0);
    r.vs = 0.0;
    r.brick_offset = ~0u;
    r.level = lev;
    r.found = false;
    r.inside = false;

    float3 org = clip_levels[lev].world_origin;
    float  vs  = clip_levels[lev].voxel_size;
    float  bws = float(BRICK_DIM) * vs;
    float3 gmax = org + float(CLIP_GRID) * bws;
    if (any(p < org) || any(p >= gmax))
        return r; // not in this level's window

    int3 b  = clamp(int3(floor((p - org) / bws)), int3(0), int3(CLIP_GRID - 1));
    int3 ob = int3(round(org / bws));
    int3 cell = int3(posmod(ob.x + b.x, CLIP_GRID),
                     posmod(ob.y + b.y, CLIP_GRID),
                     posmod(ob.z + b.z, CLIP_GRID));
    uint page = page_table[lev * GRID_VOL + brickFlat(cell)];

    float3 amin = org + float3(b) * bws;
    r.aabb_min = amin;
    r.aabb_max = amin + bws;
    r.vs       = vs;
    r.inside   = true;
    if (page != PAGE_EMPTY)
    {
        r.brick_offset = (lev * MAX_BRICKS + page) * BRICK_BYTES;
        r.found = true;
    }
    return r;
}

// Cross-fade weight toward the NEXT coarser level: 0 in the interior, ramping to 1 across
// the outer LEVEL_BLEND fraction of this level's window (near its faces). Blending the
// marched SDF by this removes the hard resolution seam at the window boundary AND the
// temporal pop when the window shifts (the entering border eases between levels).
static const float LEVEL_BLEND = 0.12;
float levelBlend(float3 p, uint lev)
{
    float3 org = clip_levels[lev].world_origin;
    float  bws = float(BRICK_DIM) * clip_levels[lev].voxel_size;
    float3 q   = (p - org) / (float(CLIP_GRID) * bws); // [0,1] within the window
    float3 d2  = min(q, 1.0 - q);                       // normalised dist to nearest face
    float  edge = min(d2.x, min(d2.y, d2.z));
    return 1.0 - smoothstep(0.0, LEVEL_BLEND, edge);    // 1 at a face → 0 by LEVEL_BLEND inward
}

// O(1) leaf lookup (replaces the SVO root→leaf descent — the old #1 hotspot). Picks the
// FINEST cascade level whose toroidal window contains p, starting at the distance-derived
// level (the windows are concentric, each 2× the previous, so the finest level that can
// contain p is ≈ log2(dist / L0-half-extent) — distant pixels otherwise burned a window
// test + buffer load per fine level they overshoot, >half the pass). Biased down one level
// so a snapped window never skips the correct finer level (≤1 extra iteration near a ring).
LeafRef clipmapLookup(float3 p)
{
    float3 cam = g_common.GetCameraPos();
    float  dax = max(abs(p.x - cam.x), max(abs(p.y - cam.y), abs(p.z - cam.z)));
    float  h0  = float(CLIP_GRID / 2) * float(BRICK_DIM) * clip_levels[0].voxel_size;
    int    lev0 = clamp(int(floor(log2(max(dax / h0, 1.0)))) - 1, 0, int(pc.num_levels) - 1);

    [loop]
    for (uint lev = uint(lev0); lev < pc.num_levels; lev++)
    {
        LeafRef r = lookupAtLevel(p, lev);
        if (r.inside)
            return r; // finest containing level is authoritative
    }

    LeafRef r;
    r.aabb_min = float3(0, 0, 0);
    r.aabb_max = float3(0, 0, 0);
    r.vs = 0.0;
    r.brick_offset = ~0u;
    r.level = 0u;
    r.found = false;
    r.inside = false;
    return r; // outside every window ⇒ inside=false
}

// Reconstruct the SDF value + normal at p for an already-located surface leaf — no
// tree descent. Gradient-augmented tangent-plane blend: each of the 8 corner voxels
// stores distance + analytic normal, defining a plane d_i + n_i·(p − corner_i);
// trilinearly blending those is exact for planes, Hermite-smooth for curves, and stays
// 1-Lipschitz (|Σ w_i n_i| ≤ 1) so it's safe to sphere-march. This was the cheap (~10%)
// half of the old combined descent; calling it per step (vs re-descending) is the win.
BrickSample reconstructBrick(LeafRef lf, float3 p)
{
    BrickSample s;
    s.cmin = lf.aabb_min;
    s.cmax = lf.aabb_max;
    s.voxel_size = lf.vs;
    s.half_diag = 0.0;
    s.found = true;
    s.inside = true;

#if SVO_DEBUG
    // Debug bit 5: traverse the tree but sample analytic SDF (bypass atlas).
    if ((pc.debug_flags & 32u) != 0u)
    {
        uint m;
        s.sdf = gtScene(p, m);
        s.mat_id = m;
        s.grad = float3(0, 1, 0);
        return s;
    }
#endif

    float vs = lf.vs;
    float3 lp = (p - lf.aabb_min) / vs;
    lp = clamp(lp, 0.0, float(BRICK_DIM) - 1e-4);
    int3 vi = min(int3(floor(lp)), int3(BRICK_DIM - 1));
    float3 fr = lp - float3(vi);
    uint bd = BRICK_DIM + 1u;
    float3 c000 = lf.aabb_min + float3(vi) * vs; // vi corner voxel centre
    float band = BAND * vs;

    float dacc = 0.0;
    float3 nacc = float3(0, 0, 0);
    uint mat = 0u;
    float mbest = -1.0;
    // NOT unrolled: this runs once per pixel (at the hit), so loop overhead is free, but
    // rolling reuses the per-corner temporaries (loadVoxel/octDecode) instead of keeping
    // 8 copies live → lower register peak → higher occupancy for the whole shader.
    for (uint c = 0u; c < 8u; c++)
    {
        uint ix = c & 1u, iy = (c >> 1u) & 1u, iz = (c >> 2u) & 1u;
        Voxel v = loadVoxel(lf.brick_offset, (uint(vi.x) + ix) + bd * ((uint(vi.y) + iy) + bd * (uint(vi.z) + iz)), band);
        float wx = (ix == 0u) ? (1.0 - fr.x) : fr.x;
        float wy = (iy == 0u) ? (1.0 - fr.y) : fr.y;
        float wz = (iz == 0u) ? (1.0 - fr.z) : fr.z;
        float w = wx * wy * wz;
        float3 corner = c000 + float3(ix, iy, iz) * vs;
        dacc += w * (v.dist + dot(v.normal, p - corner)); // tangent plane
        nacc += w * v.normal;
        if (w > mbest)
        {
            mbest = w;
            mat = v.mat;
        } // material = dominant corner
    }

    s.sdf = dacc;
    s.grad = nacc;
    s.mat_id = mat;
    return s;
}

// Distance-ONLY brick reconstruction for the sphere-march: plain trilinear of the 8
// corner distances (value-only — no per-corner normal, so no tangent-plane term and no
// normal decode). 8 loads of 4 B + 8 distance decodes per step, vs 8×8 B + 24 decodes
// for the full reconstruct. Safe to march (the stored field is ~1-Lipschitz, same as the
// clipmap marched on values). The full tangent-plane normal+material is done once at the
// hit via `reconstructBrick`.
float reconstructDist(LeafRef lf, float3 p)
{
    float vs = lf.vs;
    float3 lp = clamp((p - lf.aabb_min) / vs, 0.0, float(BRICK_DIM) - 1e-4);
    int3 vi = min(int3(floor(lp)), int3(BRICK_DIM - 1));
    float3 fr = lp - float3(vi);
    uint bd = BRICK_DIM + 1u;
    float band = BAND * vs;

    float dacc = 0.0;
    for (uint c = 0u; c < 8u; c++)
    {
        uint ix = c & 1u, iy = (c >> 1u) & 1u, iz = (c >> 2u) & 1u;
        float d = loadVoxelDist(lf.brick_offset, (uint(vi.x) + ix) + bd * ((uint(vi.y) + iy) + bd * (uint(vi.z) + iz)), band);
        float w = (ix == 0u ? (1.0 - fr.x) : fr.x) * (iy == 0u ? (1.0 - fr.y) : fr.y) * (iz == 0u ? (1.0 - fr.z) : fr.z);
        dacc += w * d;
    }
    return dacc;
}

// Forward ray exit distance from p along rd out of box [bmin,bmax]. p assumed inside.
float boxExit(float3 p, float3 rd, float3 bmin, float3 bmax)
{
    float3 inv = 1.0 / (abs(rd) > 1e-8 ? rd : float3(1e-8, 1e-8, 1e-8) * sign(rd + 1e-12));
    float3 t1 = (bmin - p) * inv;
    float3 t2 = (bmax - p) * inv;
    float3 tmax = max(t1, t2);
    return min(tmax.x, min(tmax.y, tmax.z));
}

// ─── Material: surface-proximity-weighted trilinear blend ─────────────────────
// Descends to the leaf at p and blends the 8 corner materials, weighting each by
// trilinear weight × exp(-max(sdf,0)/vs): corners inside the solid (sdf<0)
// contribute fully, corners outside attenuate over ~1 voxel. Smooth material
// transitions instead of blocky per-voxel pick. Called once per pixel at the hit.
Material sampleMaterial(float3 p)
{
    LeafRef lf = clipmapLookup(p);
    if (!lf.found)
        return kMaterials[0];

    float vs = lf.vs;
    float3 lp = clamp((p - lf.aabb_min) / vs, 0.0, float(BRICK_DIM) - 1e-4);
    int3 vi = min(int3(floor(lp)), int3(BRICK_DIM - 1));
    float3 fr = lp - float3(vi);
    uint bd = BRICK_DIM + 1u;
    float band = BAND * vs;

    Material result;
    result.albedo = float3(0, 0, 0);
    result.roughness = 0.0;
    result.metalness = 0.0;
    float total = 0.0;
    for (uint iz = 0u; iz <= 1u; iz++)
        for (uint iy = 0u; iy <= 1u; iy++)
            for (uint ix = 0u; ix <= 1u; ix++)
            {
                Voxel v = loadVoxel(lf.brick_offset, uint(vi.x + ix) + bd * (uint(vi.y + iy) + bd * uint(vi.z + iz)), band);
                float wx = (ix == 0u) ? (1.0 - fr.x) : fr.x;
                float wy = (iy == 0u) ? (1.0 - fr.y) : fr.y;
                float wz = (iz == 0u) ? (1.0 - fr.z) : fr.z;
                float w = wx * wy * wz * exp(-max(v.dist, 0.0) / vs);
                Material m = kMaterials[min(v.mat, 7u)];
                result.albedo += m.albedo * w;
                result.roughness += m.roughness * w;
                result.metalness += m.metalness * w;
                total += w;
            }
    if (total > 1e-6)
    {
        result.albedo /= total;
        result.roughness /= total;
        result.metalness /= total;
        return result;
    }
    return kMaterials[0];
}

// ─── Terrain shading: slope + height layer blend ──────────────────────────────
// First cut: solid layer colours (grass / rock / snow / dirt) blended by surface
// slope (n.y) and height, with a low-frequency fBm jitter so the bands aren't
// perfectly horizontal. Image textures (triplanar) slot in here later, per layer.
// Triplanar sample of one bindless texture (idx < 0 → returns the fallback).
// `w` are the (normalised) triplanar projection weights, `tiling` the world→uv
// scale and `lod` the mip level (distance-based; screen derivatives are not
// reliable under sphere-tracing).
float4 triSample(int idx, float3 p, float3 w, float tiling, float lod, float4 fallback)
{
    if (idx < 0)
        return fallback;
    // SAMPLER_ANISO is the only Repeat-addressed sampler — required since the
    // triplanar UVs (world·tiling) run far outside [0,1]; ClampToEdge would smear
    // the border texel into long streaks.
    SamplerState s = bindlessSamplers[SAMPLER_ANISO];
    Texture2D t = bindlessTextures[idx];
    float4 cx = t.SampleLevel(s, p.zy * tiling, lod); // X-facing plane
    float4 cy = t.SampleLevel(s, p.xz * tiling, lod); // Y-facing plane
    float4 cz = t.SampleLevel(s, p.xy * tiling, lod); // Z-facing plane
    return cx * w.x + cy * w.y + cz * w.z;
}

// Unpack a BC5_UNORM (2-channel) tangent-space normal: RG → [-1,1], reconstruct Z.
float3 unpackNormalBC5(float2 rg)
{
    float2 xy = rg * 2.0 - 1.0;
    return float3(xy, sqrt(saturate(1.0 - dot(xy, xy))));
}

// Trilinear read of the BAKED layer weights (RGBA8 = grass/rock/snow/dirt) at p inside
// the hit leaf. The pre-mix: the slope/height/biome blend was computed ONCE at bake;
// here we only fetch it (8 corner reads). The weight atlas mirrors geom at half scale,
// so the byte address is exactly the geom address >> 1.
float4 loadWeights(uint brick_offset, float3 aabb_min, float vs, float3 p)
{
    float3 lp = clamp((p - aabb_min) / vs, 0.0, float(BRICK_DIM) - 1e-4);
    int3   vi = min(int3(floor(lp)), int3(BRICK_DIM - 1));
    float3 fr = lp - float3(vi);
    uint   bd = BRICK_DIM + 1u;
    float4 acc = float4(0, 0, 0, 0);
    for (uint c = 0u; c < 8u; c++)
    {
        uint ix = c & 1u, iy = (c >> 1u) & 1u, iz = (c >> 2u) & 1u;
        uint vox = (uint(vi.x) + ix) + bd * ((uint(vi.y) + iy) + bd * (uint(vi.z) + iz));
        uint pk = weight_atlas.Load((brick_offset + vox * 8u) >> 1u);
        float4 w4 = float4(pk & 0xFFu, (pk >> 8u) & 0xFFu, (pk >> 16u) & 0xFFu, (pk >> 24u) & 0xFFu) / 255.0;
        float w = (ix == 0u ? (1.0 - fr.x) : fr.x) * (iy == 0u ? (1.0 - fr.y) : fr.y) * (iz == 0u ? (1.0 - fr.z) : fr.z);
        acc += w * w4;
    }
    return acc;
}

Material terrainShade(float3 p, inout float3 n, float dist, float4 wl)
{
    float3 geoN = n;   // geometric (analytic) normal
    // Layer weights (grass, rock, snow, dirt) are BAKED (pre-mix) and passed in — the
    // slope/height/fBm/biome blend lives in the bake now, not here. We only do the
    // crisp, per-pixel triplanar texture sampling, weighted by these.
    float total = wl.x + wl.y + wl.z + wl.w + 1e-4;

    // Triplanar projection weights (sharpened so each plane dominates its axis).
    float3 tw = pow(abs(n), 4.0);
    tw /= (tw.x + tw.y + tw.z + 1e-5);

    float tiling = pc.thr1.z;
    // Distance LOD: texel/world rate ≈ tiling·texSize (1K), pixel world footprint
    // ≈ dist/screenH. texSize/screenH ≈ 1 → lod ≈ log2(dist·tiling), biased ×2 to
    // favour higher mips (less shimmer). Refined to a proper cone in A-cut3.
    float lod = max(0.0, log2(dist * tiling * 2.0 + 1e-3));

    // Fallback colours / roughness when a layer has no texture bound.
    float4 fbAlb[4] = {
        float4(0.13, 0.34, 0.10, 1),
        float4(0.30, 0.28, 0.26, 1),
        float4(0.90, 0.93, 0.97, 1),
        float4(0.26, 0.19, 0.12, 1),
    };
    float fbRough[4] = { 0.85, 0.90, 0.50, 0.88 };

    SamplerState sA = bindlessSamplers[SAMPLER_ANISO];

    float3 albedo = float3(0, 0, 0);
    float rough = 0.0;
    // Per-plane tangent-space normals, layer-blended (whiteout-blended below).
    float3 tnX = float3(0, 0, 0), tnY = float3(0, 0, 0), tnZ = float3(0, 0, 0);
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        float wi = wl[i] / total;
        if (wi < 0.004)
            continue;   // baked weight ~0 → skip this layer's samples entirely
        albedo += triSample(pc.idxAlbedo[i], p, tw, tiling, lod, fbAlb[i]).rgb * wi;
        // ORM packing: R=AO, G=Roughness, B=Metallic.
        float rg = triSample(pc.idxOrm[i], p, tw, tiling, lod, float4(fbRough[i].xxx, 1)).g;
        rough += rg * wi;

        // Tangent normals per plane (flat fallback when the layer has no normal map).
        if (pc.idxNormal[i] >= 0)
        {
            Texture2D tn = bindlessTextures[pc.idxNormal[i]];
            tnX += unpackNormalBC5(tn.SampleLevel(sA, p.zy * tiling, lod).rg) * wi;
            tnY += unpackNormalBC5(tn.SampleLevel(sA, p.xz * tiling, lod).rg) * wi;
            tnZ += unpackNormalBC5(tn.SampleLevel(sA, p.xy * tiling, lod).rg) * wi;
        }
        else
        {
            float3 flat = float3(0, 0, 1) * wi;
            tnX += flat;
            tnY += flat;
            tnZ += flat;
        }
    }

    // Triplanar normal mapping — Whiteout blend (bgolus): reorient each plane's
    // tangent normal by the geometric normal, swizzle to world axes, blend by `tw`.
    // Bit 7 disables it (A/B compare against the pure analytic normal).
    bool haveNrm = max(max(pc.idxNormal[0], pc.idxNormal[1]),
                       max(pc.idxNormal[2], pc.idxNormal[3])) >= 0;
    if (haveNrm && (pc.debug_flags & 128u) == 0u)
    {
        // Out-of-plane component must carry the SIGN of the geometric normal's axis
        // (abs on the tangent's z, which is ≥0), else faces pointing toward −X/−Z get
        // their normal flipped → inverted shading / AO / IBL on those slopes.
        tnX = float3(tnX.xy + geoN.zy, abs(tnX.z) * geoN.x);
        tnY = float3(tnY.xy + geoN.xz, abs(tnY.z) * geoN.y);
        tnZ = float3(tnZ.xy + geoN.xy, abs(tnZ.z) * geoN.z);
        n = normalize(tnX.zyx * tw.x + tnY.xzy * tw.y + tnZ.xyz * tw.z);
    }

    Material m;
    m.albedo = albedo;
    m.roughness = rough;
    m.metalness = 0.0;
    return m;
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

[shader("vertex")]
float4 vsMain(uint vid: SV_VertexID) : SV_Position
{
    float2 uv = float2((vid << 1u) & 2u, vid & 2u);
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

[shader("fragment")]
GBufferOut fsMain(float4 svPos: SV_Position)
{
    float2 screenSize = g_common.GetScreenSize();
    float2 uv = svPos.xy / screenSize;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float4 wFar = mul(g_common.invViewProjection, float4(ndc, 0.0, 1.0));
    wFar /= wFar.w;

    float3 ro = g_common.GetCameraPos();
    float3 rd = normalize(wFar.xyz - ro);

    float far = g_common.GetFarPlane();

#if SVO_DEBUG
    // ── Debug bit 4: analytic ground-truth (ignore bricks & tree entirely) ────
    // Sphere-traces the CSG of the primitives directly. If THIS is clean but the
    // normal SVO render is not, the bug is in brick bake/management or traversal,
    // not the scene data.
    if ((pc.debug_flags & 16u) != 0u)
    {
        float t = 0.1;
        [loop]
        for (int i = 0; i < 512; i++)
        {
            float3 p = ro + rd * t;
            uint m;
            float d = gtScene(p, m);
            if (d < 0.002)
            {
                float3 e = float3(0.002, 0.0, 0.0);
                uint mm;
                float3 n = normalize(float3(
                    gtScene(p + e.xyy, mm) - gtScene(p - e.xyy, mm),
                    gtScene(p + e.yxy, mm) - gtScene(p - e.yxy, mm),
                    gtScene(p + e.yyx, mm) - gtScene(p - e.yyx, mm)));
                Material mat = kMaterials[min(m, 7u)];
                float4 clip = mul(g_common.viewProjection, float4(p, 1.0));
                float nd = clip.z / clip.w;
                GBufferOut o;
                o.albedo = float4(mat.albedo, 1.0);
                o.normal = encodeOctahedral(n);
                o.roughMetal = float2(mat.roughness, mat.metalness);
                o.emissive = float3(0, 0, 0);
                o.hizDepth = nd;
                o.depth = nd;
                return o;
            }
            t += max(d * 0.9, 0.005);
            if (t > far)
                break;
        }
        discard;
    }
#endif // SVO_DEBUG

    float t = 0.1;
    bool hit = false;
    float3 hitPos = ro;
    uint hit_mat = 0u;
    float hit_vs = 0.01;
    float3 hit_grad = float3(0, 1, 0);
    int steps = 0;

    LeafRef lf;
    lf.aabb_min = float3(0, 0, 0);
    lf.aabb_max = float3(0, 0, 0);
    bool haveLeaf = false;

    [loop]
    for (int i = 0; i < 512; i++)
    {
        float3 p = ro + rd * t;
        steps++;

        // Look up the leaf ONLY when the ray has crossed into a new cell; otherwise reuse
        // the cached leaf → lookups drop from O(steps) to O(cells).
        if (!haveLeaf || any(p < lf.aabb_min) || any(p > lf.aabb_max))
        {
            lf = clipmapLookup(p);
            haveLeaf = true;

            if (!lf.inside)
                break; // ray left the tree → no hit

            if (!lf.found)
            {
                // Empty cell: jump exactly to its exit (no overshoot), then re-descend.
                t += max(boxExit(p, rd, lf.aabb_min, lf.aabb_max), 1e-3) + 1e-3;
                haveLeaf = false;
                if (t > far)
                    break;
                continue;
            }
        }

        // Surface leaf — distance-only reconstruct (cheap: no normal/material decode).
        float sdf = reconstructDist(lf, p);

        // Inter-level cross-fade: near this level's window edge, blend the marched field
        // toward the next coarser level. Makes the surface position continuous across the
        // resolution change (no static seam) AND eases the temporal pop when the window
        // shifts. Double-sample only inside the thin band where both levels have a brick.
        float blend = 0.0;
        LeafRef lfc = lf;
        if (lf.level + 1u < pc.num_levels)
        {
            blend = levelBlend(p, lf.level);
            if (blend > 0.001)
            {
                lfc = lookupAtLevel(p, lf.level + 1u);
                if (lfc.found) sdf = lerp(sdf, reconstructDist(lfc, p), blend);
                else           blend = 0.0;
            }
        }

        // Accept hit when within a small fraction of this leaf's voxel of the surface,
        // then do the FULL reconstruct ONCE (normal + material) and refine to the
        // zero-crossing along the gradient (removes the threshold bulge).
        if (sdf < lf.vs * 0.08)
        {
            BrickSample s = reconstructBrick(lf, p);
            if (blend > 0.001)
            {
                BrickSample sc = reconstructBrick(lfc, p);
                s.sdf  = lerp(s.sdf,  sc.sdf,  blend);
                s.grad = lerp(s.grad, sc.grad, blend);
            }
            float gdotr = dot(normalize(s.grad), rd);
            float back = (abs(gdotr) > 1e-3) ? (s.sdf / gdotr) : 0.0; // signed dist along ray
            hit = true;
            hitPos = p - rd * clamp(back, -lf.vs, lf.vs);
            hit_mat = s.mat_id;
            hit_vs = lf.vs;
            hit_grad = s.grad;
            break;
        }

        // Sphere-march, but never step past this leaf's exit (the top-of-loop cell test
        // then re-descends at the next cell — avoids skipping a finer neighbour's surface).
        // Floor the step at ~one pixel's world footprint (∝ distance): at grazing angles
        // near the horizon `sdf` stays tiny so `sdf*0.8` crawls and exhausts the 512-step
        // budget → a miss line. Detail finer than a pixel is invisible anyway, so a
        // distance-proportional minimum step kills the crawl (the overshoot is sub-pixel and
        // the hit's gradient back-step corrects it).
        float adv = max(max(sdf * 0.8, lf.vs * 0.02), t * 6e-4);
        float exit = boxExit(p, rd, lf.aabb_min, lf.aabb_max) + 1e-3;
        t += min(adv, exit);
        if (t > far)
            break;
    }

    if (!hit)
        discard;

    // Normal straight from the brick. The bake stores the TRUE combined-CSG normal per
    // voxel (each operand's own normal, selected by the active min/max term — terrain
    // analytic, a subtract's inward −∇empty, a solid prim's ∇), so it is correct on
    // carved surfaces and elsewhere. No per-case re-evaluation here.
    float3 n = normalize(hit_grad);

    float4 clip = mul(g_common.viewProjection, float4(hitPos, 1.0));
    float ndcDepth = clip.z / clip.w;

#if SVO_NO_MATERIAL
    // Material/texturing compiled out (perf test): flat albedo, brick normal kept.
    Material mat;
    mat.albedo = float3(0.5, 0.5, 0.5);
    mat.roughness = 0.9;
    mat.metalness = 0.0;
#else
    // Terrain → triplanar texturing weighted by the BAKED pre-mix weights (read here);
    // everything else → the brick material blend.
    float hitDist = length(hitPos - g_common.GetCameraPos());
    Material mat;
    if (pc.terrain_on != 0u && hit_mat == 5u) // TERRAIN_MAT
    {
        float4 wl = loadWeights(lf.brick_offset, lf.aabb_min, lf.vs, hitPos);
        mat = terrainShade(hitPos, n, hitDist, wl);
    }
    else
        mat = sampleMaterial(hitPos);
#endif

    GBufferOut o;
    o.albedo = float4(mat.albedo, 1.0);

#if SVO_DEBUG
    // bit 6 – residual error: |analytic SDF at the brick-hit point|.
    // A correct brick puts the surface where the analytic SDF is ~0. Red = the
    // brick surface is misplaced (stale/garbage/hazard). Green = brick agrees.
    if ((pc.debug_flags & 64u) != 0u)
    {
        uint mm;
        float err = abs(gtScene(hitPos, mm));
        float h = saturate(err / max(hit_vs, 1e-4)); // normalized to one voxel
        o.albedo = float4(h, 1.0 - h, 0.0, 1.0);
        o.normal = encodeOctahedral(n);
        o.roughMetal = float2(1.0, 0.0);
        o.emissive = float3(0, 0, 0);
        o.hizDepth = ndcDepth;
        o.depth = ndcDepth;
        return o;
    }
#endif

    // bit 0 – SVO depth colors (approximate via node size → voxel size)
    if ((pc.debug_flags & 1u) != 0u)
    {
        float depth_t = saturate(log2(hit_vs / 0.01) / float(MAX_SVO_DEPTH));
        float3 c = lerp(float3(0.05, 0.20, 1.0), float3(0.90, 0.25, 0.05), depth_t);
        o.albedo = float4(c, 1.0);
    }

    // bit 1 – node AABB wireframe (highlight voxel-aligned edges)
    if ((pc.debug_flags & 2u) != 0u)
    {
        float3 bf = frac(hitPos / hit_vs / float(BRICK_DIM));
        float3 be = min(bf, 1.0 - bf);
        float edge = min(be.x, min(be.y, be.z));
        float t2 = 1.0 - smoothstep(0.0, 0.04, edge);
        o.albedo = lerp(o.albedo, float4(0.0, 0.0, 0.0, 1.0), t2);
    }

    // bit 2 – error heatmap (higher voxel size = warmer = coarser)
    if ((pc.debug_flags & 4u) != 0u)
    {
        float heat = saturate(log2(hit_vs * 100.0 + 1.0) * 0.4);
        o.albedo = float4(heat, 1.0 - heat, 0.0, 1.0);
    }

    // bit 3 – step count heat
    if ((pc.debug_flags & 8u) != 0u)
    {
        float heat = saturate(float(steps) / 256.0);
        o.albedo = float4(heat, 1.0 - heat, 0.0, 1.0);
    }

    o.normal = encodeOctahedral(n);
    o.roughMetal = float2(mat.roughness, mat.metalness);
    o.emissive = float3(0.0, 0.0, 0.0);
    o.hizDepth = ndcDepth;
    o.depth = ndcDepth;
    return o;
}
