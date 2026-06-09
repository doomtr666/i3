#include "common.slangh"
#include "noise.slangh"   // terrainDensity — for analytic debug (bit 4/5) parity with the bake

// ─── Set 0: SVO buffers ───────────────────────────────────────────────────────
struct GpuSvoNode
{
    float3 aabb_min;
    uint   first_child;   // u32::MAX = leaf
    float3 aabb_max;
    uint   brick_offset;  // byte offset in atlas; u32::MAX = not baked
};

struct GpuPrimitive
{
    float4 row0; float4 row1; float4 row2;
    float  inv_scale;
    float  data0; float data1; float data2;
    uint   prim_type; uint subtract; uint material_id; uint _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuSvoNode>   node_pool;
[[vk::binding(1, 0)]] ByteAddressBuffer              geom_atlas;   // 8 bytes/voxel
[[vk::binding(2, 0)]] StructuredBuffer<GpuPrimitive> prims;

// ─── Push constants ───────────────────────────────────────────────────────────
struct SvoPC {
    uint node_count; uint debug_flags; uint prim_count; uint terrain_on;
    // Terrain layers (0=grass 1=rock 2=snow 3=dirt): bindless texture indices (-1 = none)
    int4 idxAlbedo; int4 idxNormal; int4 idxOrm;
    float4 thr0;   // snowLo, snowHi, slopeLo, slopeHi
    float4 thr1;   // dirtLo, dirtHi, tiling, jitterAmp
};
[[vk::push_constant]] SvoPC pc;

// ─── Analytic SDF (ground-truth, bypasses bricks) ─────────────────────────────
float gtSphere(float3 p, float r)   { return length(p) - r; }
float gtBox(float3 p, float3 he)    { float3 q = abs(p) - he; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float gtCapsule(float3 p, float hh, float r)  { float cy = clamp(p.y, -hh, hh); return length(p - float3(0, cy, 0)) - r; }
float gtCylinder(float3 p, float hh, float r) { float dxz = length(p.xz) - r; float dy = abs(p.y) - hh; return length(max(float2(dxz, dy), 0.0)) + min(max(dxz, dy), 0.0); }
float gtTorus(float3 p, float R, float r)     { float2 q = float2(length(p.xz) - R, p.y); return length(q) - r; }

float gtEvalPrim(float3 pw, GpuPrimitive prim)
{
    float3 pc2 = pw - float3(prim.row0.w, prim.row1.w, prim.row2.w);
    float3 pr  = float3(dot(prim.row0.xyz, pc2), dot(prim.row1.xyz, pc2), dot(prim.row2.xyz, pc2));
    float3 pl  = pr * prim.inv_scale;
    float  d;
    if      (prim.prim_type == 0u) d = gtSphere(pl, prim.data0);
    else if (prim.prim_type == 1u) d = gtBox(pl, float3(prim.data0, prim.data1, prim.data2));
    else if (prim.prim_type == 2u) d = gtCapsule(pl, prim.data0, prim.data1);
    else if (prim.prim_type == 3u) d = gtCylinder(pl, prim.data0, prim.data1);
    else                            d = gtTorus(pl, prim.data0, prim.data1);
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
        if (prims[i].subtract != 0u) { empty = min(empty, d); }
        else { solid = min(solid, d); if (d < mind) { mind = d; mat_id = prims[i].material_id; } }
    }
    if (pc.terrain_on != 0u)
    {
        float td = terrainDensity(p, float3(0, 0, 0)).v;
        if (td < solid) { solid = td; mat_id = 5u; }   // TERRAIN_MAT
    }
    return max(solid, -empty);
}

// ─── Constants ────────────────────────────────────────────────────────────────
static const uint  BRICK_DIM    = 8u;
// MUST be ≥ the tree's max_depth (SvoTree::new) or deep nodes become unreachable
// during traversal → the finest-LOD cubes stop rendering (holes when you approach).
static const uint  MAX_SVO_DEPTH = 18u;
static const float INF           = 1e10;

// ─── Materials (same as clipmap) ─────────────────────────────────────────────
struct Material { float3 albedo; float roughness; float metalness; };
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
struct Voxel { float dist; float3 normal; uint mat; };

float snorm16decode(uint u) { int s = (int(u) << 16) >> 16; return max(float(s) / 32767.0, -1.0); }

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
    v.dist   = snorm16decode(w.x & 0xFFFFu) * band;   // ±BAND·vs world units
    v.normal = octDecode(float2(snorm16decode((w.x >> 16u) & 0xFFFFu),
                                snorm16decode( w.y         & 0xFFFFu)));
    v.mat    = (w.y >> 16u) & 0xFFu;
    return v;
}

// ─── SVO tree descent ─────────────────────────────────────────────────────────

// found  = p is in a baked (surface) leaf, sdf/mat/vs valid.
// inside = p is inside the tree at all (false ⇒ outside root ⇒ ray left the scene).
// cmin/cmax = bounds of the cell p is in (empty leaf or outside), for exact skipping.
struct SvoSample {
    float sdf; uint mat_id; float voxel_size; float half_diag;
    bool found; bool inside; float3 cmin; float3 cmax; float3 grad;
};

SvoSample sampleSvo(float3 p)
{
    float3 rmin = node_pool[0].aabb_min, rmax = node_pool[0].aabb_max;
    if (pc.node_count == 0u)
        return { INF, 0u, 0.0, 0.0, false, false, rmin, rmax, float3(0,1,0) };

    uint idx = 0u;
    [loop]
    for (uint d = 0u; d <= MAX_SVO_DEPTH; d++)
    {
        if (idx >= pc.node_count) break;
        GpuSvoNode node = node_pool[idx];

        if (any(p < node.aabb_min) || any(p > node.aabb_max))
            return { INF, 0u, 0.0, 0.0, false, false, node.aabb_min, node.aabb_max, float3(0,1,0) };

        if (node.first_child == ~0u)
        {
            float3 sz   = node.aabb_max - node.aabb_min;
            float  vs   = sz.x / float(BRICK_DIM);

            // Empty (culled) leaf — no surface here. Return its bounds so the tracer
            // skips exactly to the cell exit instead of a fixed coarse step.
            if (node.brick_offset == ~0u)
                return { INF, 0u, vs, 0.0, false, true, node.aabb_min, node.aabb_max, float3(0,1,0) };

            // Debug bit 5: traverse the tree but sample analytic SDF (bypass atlas).
            if ((pc.debug_flags & 32u) != 0u)
            {
                uint m;
                float dsdf = gtScene(p, m);
                return { dsdf, m, vs, 0.0, true, true, node.aabb_min, node.aabb_max, float3(0,1,0) };
            }

            // ── Gradient-augmented (tangent-plane) reconstruction ─────────────
            // Each of the 8 corner voxels stores distance + analytic normal, so it
            // defines a tangent plane d_i(p) = d_i + n_i·(p − corner_i). Trilinearly
            // blending these gives a far better distance estimate than blending the
            // values alone: exact for planes, Hermite-smooth for curves, and the
            // blended field stays 1-Lipschitz (|Σ w_i n_i| ≤ 1) → safe to march.
            float3 lp   = (p - node.aabb_min) / vs - 0.5;
            lp          = clamp(lp, 0.0, float(BRICK_DIM) - 1e-4);
            int3   vi   = min(int3(floor(lp)), int3(BRICK_DIM - 1));
            float3 fr   = lp - float3(vi);
            uint   bd   = BRICK_DIM + 1u;

            // World position of the vi corner's voxel centre.
            float3 c000 = node.aabb_min + (float3(vi) + 0.5) * vs;
            float  band = BAND * vs;

            float  dacc = 0.0;     // blended tangent-plane distance
            float3 nacc = float3(0,0,0);
            uint   mat  = 0u;
            float  mbest = -1.0;
            [unroll] for (uint c = 0u; c < 8u; c++)
            {
                uint ix = c & 1u, iy = (c >> 1u) & 1u, iz = (c >> 2u) & 1u;
                Voxel v = loadVoxel(node.brick_offset, (uint(vi.x)+ix) + bd*((uint(vi.y)+iy) + bd*(uint(vi.z)+iz)), band);
                float wx = (ix == 0u) ? (1.0 - fr.x) : fr.x;
                float wy = (iy == 0u) ? (1.0 - fr.y) : fr.y;
                float wz = (iz == 0u) ? (1.0 - fr.z) : fr.z;
                float w  = wx * wy * wz;
                float3 corner = c000 + float3(ix, iy, iz) * vs;
                dacc += w * (v.dist + dot(v.normal, p - corner));   // tangent plane
                nacc += w * v.normal;
                if (w > mbest) { mbest = w; mat = v.mat; }          // material = dominant corner
            }

            return { dacc, mat, vs, 0.0, true, true, node.aabb_min, node.aabb_max, nacc };
        }

        // descend to octant
        float3 mid    = (node.aabb_min + node.aabb_max) * 0.5;
        uint   octant = (p.x > mid.x ? 1u : 0u)
                      | (p.y > mid.y ? 2u : 0u)
                      | (p.z > mid.z ? 4u : 0u);
        idx = node.first_child + octant;
    }
    return { INF, 0u, 0.0, 0.0, false, false, rmin, rmax, float3(0,1,0) };
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
    uint idx = 0u;
    [loop]
    for (uint d = 0u; d <= MAX_SVO_DEPTH; d++)
    {
        if (idx >= pc.node_count) break;
        GpuSvoNode node = node_pool[idx];
        if (any(p < node.aabb_min) || any(p > node.aabb_max)) break;

        if (node.first_child == ~0u)
        {
            if (node.brick_offset == ~0u) break;
            float3 sz = node.aabb_max - node.aabb_min;
            float  vs = sz.x / float(BRICK_DIM);
            float3 lp = (p - node.aabb_min) / vs - 0.5;
            lp = clamp(lp, 0.0, float(BRICK_DIM) - 1e-4);
            int3   vi = min(int3(floor(lp)), int3(BRICK_DIM - 1));
            float3 fr = lp - float3(vi);
            uint   bd = BRICK_DIM + 1u;
            float  band = BAND * vs;

            Material result; result.albedo = float3(0,0,0); result.roughness = 0.0; result.metalness = 0.0;
            float total = 0.0;
            for (uint iz = 0u; iz <= 1u; iz++)
            for (uint iy = 0u; iy <= 1u; iy++)
            for (uint ix = 0u; ix <= 1u; ix++)
            {
                Voxel v   = loadVoxel(node.brick_offset, uint(vi.x+ix) + bd*(uint(vi.y+iy) + bd*uint(vi.z+iz)), band);
                float wx  = (ix == 0u) ? (1.0 - fr.x) : fr.x;
                float wy  = (iy == 0u) ? (1.0 - fr.y) : fr.y;
                float wz  = (iz == 0u) ? (1.0 - fr.z) : fr.z;
                float w   = wx * wy * wz * exp(-max(v.dist, 0.0) / vs);
                Material m = kMaterials[min(v.mat, 7u)];
                result.albedo    += m.albedo    * w;
                result.roughness += m.roughness * w;
                result.metalness += m.metalness * w;
                total            += w;
            }
            if (total > 1e-6) { result.albedo /= total; result.roughness /= total; result.metalness /= total; return result; }
            break;
        }
        float3 mid = (node.aabb_min + node.aabb_max) * 0.5;
        uint   oct = (p.x > mid.x ? 1u : 0u) | (p.y > mid.y ? 2u : 0u) | (p.z > mid.z ? 4u : 0u);
        idx = node.first_child + oct;
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
    if (idx < 0) return fallback;
    // SAMPLER_ANISO is the only Repeat-addressed sampler — required since the
    // triplanar UVs (world·tiling) run far outside [0,1]; ClampToEdge would smear
    // the border texel into long streaks.
    SamplerState s = bindlessSamplers[SAMPLER_ANISO];
    Texture2D    t = bindlessTextures[idx];
    float4 cx = t.SampleLevel(s, p.zy * tiling, lod);   // X-facing plane
    float4 cy = t.SampleLevel(s, p.xz * tiling, lod);   // Y-facing plane
    float4 cz = t.SampleLevel(s, p.xy * tiling, lod);   // Z-facing plane
    return cx * w.x + cy * w.y + cz * w.z;
}

// Unpack a BC5_UNORM (2-channel) tangent-space normal: RG → [-1,1], reconstruct Z.
float3 unpackNormalBC5(float2 rg)
{
    float2 xy = rg * 2.0 - 1.0;
    return float3(xy, sqrt(saturate(1.0 - dot(xy, xy))));
}

Material terrainShade(float3 p, inout float3 n, float dist)
{
    float3 geoN = n;                                   // geometric (analytic) normal
    float slope = saturate(n.y);                       // 1 = flat, 0 = vertical
    float jit   = fbm3(p * 0.06, float3(0,0,0), 7, 3, 2.0, 0.5).v;
    float h     = p.y + jit * pc.thr1.w;               // jittered height for the bands

    // Layer weights — order: 0=grass, 1=rock, 2=snow, 3=dirt.
    float wRock  = 1.0 - smoothstep(pc.thr0.z, pc.thr0.w, slope);                 // steep → rock
    float wSnow  = smoothstep(pc.thr0.x, pc.thr0.y, h) * smoothstep(0.55, 0.78, slope); // high & flat → snow
    float wDirt  = 1.0 - smoothstep(pc.thr1.x, pc.thr1.y, h);                     // low → dirt
    float wGrass = saturate(1.0 - wRock - wSnow - wDirt);
    float wl[4]  = { wGrass, wRock, wSnow, wDirt };
    float total  = wGrass + wRock + wSnow + wDirt + 1e-4;

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
        float4(0.13, 0.34, 0.10, 1), float4(0.30, 0.28, 0.26, 1),
        float4(0.90, 0.93, 0.97, 1), float4(0.26, 0.19, 0.12, 1),
    };
    float fbRough[4] = { 0.85, 0.90, 0.50, 0.88 };

    SamplerState sA = bindlessSamplers[SAMPLER_ANISO];

    float3 albedo = float3(0, 0, 0);
    float  rough  = 0.0;
    // Per-plane tangent-space normals, layer-blended (whiteout-blended below).
    float3 tnX = float3(0, 0, 0), tnY = float3(0, 0, 0), tnZ = float3(0, 0, 0);
    [unroll] for (int i = 0; i < 4; ++i) {
        float wi = wl[i] / total;
        albedo += triSample(pc.idxAlbedo[i], p, tw, tiling, lod, fbAlb[i]).rgb * wi;
        // ORM packing: R=AO, G=Roughness, B=Metallic.
        float rg = triSample(pc.idxOrm[i], p, tw, tiling, lod, float4(fbRough[i].xxx, 1)).g;
        rough  += rg * wi;

        // Tangent normals per plane (flat fallback when the layer has no normal map).
        if (pc.idxNormal[i] >= 0) {
            Texture2D tn = bindlessTextures[pc.idxNormal[i]];
            tnX += unpackNormalBC5(tn.SampleLevel(sA, p.zy * tiling, lod).rg) * wi;
            tnY += unpackNormalBC5(tn.SampleLevel(sA, p.xz * tiling, lod).rg) * wi;
            tnZ += unpackNormalBC5(tn.SampleLevel(sA, p.xy * tiling, lod).rg) * wi;
        } else {
            float3 flat = float3(0, 0, 1) * wi;
            tnX += flat; tnY += flat; tnZ += flat;
        }
    }

    // Triplanar normal mapping — Whiteout blend (bgolus): reorient each plane's
    // tangent normal by the geometric normal, swizzle to world axes, blend by `tw`.
    // Bit 7 disables it (A/B compare against the pure analytic normal).
    bool haveNrm = max(max(pc.idxNormal[0], pc.idxNormal[1]),
                       max(pc.idxNormal[2], pc.idxNormal[3])) >= 0;
    if (haveNrm && (pc.debug_flags & 128u) == 0u) {
        // Out-of-plane component must carry the SIGN of the geometric normal's axis
        // (abs on the tangent's z, which is ≥0), else faces pointing toward −X/−Z get
        // their normal flipped → inverted shading / AO / IBL on those slopes.
        tnX = float3(tnX.xy + geoN.zy, abs(tnX.z) * geoN.x);
        tnY = float3(tnY.xy + geoN.xz, abs(tnY.z) * geoN.y);
        tnZ = float3(tnZ.xy + geoN.xy, abs(tnZ.z) * geoN.z);
        n = normalize(tnX.zyx * tw.x + tnY.xzy * tw.y + tnZ.xyz * tw.z);
    }

    Material m;
    m.albedo    = albedo;
    m.roughness = rough;
    m.metalness = 0.0;
    return m;
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

[shader("vertex")]
float4 vsMain(uint vid : SV_VertexID) : SV_Position
{
    float2 uv = float2((vid << 1u) & 2u, vid & 2u);
    return float4(uv * 2.0 - 1.0, 0.0, 1.0);
}

[shader("fragment")]
GBufferOut fsMain(float4 svPos : SV_Position)
{
    float2 screenSize = g_common.GetScreenSize();
    float2 uv  = svPos.xy / screenSize;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float4 wFar = mul(g_common.invViewProjection, float4(ndc, 0.0, 1.0));
    wFar /= wFar.w;

    float3 ro = g_common.GetCameraPos();
    float3 rd = normalize(wFar.xyz - ro);

    float  far     = g_common.GetFarPlane();

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
            if (d < 0.002) {
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
                o.hizDepth = nd; o.depth = nd;
                return o;
            }
            t += max(d * 0.9, 0.005);
            if (t > far) break;
        }
        discard;
    }

    float  t       = 0.1;
    bool   hit     = false;
    float3 hitPos  = ro;
    uint   hit_mat = 0u;
    float  hit_vs  = 0.01;
    float3 hit_grad = float3(0, 1, 0);
    int    steps   = 0;

    [loop]
    for (int i = 0; i < 512; i++)
    {
        float3 p = ro + rd * t;
        SvoSample s = sampleSvo(p);
        steps++;

        if (!s.inside)
            break;                       // ray left the tree → no hit

        if (!s.found)
        {
            // Empty cell: jump exactly to its exit (no overshoot, no wasted steps).
            t += max(boxExit(p, rd, s.cmin, s.cmax), 1e-3) + 1e-3;
            if (t > far) break;
            continue;
        }

        // Accept hit when within a small fraction of this leaf's voxel of the
        // surface, then refine to the zero-crossing along the gradient to remove
        // the threshold bulge (the trilinear field is locally ~linear).
        if (s.sdf < s.voxel_size * 0.08)
        {
            // s.grad is the blended analytic normal (already world-space direction).
            float gdotr = dot(normalize(s.grad), rd);
            float back  = (abs(gdotr) > 1e-3) ? (s.sdf / gdotr) : 0.0;  // signed dist to surface along ray
            hit     = true;
            hitPos  = p - rd * clamp(back, -s.voxel_size, s.voxel_size);
            hit_mat = s.mat_id;
            hit_vs  = s.voxel_size;
            hit_grad = s.grad;
            break;
        }

        // Sphere-march, but never step past this leaf's exit (avoids skipping a
        // surface that begins just inside a neighbouring, finer leaf).
        float adv = max(s.sdf * 0.8, s.voxel_size * 0.02);
        float exit = boxExit(p, rd, s.cmin, s.cmax) + 1e-3;
        t += min(adv, exit);
        if (t > far) break;
    }

    if (!hit) discard;

    // Normal straight from the brick. The bake stores the TRUE combined-CSG normal per
    // voxel (each operand's own normal, selected by the active min/max term — terrain
    // analytic, a subtract's inward −∇empty, a solid prim's ∇), so it is correct on
    // carved surfaces and elsewhere. No per-case re-evaluation here.
    float3 n = normalize(hit_grad);

    float4 clip     = mul(g_common.viewProjection, float4(hitPos, 1.0));
    float  ndcDepth = clip.z / clip.w;

    // Terrain → slope/height layer shading (also perturbs `n` via normal maps);
    // everything else → the brick material blend.
    float hitDist = length(hitPos - g_common.GetCameraPos());
    Material mat;
    if (pc.terrain_on != 0u && hit_mat == 5u)   // TERRAIN_MAT
        mat = terrainShade(hitPos, n, hitDist);
    else
        mat = sampleMaterial(hitPos);

    GBufferOut o;
    o.albedo = float4(mat.albedo, 1.0);

    // bit 6 – residual error: |analytic SDF at the brick-hit point|.
    // A correct brick puts the surface where the analytic SDF is ~0. Red = the
    // brick surface is misplaced (stale/garbage/hazard). Green = brick agrees.
    if ((pc.debug_flags & 64u) != 0u)
    {
        uint mm;
        float err = abs(gtScene(hitPos, mm));
        float h = saturate(err / max(hit_vs, 1e-4));   // normalized to one voxel
        o.albedo = float4(h, 1.0 - h, 0.0, 1.0);
        o.normal     = encodeOctahedral(n);
        o.roughMetal = float2(1.0, 0.0);
        o.emissive   = float3(0, 0, 0);
        o.hizDepth   = ndcDepth;
        o.depth      = ndcDepth;
        return o;
    }

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
        float3 bf   = frac(hitPos / hit_vs / float(BRICK_DIM));
        float3 be   = min(bf, 1.0 - bf);
        float  edge = min(be.x, min(be.y, be.z));
        float  t2   = 1.0 - smoothstep(0.0, 0.04, edge);
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

    o.normal     = encodeOctahedral(n);
    o.roughMetal = float2(mat.roughness, mat.metalness);
    o.emissive   = float3(0.0, 0.0, 0.0);
    o.hizDepth   = ndcDepth;
    o.depth      = ndcDepth;
    return o;
}
