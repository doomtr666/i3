#include "common.slangh"

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
[[vk::binding(1, 0)]] ByteAddressBuffer              sdf_atlas;
[[vk::binding(2, 0)]] ByteAddressBuffer              mat_atlas;
[[vk::binding(3, 0)]] StructuredBuffer<GpuPrimitive> prims;

// ─── Push constants ───────────────────────────────────────────────────────────
struct SvoPC { uint node_count; uint debug_flags; uint prim_count; uint _pad; };
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

// ─── Atlas helpers (shared with bake shader) ──────────────────────────────────
float loadSdfVoxel(uint base_bytes, uint vox, float half_diag)
{
    uint dw  = vox / 4u;
    uint bp  = vox % 4u;
    uint pk  = sdf_atlas.Load(base_bytes + dw * 4u);
    uint bv  = (pk >> (bp * 8u)) & 0xFFu;
    return (float(bv) / 127.5 - 1.0) * half_diag;
}

uint loadMatVoxel(uint base_bytes, uint vox)
{
    uint dw = vox / 4u;
    uint bp = vox % 4u;
    uint pk = mat_atlas.Load(base_bytes + dw * 4u);
    return (pk >> (bp * 8u)) & 0xFFu;
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
            float  hd   = 3.0 * vs;   // BAND_VOXELS — must match BAND_VOXELS in lib.rs

            // Empty (culled) leaf — no surface here. Return its bounds so the tracer
            // skips exactly to the cell exit instead of a fixed coarse step.
            if (node.brick_offset == ~0u)
                return { INF, 0u, vs, hd, false, true, node.aabb_min, node.aabb_max, float3(0,1,0) };

            // Debug bit 5: traverse the tree but sample analytic SDF (bypass atlas).
            if ((pc.debug_flags & 32u) != 0u)
            {
                uint m;
                float dsdf = gtScene(p, m);
                return { dsdf, m, vs, hd, true, true, node.aabb_min, node.aabb_max, float3(0,1,0) };
            }

            // Trilinear lookup inside 9³ brick
            float3 lp   = (p - node.aabb_min) / vs - 0.5;
            lp          = clamp(lp, 0.0, float(BRICK_DIM) - 1e-4);
            int3   vi   = min(int3(floor(lp)), int3(BRICK_DIM - 1));
            float3 fr   = lp - float3(vi);
            uint   bd   = BRICK_DIM + 1u;

            float s000 = loadSdfVoxel(node.brick_offset, uint(vi.x  ) + bd*(uint(vi.y  ) + bd*uint(vi.z  )), hd);
            float s100 = loadSdfVoxel(node.brick_offset, uint(vi.x+1) + bd*(uint(vi.y  ) + bd*uint(vi.z  )), hd);
            float s010 = loadSdfVoxel(node.brick_offset, uint(vi.x  ) + bd*(uint(vi.y+1) + bd*uint(vi.z  )), hd);
            float s110 = loadSdfVoxel(node.brick_offset, uint(vi.x+1) + bd*(uint(vi.y+1) + bd*uint(vi.z  )), hd);
            float s001 = loadSdfVoxel(node.brick_offset, uint(vi.x  ) + bd*(uint(vi.y  ) + bd*uint(vi.z+1)), hd);
            float s101 = loadSdfVoxel(node.brick_offset, uint(vi.x+1) + bd*(uint(vi.y  ) + bd*uint(vi.z+1)), hd);
            float s011 = loadSdfVoxel(node.brick_offset, uint(vi.x  ) + bd*(uint(vi.y+1) + bd*uint(vi.z+1)), hd);
            float s111 = loadSdfVoxel(node.brick_offset, uint(vi.x+1) + bd*(uint(vi.y+1) + bd*uint(vi.z+1)), hd);

            float sdf = lerp(
                lerp(lerp(s000,s100,fr.x), lerp(s010,s110,fr.x), fr.y),
                lerp(lerp(s001,s101,fr.x), lerp(s011,s111,fr.x), fr.y),
                fr.z);

            // Analytic gradient of the trilinear field (smooth within the cell,
            // no finite-difference quantisation noise, no cross-brick sampling).
            float gx = lerp(lerp(s100-s000, s110-s010, fr.y), lerp(s101-s001, s111-s011, fr.y), fr.z);
            float gy = lerp(lerp(s010-s000, s110-s100, fr.x), lerp(s011-s001, s111-s101, fr.x), fr.z);
            float gz = lerp(lerp(s001-s000, s101-s100, fr.x), lerp(s011-s010, s111-s110, fr.x), fr.y);
            float3 grad = float3(gx, gy, gz);   // unnormalized; / vs is constant

            // Nearest-sample material
            int3   vi2   = clamp(vi + int3(fr > 0.5), 0, int3(BRICK_DIM - 1));
            uint   vf    = uint(vi2.x) + bd*(uint(vi2.y) + bd*uint(vi2.z));
            uint   mat   = loadMatVoxel(node.brick_offset, vf);

            return { sdf, mat, vs, hd, true, true, node.aabb_min, node.aabb_max, grad };
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

// ─── Normal via finite differences ───────────────────────────────────────────
float3 calcNormal(float3 p, float eps)
{
    float2 e = float2(eps, 0.0);
    SvoSample sx0 = sampleSvo(p - e.xyy); SvoSample sx1 = sampleSvo(p + e.xyy);
    SvoSample sy0 = sampleSvo(p - e.yxy); SvoSample sy1 = sampleSvo(p + e.yxy);
    SvoSample sz0 = sampleSvo(p - e.yyx); SvoSample sz1 = sampleSvo(p + e.yyx);
    float3 g = float3(
        (sx1.found ? sx1.sdf : 0.0) - (sx0.found ? sx0.sdf : 0.0),
        (sy1.found ? sy1.sdf : 0.0) - (sy0.found ? sy0.sdf : 0.0),
        (sz1.found ? sz1.sdf : 0.0) - (sz0.found ? sz0.sdf : 0.0)
    );
    float len = length(g);
    return len > 1e-5 ? g / len : float3(0, 1, 0);
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
            float gdotr = dot(normalize(s.grad / s.voxel_size), rd);
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

    // Smooth normal from the brick's analytic trilinear gradient.
    float3 n = normalize(hit_grad);

    float4 clip     = mul(g_common.viewProjection, float4(hitPos, 1.0));
    float  ndcDepth = clip.z / clip.w;

    Material mat = kMaterials[min(hit_mat, 7u)];

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
