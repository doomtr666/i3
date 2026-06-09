// SVO GPU bake — compute shader.
// One workgroup = one brick, [9,9,9] = 729 threads. Each thread bakes ONE voxel
// directly (8 bytes, aligned) — no LDS packing needed.
//
// Voxel layout (8 bytes, little-endian, matches lib.rs VOXEL_BYTES):
//   bytes 0-1  signed distance      snorm16 over ±BAND·voxel_size
//   bytes 2-3  octahedral normal X  snorm16
//   bytes 4-5  octahedral normal Y  snorm16
//   byte  6    material id          u8
//   byte  7    flags                u8
//
// Distance uses snorm16 over a band (not raw f16): pure-integer round-trip the
// toolchain reliably supports, and ~vs/8192 precision near the surface — finer
// than f16 there. Cells straddling the surface are within ±√3·vs ≪ BAND·vs, so
// the tangent-plane reconstruction is exact for them; far cells clamp to ±band
// (still a valid "far" distance for sphere-marching). BAND must match svo_render.
static const float BAND = 4.0;

#include "noise.slangh"   // perlin3 / fbm3 / terrainDensity (CPU-parity volumetric terrain)

static const uint TERRAIN_MAT = 5u;   // material id for terrain voxels (kMaterials[5])

struct GpuBrickJob
{
    float3 brick_world_min;
    float  voxel_size;
    uint   atlas_offset;   // byte offset into the geometry atlas
    // Three SCALAR pads, NOT uint3: a uint3 aligns to 16 in std430, which would
    // bump this struct to 48 bytes and desync from the 32-byte #[repr(C)] CPU
    // struct → every job but the first reads shifted fields. Scalars stay at 4.
    uint   _pad0;
    uint   _pad1;
    uint   _pad2;
};

struct GpuPrimitive
{
    float4 row0;
    float4 row1;
    float4 row2;
    float  inv_scale;
    float  data0;
    float  data1;
    float  data2;
    uint   prim_type;  // 0=Sphere 1=Box 2=Capsule 3=Cylinder 4=Torus
    uint   subtract;
    uint   material_id;
    uint   _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuBrickJob>   jobs;
[[vk::binding(1, 0)]] StructuredBuffer<GpuPrimitive>  prims;
[[vk::binding(2, 0)]] RWByteAddressBuffer              geom_atlas;

struct BakePC { uint prim_count; uint terrain_on; uint _pad1; uint _pad2; };
[[vk::push_constant]] BakePC pc;

static const uint PRIM_SPHERE = 0u, PRIM_BOX = 1u, PRIM_CAPSULE = 2u, PRIM_CYLINDER = 3u, PRIM_TORUS = 4u;

float sdfSphere(float3 p, float r)   { return length(p) - r; }
float sdfBox(float3 p, float3 he)    { float3 q = abs(p) - he; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float sdfCapsule(float3 p, float hh, float r)  { float cy = clamp(p.y, -hh, hh); return length(p - float3(0, cy, 0)) - r; }
float sdfCylinder(float3 p, float hh, float r) { float dxz = length(p.xz) - r; float dy = abs(p.y) - hh; return length(max(float2(dxz, dy), 0.0)) + min(max(dxz, dy), 0.0); }
float sdfTorus(float3 p, float R, float r)     { float2 q = float2(length(p.xz) - R, p.y); return length(q) - r; }

float evalPrimitive(float3 pw, GpuPrimitive prim)
{
    float3 pc2 = pw - float3(prim.row0.w, prim.row1.w, prim.row2.w);
    float3 pr  = float3(dot(prim.row0.xyz, pc2), dot(prim.row1.xyz, pc2), dot(prim.row2.xyz, pc2));
    float3 pl  = pr * prim.inv_scale;
    float  d;
    if      (prim.prim_type == PRIM_SPHERE)   d = sdfSphere(pl, prim.data0);
    else if (prim.prim_type == PRIM_BOX)      d = sdfBox(pl, float3(prim.data0, prim.data1, prim.data2));
    else if (prim.prim_type == PRIM_CAPSULE)  d = sdfCapsule(pl, prim.data0, prim.data1);
    else if (prim.prim_type == PRIM_CYLINDER) d = sdfCylinder(pl, prim.data0, prim.data1);
    else                                       d = sdfTorus(pl, prim.data0, prim.data1);
    return d / prim.inv_scale;
}

// Union of all SOLID (non-subtract) primitives. FD of this gives a solid prim's own
// surface normal — cheap, no graph evaluation.
float solidPrimSdf(float3 p)
{
    float solid = 1e10;
    for (uint i = 0u; i < pc.prim_count; i++)
        if (prims[i].subtract == 0u) solid = min(solid, evalPrimitive(p, prims[i]));
    return solid;
}

// Union of all SUBTRACT primitives. FD of this (negated) gives a carved surface's
// normal — pointing INTO the cavity. Each CSG operand thus carries its own correct
// normal; csMain's min/max selection picks the active one, so the combined normal is
// correct everywhere with no per-case overrides.
float emptySdf(float3 p)
{
    float empty = 1e10;
    for (uint i = 0u; i < pc.prim_count; i++)
        if (prims[i].subtract != 0u) empty = min(empty, evalPrimitive(p, prims[i]));
    return empty;
}

// Primitive CSG, returning `solid` and `empty` SEPARATELY (+ nearest-solid material)
// so csMain can fold the terrain into `solid` (so subtracts carve it) and decide the
// normal source. No terrain here → no graph evaluation.
void sceneCSG(float3 p, out float solid, out float empty, out uint mat_id)
{
    solid = 1e10; empty = 1e10; float mind = 1e10; mat_id = 0u;
    for (uint i = 0u; i < pc.prim_count; i++)
    {
        float d = evalPrimitive(p, prims[i]);
        if (prims[i].subtract != 0u) { empty = min(empty, d); }
        else { solid = min(solid, d); if (d < mind) { mind = d; mat_id = prims[i].material_id; } }
    }
}

// ─── Octahedral normal encode (unit vec → [-1,1]²) ───────────────────────────
float2 octEncode(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z) + 1e-8);
    float2 e = n.xy;
    if (n.z < 0.0)
        e = (1.0 - abs(float2(e.y, e.x))) * float2(e.x >= 0.0 ? 1.0 : -1.0, e.y >= 0.0 ? 1.0 : -1.0);
    return e;   // in [-1,1]
}

uint snorm16(float v) { return uint(int(round(clamp(v, -1.0, 1.0) * 32767.0)) & 0xFFFF); }

[shader("compute")]
[numthreads(9, 9, 9)]
void csMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    GpuBrickJob job = jobs[gid.x];
    float3 p  = job.brick_world_min + (float3(lid) + 0.5) * job.voxel_size;

    float e = job.voxel_size * 0.5;

    // ── CSG by operands, each carrying its OWN correct normal ─────────────────
    // Surface = max( min(solidPrim, terrain), -empty ). The normal is the active
    // operand's gradient, so selecting by which term wins is correct EVERYWHERE —
    // no per-case overrides:
    //   • solid prim → ∇solidPrim                (FD, cheap)
    //   • terrain    → analytic ∇(p.y − A·noise)  (1 graph eval, smooth)
    //   • subtract   → −∇empty                    (points INTO the carved cavity)
    float solidP, empty; uint matP;
    sceneCSG(p, solidP, empty, matP);

    // Winning SOLID operand A = min(solidPrim, terrain), with its normal + material.
    float3 gS    = float3(
        solidPrimSdf(p + float3(e, 0, 0)) - solidPrimSdf(p - float3(e, 0, 0)),
        solidPrimSdf(p + float3(0, e, 0)) - solidPrimSdf(p - float3(0, e, 0)),
        solidPrimSdf(p + float3(0, 0, e)) - solidPrimSdf(p - float3(0, 0, e)));
    float  solid  = solidP;
    float3 nSolid = length(gS) > 1e-6 ? normalize(gS) : float3(0, 1, 0);
    uint   mat_id = matP;
    if (pc.terrain_on != 0u)
    {
        NV t = terrainDensity(p, float3(0, 0, 0));
        if (t.v < solid) { solid = t.v; nSolid = t.g; mat_id = TERRAIN_MAT; }
    }

    // f = max(A, −empty): a subtract carves where −empty wins → its inward normal.
    float  sdf = max(solid, -empty);
    float3 n   = nSolid;
    if (-empty > solid)
    {
        float3 gE = float3(
            emptySdf(p + float3(e, 0, 0)) - emptySdf(p - float3(e, 0, 0)),
            emptySdf(p + float3(0, e, 0)) - emptySdf(p - float3(0, e, 0)),
            emptySdf(p + float3(0, 0, e)) - emptySdf(p - float3(0, 0, e)));
        n = length(gE) > 1e-6 ? -normalize(gE) : nSolid;
    }

    float2 oct = octEncode(n);

    float dn = clamp(sdf / (BAND * job.voxel_size), -1.0, 1.0);   // distance → [-1,1]
    uint word0 = snorm16(dn)      | (snorm16(oct.x) << 16u);
    uint word1 = snorm16(oct.y) | ((mat_id & 0xFFu) << 16u) | (0u << 24u);  // flags = 0

    uint vox = lid.x + 9u * (lid.y + 9u * lid.z);
    geom_atlas.Store2(job.atlas_offset + vox * 8u, uint2(word0, word1));
}
