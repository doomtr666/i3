// SVO GPU bake — compute shader
// One workgroup = one brick.  Threads: [9, 9, 9] = 729 per brick.
// Atlas layout: u8-packed, 4 voxels per DWORD.  BRICK_DWORDS = 183.

struct GpuBrickJob
{
    float3 brick_world_min;
    float  voxel_size;
    uint   atlas_offset;   // byte offset into ByteAddressBuffer
    float  half_diag;
    uint2  _pad;
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
    uint   prim_type;  // 0=Sphere  1=Box  2=Capsule  3=Cylinder  4=Torus
    uint   subtract;
    uint   material_id;
    uint   _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuBrickJob>   jobs;
[[vk::binding(1, 0)]] StructuredBuffer<GpuPrimitive>  prims;
[[vk::binding(2, 0)]] RWByteAddressBuffer              sdf_atlas;
[[vk::binding(3, 0)]] RWByteAddressBuffer              mat_atlas;

struct BakePC { uint prim_count; uint _pad0; uint _pad1; uint _pad2; };
[[vk::push_constant]] BakePC pc;

static const uint PRIM_SPHERE   = 0u;
static const uint PRIM_BOX      = 1u;
static const uint PRIM_CAPSULE  = 2u;
static const uint PRIM_CYLINDER = 3u;
static const uint PRIM_TORUS    = 4u;
static const uint BRICK_VOXELS  = 729u;
static const uint BRICK_DWORDS  = 183u;

groupshared uint lds_sdf[729];
groupshared uint lds_mat[729];

float sdfSphere(float3 p, float r)   { return length(p) - r; }

float sdfBox(float3 p, float3 he)
{
    float3 q = abs(p) - he;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdfCapsule(float3 p, float hh, float r)
{
    float cy = clamp(p.y, -hh, hh);
    return length(p - float3(0, cy, 0)) - r;
}

float sdfCylinder(float3 p, float hh, float r)
{
    float dxz = length(p.xz) - r;
    float dy   = abs(p.y) - hh;
    return length(max(float2(dxz, dy), 0.0)) + min(max(dxz, dy), 0.0);
}

float sdfTorus(float3 p, float R, float r)
{
    float2 q = float2(length(p.xz) - R, p.y);
    return length(q) - r;
}

float evalPrimitive(float3 pw, GpuPrimitive prim)
{
    float3 pc2   = pw - float3(prim.row0.w, prim.row1.w, prim.row2.w);
    float3 pr    = float3(dot(prim.row0.xyz, pc2), dot(prim.row1.xyz, pc2), dot(prim.row2.xyz, pc2));
    float3 pl    = pr * prim.inv_scale;
    float  d;
    if      (prim.prim_type == PRIM_SPHERE)   d = sdfSphere(pl, prim.data0);
    else if (prim.prim_type == PRIM_BOX)      d = sdfBox(pl, float3(prim.data0, prim.data1, prim.data2));
    else if (prim.prim_type == PRIM_CAPSULE)  d = sdfCapsule(pl, prim.data0, prim.data1);
    else if (prim.prim_type == PRIM_CYLINDER) d = sdfCylinder(pl, prim.data0, prim.data1);
    else                                       d = sdfTorus(pl, prim.data0, prim.data1);
    return d / prim.inv_scale;
}

[shader("compute")]
[numthreads(9, 9, 9)]
void csMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    GpuBrickJob job = jobs[gid.x];
    float3 p = job.brick_world_min + (float3(lid) + 0.5) * job.voxel_size;

    float solid_d = 1e10, empty_d = 1e10;
    uint  mat_id  = 0u;
    float min_d   = 1e10;

    for (uint i = 0u; i < pc.prim_count; i++)
    {
        float d = evalPrimitive(p, prims[i]);
        if (prims[i].subtract != 0u) { empty_d = min(empty_d, d); }
        else { solid_d = min(solid_d, d); if (d < min_d) { min_d = d; mat_id = prims[i].material_id; } }
    }

    float sdf      = max(solid_d, -empty_d);
    float norm     = clamp(sdf / job.half_diag, -1.0, 1.0);
    uint  sdf_byte = uint(round((norm + 1.0) * 127.5));
    uint  vox      = lid.x + 9u * (lid.y + 9u * lid.z);

    lds_sdf[vox] = sdf_byte;
    lds_mat[vox] = mat_id & 0xFFu;

    GroupMemoryBarrierWithGroupSync();

    if (vox < BRICK_DWORDS)
    {
        uint base = vox * 4u;
        uint b0 = lds_sdf[base];
        uint b1 = (base + 1u < BRICK_VOXELS) ? lds_sdf[base + 1u] : 0u;
        uint b2 = (base + 2u < BRICK_VOXELS) ? lds_sdf[base + 2u] : 0u;
        uint b3 = (base + 3u < BRICK_VOXELS) ? lds_sdf[base + 3u] : 0u;
        sdf_atlas.Store(job.atlas_offset + vox * 4u,
                        b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u));

        uint m0 = lds_mat[base];
        uint m1 = (base + 1u < BRICK_VOXELS) ? lds_mat[base + 1u] : 0u;
        uint m2 = (base + 2u < BRICK_VOXELS) ? lds_mat[base + 2u] : 0u;
        uint m3 = (base + 3u < BRICK_VOXELS) ? lds_mat[base + 3u] : 0u;
        mat_atlas.Store(job.atlas_offset + vox * 4u,
                        m0 | (m1 << 8u) | (m2 << 16u) | (m3 << 24u));
    }
}
