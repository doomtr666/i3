// Brickmap GPU bake — compute shader
// One workgroup = one brick.  Threads: [9, 9, 9] = 729 per brick.
// Dispatched indirectly from dispatch_jobs_buf; jobs were emitted by the cull pass.
//
// Atlas layout: u8-packed, 4 voxels per DWORD.
// BRICK_DWORDS = ceil(729/4) = 183.  atlas_offset = byte offset in ByteAddressBuffer.

// ─── Bindings ─────────────────────────────────────────────────────────────────

struct GpuBrickJob
{
    float3 brick_world_min;
    float  voxel_size;
    uint   atlas_offset;   // byte offset into ByteAddressBuffer atlas
    float  half_diag;      // brick half-diagonal (world space), used for SDF normalization
    uint2  _pad;
};

struct GpuPrimitive
{
    float4 row0;       // inv_rot row 0 [xyz] + translation.x [w]
    float4 row1;       // inv_rot row 1 [xyz] + translation.y [w]
    float4 row2;       // inv_rot row 2 [xyz] + translation.z [w]
    float  inv_scale;
    float  data0;      // radius / half_extents.x / half_height / major_radius
    float  data1;      // half_extents.y / radius / minor_radius
    float  data2;      // half_extents.z
    uint   prim_type;  // 0=Sphere  1=Box  2=Capsule  3=Cylinder  4=Torus
    uint   subtract;   // 0 or 1
    uint   material_id;
    uint   _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuBrickJob>   jobs;
[[vk::binding(1, 0)]] StructuredBuffer<GpuPrimitive>  prims;
[[vk::binding(2, 0)]] RWByteAddressBuffer              sdf_atlas;
[[vk::binding(3, 0)]] RWByteAddressBuffer              mat_atlas;

struct BakePushConstants
{
    uint prim_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::push_constant]] BakePushConstants pc;

// ─── Constants ────────────────────────────────────────────────────────────────

static const uint PRIM_SPHERE   = 0u;
static const uint PRIM_BOX      = 1u;
static const uint PRIM_CAPSULE  = 2u;
static const uint PRIM_CYLINDER = 3u;
static const uint PRIM_TORUS    = 4u;

static const uint BRICK_VOXELS = 729u;   // 9³
static const uint BRICK_DWORDS = 183u;   // ceil(729/4)

// ─── LDS — one slot per voxel (both atlases) ─────────────────────────────────

groupshared uint lds_sdf[729];
groupshared uint lds_mat[729];

// ─── SDF functions in local (un-transformed) space ───────────────────────────

float sdfSphere(float3 p, float radius)
{
    return length(p) - radius;
}

float sdfBox(float3 p, float3 he)
{
    float3 q = abs(p) - he;
    return length(max(q, float3(0.0, 0.0, 0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdfCapsule(float3 p, float half_height, float radius)
{
    float clamped_y = clamp(p.y, -half_height, half_height);
    return length(p - float3(0.0, clamped_y, 0.0)) - radius;
}

float sdfCylinder(float3 p, float half_height, float radius)
{
    float d_xz = length(float2(p.x, p.z)) - radius;
    float d_y  = abs(p.y) - half_height;
    return length(max(float2(d_xz, d_y), float2(0.0, 0.0))) + min(max(d_xz, d_y), 0.0);
}

float sdfTorus(float3 p, float major_r, float minor_r)
{
    float2 q = float2(length(float2(p.x, p.z)) - major_r, p.y);
    return length(q) - minor_r;
}

// ─── Evaluate one primitive at world-space position p ────────────────────────

float evalPrimitive(float3 p_world, GpuPrimitive prim)
{
    float3 pc     = p_world - float3(prim.row0.w, prim.row1.w, prim.row2.w);
    float3 p_rot  = float3(dot(prim.row0.xyz, pc),
                            dot(prim.row1.xyz, pc),
                            dot(prim.row2.xyz, pc));
    float3 p_local = p_rot * prim.inv_scale;

    float d_local;
    if (prim.prim_type == PRIM_SPHERE)
        d_local = sdfSphere(p_local, prim.data0);
    else if (prim.prim_type == PRIM_BOX)
        d_local = sdfBox(p_local, float3(prim.data0, prim.data1, prim.data2));
    else if (prim.prim_type == PRIM_CAPSULE)
        d_local = sdfCapsule(p_local, prim.data0, prim.data1);
    else if (prim.prim_type == PRIM_CYLINDER)
        d_local = sdfCylinder(p_local, prim.data0, prim.data1);
    else
        d_local = sdfTorus(p_local, prim.data0, prim.data1);

    return d_local / prim.inv_scale;
}

// ─── Compute entry point ─────────────────────────────────────────────────────

[shader("compute")]
[numthreads(9, 9, 9)]
void csMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    GpuBrickJob job = jobs[gid.x];

    float3 p = job.brick_world_min + (float3(lid) + float3(0.5, 0.5, 0.5)) * job.voxel_size;

    // ── Phase 1: evaluate SDF + track closest primitive for material ──────────

    float solid_d = 1e10;
    float empty_d = 1e10;
    uint  mat_id  = 0u;
    float min_solid_d = 1e10;

    for (uint i = 0u; i < pc.prim_count; i++)
    {
        float d = evalPrimitive(p, prims[i]);
        if (prims[i].subtract != 0u)
        {
            empty_d = min(empty_d, d);
        }
        else
        {
            solid_d = min(solid_d, d);
            if (d < min_solid_d)
            {
                min_solid_d = d;
                mat_id = prims[i].material_id;
            }
        }
    }

    float sdf = max(solid_d, -empty_d);

    // Quantize SDF to u8: range [-half_diag, +half_diag] → [0, 255]
    float norm = clamp(sdf / job.half_diag, -1.0, 1.0);
    uint  sdf_byte = uint(round((norm + 1.0) * 127.5));

    uint vox_idx = lid.x + 9u * (lid.y + 9u * lid.z);

    lds_sdf[vox_idx] = sdf_byte;
    lds_mat[vox_idx] = mat_id & 0xFFu;

    // ── Phase 2: pack 4 bytes per DWORD and write to atlas ───────────────────

    GroupMemoryBarrierWithGroupSync();

    if (vox_idx < BRICK_DWORDS)
    {
        uint base = vox_idx * 4u;
        uint b0 = lds_sdf[base];
        uint b1 = (base + 1u < BRICK_VOXELS) ? lds_sdf[base + 1u] : 0u;
        uint b2 = (base + 2u < BRICK_VOXELS) ? lds_sdf[base + 2u] : 0u;
        uint b3 = (base + 3u < BRICK_VOXELS) ? lds_sdf[base + 3u] : 0u;
        sdf_atlas.Store(job.atlas_offset + vox_idx * 4u,
                        b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u));

        uint m0 = lds_mat[base];
        uint m1 = (base + 1u < BRICK_VOXELS) ? lds_mat[base + 1u] : 0u;
        uint m2 = (base + 2u < BRICK_VOXELS) ? lds_mat[base + 2u] : 0u;
        uint m3 = (base + 3u < BRICK_VOXELS) ? lds_mat[base + 3u] : 0u;
        mat_atlas.Store(job.atlas_offset + vox_idx * 4u,
                        m0 | (m1 << 8u) | (m2 << 16u) | (m3 << 24u));
    }
}
