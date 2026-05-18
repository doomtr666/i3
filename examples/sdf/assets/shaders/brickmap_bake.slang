// Brickmap GPU bake — compute shader
// One workgroup = one brick.  Threads: [8, 8, 8] = 512 per brick.
//
// CPU does: BVH query, centre-SDF cull, atlas-slot allocation.
// GPU does: all 512 voxel SDF evaluations.

// ─── Bindings ─────────────────────────────────────────────────────────────────

struct GpuBrickJob
{
    float3 brick_world_min;
    float  voxel_size;
    uint   atlas_offset;   // = (lev*MAX_BRICKS + brick_idx) * BRICK_VOXELS
    float  half_diag;      // unused in shader, kept for layout compat
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
    uint2  _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuBrickJob>   jobs;
[[vk::binding(1, 0)]] StructuredBuffer<GpuPrimitive>  prims;
[[vk::binding(2, 0)]] RWStructuredBuffer<float>       sdf_atlas;

struct BakePushConstants
{
    uint prim_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::push_constant]] BakePushConstants pc;

// ─── Primitive type constants ─────────────────────────────────────────────────
static const uint PRIM_SPHERE   = 0u;
static const uint PRIM_BOX      = 1u;
static const uint PRIM_CAPSULE  = 2u;
static const uint PRIM_CYLINDER = 3u;
static const uint PRIM_TORUS    = 4u;

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
    // Transform to local space
    float3 pc     = p_world - float3(prim.row0.w, prim.row1.w, prim.row2.w);
    float3 p_rot  = float3(dot(prim.row0.xyz, pc),
                            dot(prim.row1.xyz, pc),
                            dot(prim.row2.xyz, pc));
    float3 p_local = p_rot * prim.inv_scale;

    // Evaluate primitive SDF in local space
    float d_local;
    if (prim.prim_type == PRIM_SPHERE)
        d_local = sdfSphere(p_local, prim.data0);
    else if (prim.prim_type == PRIM_BOX)
        d_local = sdfBox(p_local, float3(prim.data0, prim.data1, prim.data2));
    else if (prim.prim_type == PRIM_CAPSULE)
        d_local = sdfCapsule(p_local, prim.data0, prim.data1);
    else if (prim.prim_type == PRIM_CYLINDER)
        d_local = sdfCylinder(p_local, prim.data0, prim.data1);
    else // TORUS
        d_local = sdfTorus(p_local, prim.data0, prim.data1);

    // Scale back to world space: d_world = d_local * scale = d_local / inv_scale
    return d_local / prim.inv_scale;
}

// ─── Compute entry point ─────────────────────────────────────────────────────

// One workgroup = one brick.  9×9×9 threads produce 729 voxels per brick:
// the inner 8×8×8 are the visible data, the extra slice on each positive face
// is the 1-voxel overlap that gives the trilinear sampler cross-brick corners.
[shader("compute")]
[numthreads(9, 9, 9)]
void csMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    GpuBrickJob job = jobs[gid.x];

    // World-space position of the centre of this voxel (lid goes 0..8 inclusive).
    // lid == 8 along an axis samples the neighbouring brick's first voxel.
    float3 p = job.brick_world_min + (float3(lid) + float3(0.5, 0.5, 0.5)) * job.voxel_size;

    float solid_d = 1e10;
    float empty_d = 1e10;

    for (uint i = 0u; i < pc.prim_count; i++)
    {
        float d = evalPrimitive(p, prims[i]);
        if (prims[i].subtract != 0u)
            empty_d = min(empty_d, d);
        else
            solid_d = min(solid_d, d);
    }

    float sdf = max(solid_d, -empty_d);

    // Atlas layout: stride 9 per axis (matching BRICK_VOXELS = 9³ = 729).
    uint vox_idx = lid.x + 9u * (lid.y + 9u * lid.z);
    sdf_atlas[job.atlas_offset + vox_idx] = sdf;
}
