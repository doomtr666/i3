// Common structs and constants shared across all GPU brickmap passes.

// ─── Constants ────────────────────────────────────────────────────────────────

static const uint GRID_X    = 32u;
static const uint GRID_Y    = 32u;
static const uint GRID_Z    = 32u;
static const uint GRID_VOL  = 32768u;  // 32³
static const uint NUM_LEVELS = 10u;
static const uint BRICK_SIZE = 8u;
static const uint MAX_BRICKS = 8192u;
static const uint MAX_JOBS   = NUM_LEVELS * MAX_BRICKS; // 81920 — total job budget per frame
static const uint BRICK_VOXELS = 729u; // (BRICK_SIZE+1)³
static const uint BRICK_DWORDS = 183u; // ceil(729/4)
static const uint PAGE_EMPTY   = 0xFFFFFFFFu;

// dirty_list entry: one uint per entry = lev * GRID_VOL + flat (lev_flat).
// Stride: 4 bytes per entry. Max entries: NUM_LEVELS * GRID_VOL.

// alloc_buf layout per level (offset = lev * ALLOC_STRIDE):
//   [ALLOC_COUNT]    : brick_count — monotone fresh-slot counter
//   [ALLOC_FREE_TOP] : free_top   — height of the recycled-slot stack
//   [ALLOC_STACK + i*4] : free_stack[i] — recycled slot indices
// dirty_pass pushes freed old_slots here; cull_pass pops via CAS before fresh-alloc.
static const uint ALLOC_STRIDE   = 8u + MAX_BRICKS * 4u;
static const uint ALLOC_COUNT    = 0u;
static const uint ALLOC_FREE_TOP = 4u;
static const uint ALLOC_STACK    = 8u;

// ─── Shared structs ───────────────────────────────────────────────────────────

struct LevelState
{
    float3 old_origin;
    float  voxel_size;
    float3 new_origin;
    uint   giant_jump;     // 1 = rebake entire level
    int3   old_ob;
    uint   _pad0;
    int3   new_ob;
    uint   _pad1;
};

struct InvalidationSphere
{
    float3 center;
    float  radius;
};

struct GpuBrickJob
{
    float3 brick_world_min;
    float  voxel_size;
    uint   atlas_offset;
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
    uint   prim_type;
    uint   subtract;
    uint   material_id;
    uint   _pad;
};

struct GpuBvhNode
{
    float3 aabb_min;
    uint   left;           // u32::MAX = leaf
    float3 aabb_max;
    uint   right_or_prim;  // right child (internal) or prim_idx (leaf)
};

// ─── SDF primitive evaluation ─────────────────────────────────────────────────

static const uint PRIM_SPHERE   = 0u;
static const uint PRIM_BOX      = 1u;
static const uint PRIM_CAPSULE  = 2u;
static const uint PRIM_CYLINDER = 3u;
static const uint PRIM_TORUS    = 4u;

float sdfSphere(float3 p, float r)   { return length(p) - r; }
float sdfBox(float3 p, float3 he)
{
    float3 q = abs(p) - he;
    return length(max(q, float3(0,0,0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}
float sdfCapsule(float3 p, float hh, float r)
{
    return length(p - float3(0, clamp(p.y, -hh, hh), 0)) - r;
}
float sdfCylinder(float3 p, float hh, float r)
{
    float2 d = float2(length(float2(p.x, p.z)) - r, abs(p.y) - hh);
    return length(max(d, float2(0,0))) + min(max(d.x, d.y), 0.0);
}
float sdfTorus(float3 p, float R, float r)
{
    return length(float2(length(float2(p.x, p.z)) - R, p.y)) - r;
}

float evalPrimitive(float3 p_world, GpuPrimitive prim)
{
    float3 pc     = p_world - float3(prim.row0.w, prim.row1.w, prim.row2.w);
    float3 p_local = float3(dot(prim.row0.xyz, pc),
                             dot(prim.row1.xyz, pc),
                             dot(prim.row2.xyz, pc)) * prim.inv_scale;
    float d;
    if      (prim.prim_type == PRIM_SPHERE)   d = sdfSphere  (p_local, prim.data0);
    else if (prim.prim_type == PRIM_BOX)      d = sdfBox     (p_local, float3(prim.data0, prim.data1, prim.data2));
    else if (prim.prim_type == PRIM_CAPSULE)  d = sdfCapsule (p_local, prim.data0, prim.data1);
    else if (prim.prim_type == PRIM_CYLINDER) d = sdfCylinder(p_local, prim.data0, prim.data1);
    else                                      d = sdfTorus   (p_local, prim.data0, prim.data1);
    return d / prim.inv_scale;
}
