// Pass 3: cull and allocate (indirect).
// One thread per dirty brick (numthreads(1,1,1), dispatch x = dirty_count).
// BVH traversal → center-SDF cull → atlas slot allocation → emit GpuBrickJob.
// Reuses old_slot from dirty_list when the brick had a previously allocated slot.

#include "brickmap_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<GpuBvhNode>     bvh_nodes;
[[vk::binding(1, 0)]] StructuredBuffer<GpuPrimitive>   prims;
[[vk::binding(2, 0)]] StructuredBuffer<LevelState>     level_states;
[[vk::binding(3, 0)]] ByteAddressBuffer                dirty_list;    // uint2 per entry
[[vk::binding(4, 0)]] RWByteAddressBuffer              page_table;
[[vk::binding(5, 0)]] RWByteAddressBuffer              alloc_buf;     // u32 brick_count per level
[[vk::binding(6, 0)]] RWStructuredBuffer<GpuBrickJob>  jobs;
[[vk::binding(7, 0)]] RWByteAddressBuffer              job_count;

struct CullPC
{
    uint prim_count;
    uint bvh_root;
    uint _pad0;
    uint _pad1;
};
[[vk::push_constant]] CullPC pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void csMain(uint3 dtid : SV_DispatchThreadID)
{
    uint idx = dtid.x;
    if (idx >= NUM_LEVELS * GRID_VOL) return;  // safety guard against oversized dispatch

    uint lev_flat = dirty_list.Load(idx * 4u);

    uint lev  = lev_flat / GRID_VOL;
    uint flat = lev_flat % GRID_VOL;
    uint cx = flat % GRID_X;
    uint cy = (flat / GRID_X) % GRID_Y;
    uint cz = flat / (GRID_X * GRID_Y);

    LevelState ls = level_states[lev];
    float voxel_size = ls.voxel_size;
    float bws       = (float)BRICK_SIZE * voxel_size;
    float half_diag = bws * 0.8660254;  // sqrt(3)/2

    int gx = (int)GRID_X, gy = (int)GRID_Y, gz = (int)GRID_Z;
    int3 new_ob = ls.new_ob;
    int ob_x_mod = ((new_ob.x % gx) + gx) % gx;
    int ob_y_mod = ((new_ob.y % gy) + gy) % gy;
    int ob_z_mod = ((new_ob.z % gz) + gz) % gz;
    int off_x = ((int)cx - ob_x_mod + gx) % gx;
    int off_y = ((int)cy - ob_y_mod + gy) % gy;
    int off_z = ((int)cz - ob_z_mod + gz) % gz;

    float3 brick_min = float3(
        (new_ob.x + off_x) * bws,
        (new_ob.y + off_y) * bws,
        (new_ob.z + off_z) * bws
    );
    float3 brick_max = brick_min + float3(bws, bws, bws);
    float3 center    = (brick_min + brick_max) * 0.5;

    // ── BVH traversal: evaluate SDF at brick center ───────────────────────────
    float solid_d = 1e10;
    float empty_d = 1e10;
    bool  any_candidate = false;

    if (pc.bvh_root != 0xFFFFFFFFu && pc.prim_count > 0u)
    {
        float3 query_min = brick_min - float3(voxel_size * 2.0, voxel_size * 2.0, voxel_size * 2.0);
        float3 query_max = brick_max + float3(voxel_size * 2.0, voxel_size * 2.0, voxel_size * 2.0);

        uint stack[64];
        uint top = 0u;
        stack[top++] = pc.bvh_root;
        while (top > 0u)
        {
            uint ni = stack[--top];
            GpuBvhNode node = bvh_nodes[ni];
            if (any(query_min > node.aabb_max) || any(query_max < node.aabb_min))
                continue;
            if (node.left == 0xFFFFFFFFu)
            {
                uint pi = node.right_or_prim;
                if (pi < pc.prim_count)
                {
                    float d = evalPrimitive(center, prims[pi]);
                    any_candidate = true;
                    if (prims[pi].subtract != 0u) empty_d = min(empty_d, d);
                    else                          solid_d = min(solid_d, d);
                }
            }
            else
            {
                if (top + 1u < 64u)
                {
                    stack[top++] = node.left;
                    stack[top++] = node.right_or_prim;
                }
            }
        }
    }
    else if (pc.prim_count > 0u)
    {
        any_candidate = true;
        for (uint i = 0u; i < pc.prim_count; i++)
        {
            float d = evalPrimitive(center, prims[i]);
            if (prims[i].subtract != 0u) empty_d = min(empty_d, d);
            else                         solid_d = min(solid_d, d);
        }
    }

    if (!any_candidate) return;
    float center_sdf = max(solid_d, -empty_d);
    if (abs(center_sdf) > half_diag * 2.0) return;

    // ── Atlas slot allocation ─────────────────────────────────────────────────
    // Try to pop a recycled slot from the per-level free stack (pushed by dirty_pass).
    // dirty_pass is fully complete (barrier), so the stack is stable — no concurrent pushes.
    uint lev_stride = lev * ALLOC_STRIDE;
    uint slot = MAX_BRICKS;  // sentinel: no slot yet
    {
        uint expected = alloc_buf.Load(lev_stride + ALLOC_FREE_TOP);
        while (expected > 0u)
        {
            uint result;
            alloc_buf.InterlockedCompareExchange(
                lev_stride + ALLOC_FREE_TOP, expected, expected - 1u, result);
            if (result == expected)
            {
                slot = alloc_buf.Load(lev_stride + ALLOC_STACK + (expected - 1u) * 4u);
                break;
            }
            expected = result;
        }
    }
    if (slot == MAX_BRICKS)
    {
        // Free stack empty: allocate a fresh slot from the monotone counter.
        alloc_buf.InterlockedAdd(lev_stride + ALLOC_COUNT, 1u, slot);
        if (slot >= MAX_BRICKS) return;  // atlas full
    }

    page_table.Store(lev_flat * 4u, slot);

    // ── Emit GpuBrickJob ──────────────────────────────────────────────────────
    uint job_idx;
    job_count.InterlockedAdd(0, 1u, job_idx);
    if (job_idx < MAX_JOBS)
    {
        uint atlas_offset = (lev * MAX_BRICKS + slot) * BRICK_DWORDS * 4u;
        GpuBrickJob job;
        job.brick_world_min = brick_min;
        job.voxel_size      = voxel_size;
        job.atlas_offset    = atlas_offset;
        job.half_diag       = half_diag;
        job._pad            = uint2(0u, 0u);
        jobs[job_idx] = job;
    }
}
