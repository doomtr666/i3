// Pass 1: dirty detection.
// One thread per (level, cell) pair = NUM_LEVELS × GRID_VOL threads.
// Detects bricks that left the toroidal view (camera shift) or overlap edit spheres.
// Emits uint2(lev_flat, old_slot) to dirty_list for the cull pass to consume.

#include "brickmap_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<LevelState>          level_states;
[[vk::binding(1, 0)]] StructuredBuffer<InvalidationSphere>  spheres;
[[vk::binding(2, 0)]] RWByteAddressBuffer                   page_table;    // uint per cell
[[vk::binding(3, 0)]] RWByteAddressBuffer                   dirty_list;    // uint per entry (lev_flat)
[[vk::binding(4, 0)]] RWByteAddressBuffer                   dirty_count;
[[vk::binding(5, 0)]] RWByteAddressBuffer                   alloc_buf;

struct DirtyPC
{
    uint sphere_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::push_constant]] DirtyPC pc;

void emitDirty(uint lev_flat, uint old_slot)
{
    uint idx;
    dirty_count.InterlockedAdd(0, 1u, idx);
    if (idx < NUM_LEVELS * GRID_VOL)
        dirty_list.Store(idx * 4u, lev_flat);
    // Push freed slot onto the per-level free stack so cull_pass can recycle it.
    // dirty_pass runs entirely before cull_pass (barrier), so no concurrent pop race.
    if (old_slot != PAGE_EMPTY)
    {
        uint lev = lev_flat / GRID_VOL;
        uint top;
        alloc_buf.InterlockedAdd(lev * ALLOC_STRIDE + ALLOC_FREE_TOP, 1u, top);
        alloc_buf.Store(lev * ALLOC_STRIDE + ALLOC_STACK + top * 4u, old_slot);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 dtid : SV_DispatchThreadID)
{
    uint gid = dtid.x;
    if (gid >= NUM_LEVELS * GRID_VOL) return;

    uint lev  = gid / GRID_VOL;
    uint flat = gid % GRID_VOL;
    uint cx = flat % GRID_X;
    uint cy = (flat / GRID_X) % GRID_Y;
    uint cz = flat / (GRID_X * GRID_Y);

    LevelState ls = level_states[lev];
    float voxel_size = ls.voxel_size;
    float bws = (float)BRICK_SIZE * voxel_size;
    int gx = (int)GRID_X, gy = (int)GRID_Y, gz = (int)GRID_Z;

    uint pt_byte = gid * 4u;

    // ── Giant jump: wipe entire level ─────────────────────────────────────────
    if (ls.giant_jump != 0u)
    {
        uint old_slot = page_table.Load(pt_byte);
        page_table.Store(pt_byte, PAGE_EMPTY);
        emitDirty(gid, old_slot);
        return;
    }

    // ── Camera shift: toroidal leave detection ────────────────────────────────
    int3 old_ob = ls.old_ob;
    int3 new_ob = ls.new_ob;
    int3 delta  = new_ob - old_ob;

    if (delta.x != 0 || delta.y != 0 || delta.z != 0)
    {
        int ob_x_mod = ((old_ob.x % gx) + gx) % gx;
        int ob_y_mod = ((old_ob.y % gy) + gy) % gy;
        int ob_z_mod = ((old_ob.z % gz) + gz) % gz;
        int off_x = ((int)cx - ob_x_mod + gx) % gx;
        int off_y = ((int)cy - ob_y_mod + gy) % gy;
        int off_z = ((int)cz - ob_z_mod + gz) % gz;
        int dx = delta.x, dy = delta.y, dz = delta.z;
        bool camera_dirty =
            (dx > 0 && off_x < dx)        ||
            (dx < 0 && off_x >= gx + dx)  ||
            (dy > 0 && off_y < dy)        ||
            (dy < 0 && off_y >= gy + dy)  ||
            (dz > 0 && off_z < dz)        ||
            (dz < 0 && off_z >= gz + dz);
        if (camera_dirty)
        {
            uint old_slot = page_table.Load(pt_byte);
            page_table.Store(pt_byte, PAGE_EMPTY);
            emitDirty(gid, old_slot);
            return;
        }
    }

    // ── Sphere invalidation ───────────────────────────────────────────────────
    {
        int ob_x_mod2 = ((new_ob.x % gx) + gx) % gx;
        int ob_y_mod2 = ((new_ob.y % gy) + gy) % gy;
        int ob_z_mod2 = ((new_ob.z % gz) + gz) % gz;
        int off_x2 = ((int)cx - ob_x_mod2 + gx) % gx;
        int off_y2 = ((int)cy - ob_y_mod2 + gy) % gy;
        int off_z2 = ((int)cz - ob_z_mod2 + gz) % gz;
        float3 brick_min = float3(
            (new_ob.x + off_x2) * bws,
            (new_ob.y + off_y2) * bws,
            (new_ob.z + off_z2) * bws
        );
        float3 brick_max = brick_min + float3(bws, bws, bws);
        for (uint s = 0u; s < pc.sphere_count; s++)
        {
            InvalidationSphere sp = spheres[s];
            float r = sp.radius + voxel_size * 4.0;
            float3 closest = clamp(sp.center, brick_min, brick_max);
            float dist2 = dot(sp.center - closest, sp.center - closest);
            if (dist2 <= r * r)
            {
                uint old_slot = page_table.Load(pt_byte);
                page_table.Store(pt_byte, PAGE_EMPTY);
                emitDirty(gid, old_slot);
                return;
            }
        }
    }
}
