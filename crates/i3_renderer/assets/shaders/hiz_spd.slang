// HiZ Single-Pass Downsampler — max reduction (Reverse-Z: largest = nearest)
//
// Each thread group processes a 32×32 tile of mip 0 and independently
// generates mips 1..5 via LDS reductions. No cross-group atomics needed
// because mip 5 has one pixel per group (groupId.xy == mip5_coord).
//
// [numthreads(16, 16, 1)] = 256 threads per group.
// Thread (tx,ty) reads a 2×2 block at (tile_base + local*2) for the mip 1 step,
// then successive halving in LDS for mips 2..5.
//
// Dispatch: ceil(width/32) × ceil(height/32).
//
// Boundary handling: out-of-bounds threads write 0.0 to g_lds (= "sky / no geometry"
// in Reverse-Z). max(valid, 0) == valid, so phantom 0s never corrupt in-bounds max
// results in subsequent steps.
//
// Intra-step race: in step 2, thread (i,j) reads g_lds[2j][2i] AND writes g_lds[j][i].
// These ranges overlap, causing a RAW hazard between warps on NVIDIA (warp size 32,
// step 2 uses 64 threads = 2 warps). Fix: split read and write phases with a
// GroupMemoryBarrierWithGroupSync between them.

struct HiZSpdPc
{
    uint2 mip0_size; // full dimensions of mip 0
};
[[vk::push_constant]]
HiZSpdPc pc;

[[vk::binding(0, 0)]]
Texture2D<float> hiz_mip0; // source (written by GBuffer MRT)
[[vk::binding(1, 0)]]
RWTexture2D<float> hiz_mip1;
[[vk::binding(2, 0)]]
RWTexture2D<float> hiz_mip2;
[[vk::binding(3, 0)]]
RWTexture2D<float> hiz_mip3;
[[vk::binding(4, 0)]]
RWTexture2D<float> hiz_mip4;
[[vk::binding(5, 0)]]
RWTexture2D<float> hiz_mip5;

groupshared float g_lds[16][16];

float LoadClamped(uint2 coord)
{
    return hiz_mip0.Load(int3(min(coord, pc.mip0_size - 1u), 0));
}

[numthreads(16, 16, 1)]
void computeMain(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint2 local = tid.xy;
    uint2 tile_base = gid.xy * 32u;

    // Ceil-halving iteratively: (n+1)>>1 avoids floor for odd dimensions (e.g. 1080p).
    uint2 mip1_size = max((pc.mip0_size + 1u) >> 1u, uint2(1u, 1u));
    uint2 mip2_size = max((mip1_size + 1u) >> 1u, uint2(1u, 1u));
    uint2 mip3_size = max((mip2_size + 1u) >> 1u, uint2(1u, 1u));
    uint2 mip4_size = max((mip3_size + 1u) >> 1u, uint2(1u, 1u));
    uint2 mip5_size = max((mip4_size + 1u) >> 1u, uint2(1u, 1u));

    // ── Step 1: mip 0 → mip 1 (all 256 threads) ─────────────────────────────
    uint2 src = tile_base + local * 2u;
    float a = LoadClamped(src);
    float b = LoadClamped(src + uint2(1u, 0u));
    float c = LoadClamped(src + uint2(0u, 1u));
    float d = LoadClamped(src + uint2(1u, 1u));
    float v1 = max(max(a, b), max(c, d));

    uint2 mip1_coord = gid.xy * 16u + local;
    if (all(mip1_coord < mip1_size))
    {
        hiz_mip1[mip1_coord] = v1;
        g_lds[local.y][local.x] = v1;
    }
    else
    {
        g_lds[local.y][local.x] = 0.0f; // OOB: neutral for max reduction
    }

    GroupMemoryBarrierWithGroupSync();

    // ── Step 2: mip 1 → mip 2 (64 threads: local < (8,8)) ───────────────────
    // Phase A: reads only — store result in register to survive the mid-step barrier.
    float step2_val = 0.0f;
    if (local.x < 8u && local.y < 8u)
    {
        uint2 s = local * 2u;
        float v2 = max(max(g_lds[s.y][s.x], g_lds[s.y][s.x + 1u]),
                       max(g_lds[s.y + 1u][s.x], g_lds[s.y + 1u][s.x + 1u]));

        uint2 mip2_coord = gid.xy * 8u + local;
        if (all(mip2_coord < mip2_size))
        {
            hiz_mip2[mip2_coord] = v2;
            step2_val = v2;
        }
        // OOB: step2_val stays 0.0f (neutral for max)
    }
    // Phase B: all reads done; now write to g_lds (prevents inter-warp RAW hazard).
    GroupMemoryBarrierWithGroupSync();
    if (local.x < 8u && local.y < 8u)
        g_lds[local.y][local.x] = step2_val;

    GroupMemoryBarrierWithGroupSync();

    // ── Step 3: mip 2 → mip 3 (16 threads: local < (4,4)) ───────────────────
    // 16 threads fit in one warp on NVIDIA (warp=32) so no inter-warp race,
    // but split phases for correctness on AMD (wavefront=64, still safe) and future HW.
    float step3_val = 0.0f;
    if (local.x < 4u && local.y < 4u)
    {
        uint2 s = local * 2u;
        float v3 = max(max(g_lds[s.y][s.x], g_lds[s.y][s.x + 1u]),
                       max(g_lds[s.y + 1u][s.x], g_lds[s.y + 1u][s.x + 1u]));

        uint2 mip3_coord = gid.xy * 4u + local;
        if (all(mip3_coord < mip3_size))
        {
            hiz_mip3[mip3_coord] = v3;
            step3_val = v3;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (local.x < 4u && local.y < 4u)
        g_lds[local.y][local.x] = step3_val;

    GroupMemoryBarrierWithGroupSync();

    // ── Step 4: mip 3 → mip 4 (4 threads: local < (2,2)) ────────────────────
    float step4_val = 0.0f;
    if (local.x < 2u && local.y < 2u)
    {
        uint2 s = local * 2u;
        float v4 = max(max(g_lds[s.y][s.x], g_lds[s.y][s.x + 1u]),
                       max(g_lds[s.y + 1u][s.x], g_lds[s.y + 1u][s.x + 1u]));

        uint2 mip4_coord = gid.xy * 2u + local;
        if (all(mip4_coord < mip4_size))
        {
            hiz_mip4[mip4_coord] = v4;
            step4_val = v4;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (local.x < 2u && local.y < 2u)
        g_lds[local.y][local.x] = step4_val;

    GroupMemoryBarrierWithGroupSync();

    // ── Step 5: mip 4 → mip 5 (1 thread: local == (0,0)) ────────────────────
    if (local.x == 0u && local.y == 0u)
    {
        float v5 = max(max(g_lds[0][0], g_lds[0][1]),
                       max(g_lds[1][0], g_lds[1][1]));

        if (all(gid.xy < mip5_size))
            hiz_mip5[gid.xy] = v5;
    }
}
