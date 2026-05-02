// HDR Single-Pass Downsampler — box-filter average reduction.
//
// Copies HDR_Target → HDR_Mips mip 0 and generates mips 1..5 in one dispatch.
// Same LDS approach as hiz_spd.slang, but reduction = (a+b+c+d)*0.25.
//
// Each thread group: 16×16 threads, handles a 32×32 tile.
//   Thread (tx,ty) reads its 2×2 block → writes 4 pixels to mip 0, average → mip 1 (LDS).
//   Subsequent steps reduce LDS by 4 each time → mips 2..5.
//
// Dispatch: ceil(width/32) × ceil(height/32).
//
// Intra-step race: in step 2, thread (i,j) reads g_lds[2j][2i] AND writes g_lds[j][i].
// These ranges overlap, causing a RAW hazard between warps on NVIDIA (warp size 32,
// step 2 uses 64 threads = 2 warps). Fix: split read and write phases with a
// GroupMemoryBarrierWithGroupSync between them, holding the computed value in a register.

struct HdrSpdPc {
    uint2 src_size; // HDR_Target dimensions
};
[[vk::push_constant]] HdrSpdPc pc;

[[vk::binding(0, 0)]] Texture2D<float4>   hdr_src;   // HDR_Target
[[vk::binding(1, 0)]] RWTexture2D<float4> hdr_mip0;
[[vk::binding(2, 0)]] RWTexture2D<float4> hdr_mip1;
[[vk::binding(3, 0)]] RWTexture2D<float4> hdr_mip2;
[[vk::binding(4, 0)]] RWTexture2D<float4> hdr_mip3;
[[vk::binding(5, 0)]] RWTexture2D<float4> hdr_mip4;
[[vk::binding(6, 0)]] RWTexture2D<float4> hdr_mip5;

groupshared float4 g_lds[16][16];

float4 LoadClamped(uint2 coord)
{
    return hdr_src.Load(int3(min(coord, pc.src_size - 1u), 0));
}

[numthreads(16, 16, 1)]
void computeMain(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint2 local     = tid.xy;
    uint2 tile_base = gid.xy * 32u;

    // Ceil-halving iteratively: (n+1)>>1 avoids floor for odd dimensions (e.g. 1080p).
    uint2 mip1_size = max((pc.src_size + 1u) >> 1u, uint2(1u, 1u));
    uint2 mip2_size = max((mip1_size + 1u) >> 1u, uint2(1u, 1u));
    uint2 mip3_size = max((mip2_size + 1u) >> 1u, uint2(1u, 1u));
    uint2 mip4_size = max((mip3_size + 1u) >> 1u, uint2(1u, 1u));
    uint2 mip5_size = max((mip4_size + 1u) >> 1u, uint2(1u, 1u));

    // ── Step 0: copy 2×2 block to mip 0 + average → mip 1 ───────────────────
    uint2 src = tile_base + local * 2u;
    float4 a = LoadClamped(src);
    float4 b = LoadClamped(src + uint2(1u, 0u));
    float4 c = LoadClamped(src + uint2(0u, 1u));
    float4 d = LoadClamped(src + uint2(1u, 1u));

    // Write all 4 pixels to mip 0 (copy)
    if (all(src              < pc.src_size)) hdr_mip0[src              ] = a;
    if (all(src + uint2(1,0) < pc.src_size)) hdr_mip0[src + uint2(1,0)] = b;
    if (all(src + uint2(0,1) < pc.src_size)) hdr_mip0[src + uint2(0,1)] = c;
    if (all(src + uint2(1,1) < pc.src_size)) hdr_mip0[src + uint2(1,1)] = d;

    float4 v1 = (a + b + c + d) * 0.25f;
    uint2 mip1_coord = gid.xy * 16u + local;
    if (all(mip1_coord < mip1_size))
    {
        hdr_mip1[mip1_coord] = v1;
        g_lds[local.y][local.x] = v1;
    }
    else
    {
        g_lds[local.y][local.x] = float4(0.0f); // OOB: neutral — edge_fade zeroes confidence at screen edges
    }

    GroupMemoryBarrierWithGroupSync();

    // ── Step 2: mip 1 → mip 2 (64 threads) ──────────────────────────────────
    // Phase A: reads only — store result in register to survive the mid-step barrier.
    float4 step2_val = float4(0.0f);
    if (local.x < 8u && local.y < 8u)
    {
        uint2 s = local * 2u;
        float4 v2 = (g_lds[s.y    ][s.x    ] + g_lds[s.y    ][s.x + 1u]
                   + g_lds[s.y + 1u][s.x    ] + g_lds[s.y + 1u][s.x + 1u]) * 0.25f;

        uint2 mip2_coord = gid.xy * 8u + local;
        if (all(mip2_coord < mip2_size))
        {
            hdr_mip2[mip2_coord] = v2;
            step2_val = v2;
        }
        // OOB: step2_val stays float4(0.0f) (neutral — edge_fade handles screen edges)
    }
    // Phase B: all reads done; now write to g_lds (prevents inter-warp RAW hazard).
    GroupMemoryBarrierWithGroupSync();
    if (local.x < 8u && local.y < 8u)
        g_lds[local.y][local.x] = step2_val;

    GroupMemoryBarrierWithGroupSync();

    // ── Step 3: mip 2 → mip 3 (16 threads) ──────────────────────────────────
    float4 step3_val = float4(0.0f);
    if (local.x < 4u && local.y < 4u)
    {
        uint2 s = local * 2u;
        float4 v3 = (g_lds[s.y    ][s.x    ] + g_lds[s.y    ][s.x + 1u]
                   + g_lds[s.y + 1u][s.x    ] + g_lds[s.y + 1u][s.x + 1u]) * 0.25f;

        uint2 mip3_coord = gid.xy * 4u + local;
        if (all(mip3_coord < mip3_size))
        {
            hdr_mip3[mip3_coord] = v3;
            step3_val = v3;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (local.x < 4u && local.y < 4u)
        g_lds[local.y][local.x] = step3_val;

    GroupMemoryBarrierWithGroupSync();

    // ── Step 4: mip 3 → mip 4 (4 threads) ───────────────────────────────────
    float4 step4_val = float4(0.0f);
    if (local.x < 2u && local.y < 2u)
    {
        uint2 s = local * 2u;
        float4 v4 = (g_lds[s.y    ][s.x    ] + g_lds[s.y    ][s.x + 1u]
                   + g_lds[s.y + 1u][s.x    ] + g_lds[s.y + 1u][s.x + 1u]) * 0.25f;

        uint2 mip4_coord = gid.xy * 2u + local;
        if (all(mip4_coord < mip4_size))
        {
            hdr_mip4[mip4_coord] = v4;
            step4_val = v4;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (local.x < 2u && local.y < 2u)
        g_lds[local.y][local.x] = step4_val;

    GroupMemoryBarrierWithGroupSync();

    // ── Step 5: mip 4 → mip 5 (1 thread) ────────────────────────────────────
    if (local.x == 0u && local.y == 0u)
    {
        float4 v5 = (g_lds[0][0] + g_lds[0][1] + g_lds[1][0] + g_lds[1][1]) * 0.25f;

        if (all(gid.xy < mip5_size))
            hdr_mip5[gid.xy] = v5;
    }
}
