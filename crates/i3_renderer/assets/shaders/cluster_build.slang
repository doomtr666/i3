// assets/system/shaders/cluster_build.slang
#include "lighting.slangh"

struct PushConstants {
    float4x4 inv_projection;
    uint3 grid_size;
    float near_plane;
    float far_plane;
    float2 screen_dimensions;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]]
RWStructuredBuffer<ClusterAABB> cluster_aabbs;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID) {
    if (threadId.x >= pc.grid_size.x || 
        threadId.y >= pc.grid_size.y || 
        threadId.z >= pc.grid_size.z) {
        return;
    }

    uint cluster_index = threadId.z * pc.grid_size.x * pc.grid_size.y + 
                         threadId.y * pc.grid_size.x + 
                         threadId.x;

    float2 tile_size = float2(1.0 / float(pc.grid_size.x), 1.0 / float(pc.grid_size.y));
    float2 uv_min = float2(threadId.x, threadId.y) * tile_size;
    float2 uv_max = float2(threadId.x + 1, threadId.y + 1) * tile_size;

    float z_near = pc.near_plane * pow(pc.far_plane / pc.near_plane, float(threadId.z) / float(pc.grid_size.z));
    float z_far = pc.near_plane * pow(pc.far_plane / pc.near_plane, float(threadId.z + 1) / float(pc.grid_size.z));

    float3 view_top_left = viewSpacePosFromUV(float2(uv_min.x, uv_min.y), 1.0, pc.inv_projection);
    float3 view_top_right = viewSpacePosFromUV(float2(uv_max.x, uv_min.y), 1.0, pc.inv_projection);
    float3 view_bot_left = viewSpacePosFromUV(float2(uv_min.x, uv_max.y), 1.0, pc.inv_projection);
    float3 view_bot_right = viewSpacePosFromUV(float2(uv_max.x, uv_max.y), 1.0, pc.inv_projection);

    float3 p_near_tl = view_top_left * (-z_near / view_top_left.z);
    float3 p_near_tr = view_top_right * (-z_near / view_top_right.z);
    float3 p_near_bl = view_bot_left * (-z_near / view_bot_left.z);
    float3 p_near_br = view_bot_right * (-z_near / view_bot_right.z);
    
    float3 p_far_tl = view_top_left * (-z_far / view_top_left.z);
    float3 p_far_tr = view_top_right * (-z_far / view_top_right.z);
    float3 p_far_bl = view_bot_left * (-z_far / view_bot_left.z);
    float3 p_far_br = view_bot_right * (-z_far / view_bot_right.z);

    float3 vmin = min(min(min(p_near_tl, p_near_tr), min(p_near_bl, p_near_br)),
                      min(min(p_far_tl, p_far_tr), min(p_far_bl, p_far_br)));
                      
    float3 vmax = max(max(max(p_near_tl, p_near_tr), max(p_near_bl, p_near_br)),
                      max(max(p_far_tl, p_far_tr), max(p_far_bl, p_far_br)));

    ClusterAABB aabb;
    aabb.min_point = float4(vmin, 0.0);
    aabb.max_point = float4(vmax, 0.0);

    cluster_aabbs[cluster_index] = aabb;
}
