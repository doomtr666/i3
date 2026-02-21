struct ClusterAABB {
    float4 min_point;
    float4 max_point;
};

// Push constants for cluster build
struct PushConstants {
    float4x4 inv_projection;
    uint3 grid_size;
    float near_plane;
    float far_plane;
    float2 screen_dimensions;
};

[[vk::push_constant]]
PushConstants pc;

// Output buffer
[[vk::binding(0, 0)]]
RWStructuredBuffer<ClusterAABB> cluster_aabbs;

// Helper: View space from screen UV and depth
float3 screen2view(float2 uv, float depth, float4x4 inv_proj) {
    float4 clip = float4(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    float4 view = mul(inv_proj, clip);
    return view.xyz / view.w;
}

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

    // Tile bounds in UV space [0, 1]
    float2 tile_size = float2(1.0 / float(pc.grid_size.x), 1.0 / float(pc.grid_size.y));
    float2 uv_min = float2(threadId.x, threadId.y) * tile_size;
    float2 uv_max = float2(threadId.x + 1, threadId.y + 1) * tile_size;

    // Depth bounds per slice (logarithmic)
    float z_near = pc.near_plane * pow(pc.far_plane / pc.near_plane, float(threadId.z) / float(pc.grid_size.z));
    float z_far = pc.near_plane * pow(pc.far_plane / pc.near_plane, float(threadId.z + 1) / float(pc.grid_size.z));

    // Calculate view space corners of the frustum for this tile at z_near and z_far.
    // However, since we define z in view space directly, we can find the rays going through the 4 corners at z = -1, and multiply by z_near/z_far.
    // Vulkan Clip Space Z is [0, 1], so depth=0 is near, depth=1 is far.
    // Actually simpler: reverse project the 4 corners at depth = 1 (far plane) to get the view-space edge vectors.

    float3 view_top_left = screen2view(float2(uv_min.x, uv_min.y), 1.0, pc.inv_projection);
    float3 view_top_right = screen2view(float2(uv_max.x, uv_min.y), 1.0, pc.inv_projection);
    float3 view_bot_left = screen2view(float2(uv_min.x, uv_max.y), 1.0, pc.inv_projection);
    float3 view_bot_right = screen2view(float2(uv_max.x, uv_max.y), 1.0, pc.inv_projection);

    // Normalize the eye-to-far rays (since view origin is 0,0,0) and scale by -Z.
    // Wait, screen2view gives coordinates with negative Z in right-handed systems?
    // Let's assume right handed view space where Z is negative into the screen.
    // Our z_near and z_far are positive distances.
    
    // To find the actual view-space point at distance Z along the Z axis, we scale the ray so its Z component is -znear/-zfar.
    float3 p_near_tl = view_top_left * (-z_near / view_top_left.z);
    float3 p_near_tr = view_top_right * (-z_near / view_top_right.z);
    float3 p_near_bl = view_bot_left * (-z_near / view_bot_left.z);
    float3 p_near_br = view_bot_right * (-z_near / view_bot_right.z);
    
    float3 p_far_tl = view_top_left * (-z_far / view_top_left.z);
    float3 p_far_tr = view_top_right * (-z_far / view_top_right.z);
    float3 p_far_bl = view_bot_left * (-z_far / view_bot_left.z);
    float3 p_far_br = view_bot_right * (-z_far / view_bot_right.z);

    // Calculate min/max AABB
    float3 vmin = min(min(min(p_near_tl, p_near_tr), min(p_near_bl, p_near_br)),
                      min(min(p_far_tl, p_far_tr), min(p_far_bl, p_far_br)));
                      
    float3 vmax = max(max(max(p_near_tl, p_near_tr), max(p_near_bl, p_near_br)),
                      max(max(p_far_tl, p_far_tr), max(p_far_bl, p_far_br)));

    ClusterAABB aabb;
    aabb.min_point = float4(vmin, 0.0);
    aabb.max_point = float4(vmax, 0.0);

    cluster_aabbs[cluster_index] = aabb;
}
