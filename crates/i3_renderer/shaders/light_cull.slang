struct ClusterAABB {
    float4 min_point;
    float4 max_point;
};

struct LightData {
    float3 position;
    float radius;
    float3 color;
    float intensity;
    float3 direction;
    uint light_type;
};

struct PushConstants {
    float4x4 view_matrix;
    uint3 grid_size;
    uint light_count;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]]
StructuredBuffer<ClusterAABB> cluster_aabbs;

[[vk::binding(1, 0)]]
StructuredBuffer<LightData> lights;

struct ClusterGridData {
    uint offset;
    uint count;
};

[[vk::binding(2, 0)]]
RWStructuredBuffer<ClusterGridData> cluster_grid;

[[vk::binding(3, 0)]]
RWStructuredBuffer<uint> cluster_light_indices;

bool test_aabb_sphere(ClusterAABB aabb, float3 center, float radius) {
    float3 closest = max(aabb.min_point.xyz, min(center, aabb.max_point.xyz));
    float3 diff = closest - center;
    return dot(diff, diff) <= (radius * radius);
}

#define MAX_LIGHTS_PER_CLUSTER 512

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

    ClusterAABB aabb = cluster_aabbs[cluster_index];
    
    uint visible_lights[MAX_LIGHTS_PER_CLUSTER];
    uint visible_count = 0;

    for (uint i = 0; i < pc.light_count; ++i) {
        LightData light = lights[i];
        
        float3 view_pos = mul(pc.view_matrix, float4(light.position, 1.0)).xyz;
        
        bool intersects = false;
        if (light.light_type == 1) { // Directional
            intersects = true;
        } else {
            intersects = test_aabb_sphere(aabb, view_pos, light.radius);
        }

        if (intersects && visible_count < MAX_LIGHTS_PER_CLUSTER) {
            visible_lights[visible_count] = i;
            visible_count++;
        }
    }

    uint offset = 0;
    InterlockedAdd(cluster_light_indices[0], visible_count, offset);
    
    for (uint j = 0; j < visible_count; ++j) {
        cluster_light_indices[1 + offset + j] = visible_lights[j];
    }

    ClusterGridData grid_data;
    grid_data.offset = offset;
    grid_data.count = visible_count;
    cluster_grid[cluster_index] = grid_data;
}
