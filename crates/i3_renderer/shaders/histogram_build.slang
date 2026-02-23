#define HISTOGRAM_BINS 256
#define GROUP_SIZE 16

struct PushConstants {
    float min_log_lum;
    float max_log_lum;
    float time_delta;
    uint pad;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]] Texture2D hdrImage;

// RWByteAddressBuffer or RWStructuredBuffer
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> histogram;

struct ExposureData {
    float prev_exposure;
    float new_exposure;
};
[[vk::binding(2, 0)]] StructuredBuffer<ExposureData> exposure_buffer;

// Compute luminous intensity from RGB
float luminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// Map luminance to histogram bin
uint colorToBin(float3 color, float minLogLum, float maxLogLum) {
    float lum = luminance(color);
    if (lum < 0.0001) {
        return 0;
    }
    
    // Calculate log luminance
    float logLum = log2(lum);
    
    // Map to [0.0, 1.0]
    float fBin = (logLum - minLogLum) / (maxLogLum - minLogLum);
    
    return uint(clamp(fBin, 0.0, 1.0) * float(HISTOGRAM_BINS - 1));
}

// Shared memory for per-workgroup histogram aggregation
groupshared uint sharedHistogram[HISTOGRAM_BINS];

[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID, uint gl_LocalInvocationIndex : SV_GroupIndex)
{
    // Initialize shared memory
    sharedHistogram[gl_LocalInvocationIndex] = 0;
    GroupMemoryBarrierWithGroupSync();

    // Ensure within bounds
    uint2 texDimensions;
    hdrImage.GetDimensions(texDimensions.x, texDimensions.y);
    
    if (gl_GlobalInvocationID.x < texDimensions.x && gl_GlobalInvocationID.y < texDimensions.y) {
        float3 color = hdrImage.Load(int3(gl_GlobalInvocationID.xy, 0)).rgb;
        
        // Remove Pre-Exposure Scale for accurate scene luminance analysis
        float prev_ex = exposure_buffer[0].prev_exposure;
        if (prev_ex == 0.0) { prev_ex = 1.0; }
        color /= prev_ex;
        
        uint binIndex = colorToBin(color, pc.min_log_lum, pc.max_log_lum);
        
        // Increment shared histogram bin
        InterlockedAdd(sharedHistogram[binIndex], 1);
    }
    
    GroupMemoryBarrierWithGroupSync();

    // Write shared memory to global memory
    InterlockedAdd(histogram[gl_LocalInvocationIndex], sharedHistogram[gl_LocalInvocationIndex]);
}
