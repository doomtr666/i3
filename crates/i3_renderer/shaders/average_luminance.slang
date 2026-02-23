#define HISTOGRAM_BINS 256
#define GROUP_SIZE 256

struct PushConstants {
    float min_log_lum;
    float max_log_lum;
    float time_delta;
    float adaptation_rate;
    float pixel_count;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::push_constant]]
PushConstants pc;

// The histogram buffer accumulated in the previous pass
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> histogram;

// The exposure buffer containing (prev_exposure, new_exposure)
struct ExposureData {
    float prev_exposure;
    float new_exposure;
};
// We need to write to it
[[vk::binding(1, 0)]] RWStructuredBuffer<ExposureData> exposure_buffer;

// Shared memory for parallel reduction
groupshared float sharedWeightedLogLum[GROUP_SIZE];
groupshared uint sharedCount[GROUP_SIZE];

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gl_LocalInvocationID : SV_GroupThreadID)
{
    uint localIndex = gl_LocalInvocationID.x;
    
    // Read count from this bin
    uint count = histogram[localIndex];
    
    // Clear the bin for the next frame
    histogram[localIndex] = 0;
    
    // Exclude outlier bins (0 and 255) from average calculation
    if (localIndex == 0 || localIndex == 255) {
        sharedWeightedLogLum[localIndex] = 0.0;
        sharedCount[localIndex] = 0;
    } else {
        float fBin = float(localIndex) / float(HISTOGRAM_BINS - 1);
        float logLum = lerp(pc.min_log_lum, pc.max_log_lum, fBin);
        
        sharedWeightedLogLum[localIndex] = float(count) * logLum;
        sharedCount[localIndex] = count;
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    // Parallel reduction
    for (uint cutoff = (GROUP_SIZE >> 1); cutoff > 0; cutoff >>= 1) {
        if (localIndex < cutoff) {
            sharedWeightedLogLum[localIndex] += sharedWeightedLogLum[localIndex + cutoff];
            sharedCount[localIndex] += sharedCount[localIndex + cutoff];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    // Thread 0 calculates and writes the final exposure
    if (localIndex == 0) {
        float sumWeightedLogLum = sharedWeightedLogLum[0];
        float totalValidPixels = float(max(sharedCount[0], 1));
        
        float avgLogLum = sumWeightedLogLum / totalValidPixels;
        // Clamp to avoid extreme values
        avgLogLum = clamp(avgLogLum, pc.min_log_lum, pc.max_log_lum);
        
        float avgLum = exp2(avgLogLum);
        
        // Target exposure (Key Value)
        float keyValue = 0.18; 
        float targetExposure = keyValue / max(avgLum, 0.0001);
        
        // Ensure adaptation is not too extreme
        targetExposure = clamp(targetExposure, 0.001, 1000.0);
        
        // Get previous exposure
        float prevExposure = exposure_buffer[0].new_exposure;
        if (prevExposure <= 0.0) {
            prevExposure = targetExposure;
        }
        
        // Temporal adaptation
        // Exponential moving average: new = lerp(target, prev, exp(-rate * dt))
        float factor = exp(-pc.time_delta * pc.adaptation_rate);
        float newExposure = lerp(targetExposure, prevExposure, factor);
        
        // Write back
        ExposureData data;
        // The *new* exposure becomes the *prev* exposure for the NEXT frame's histogram & lighting pass
        data.prev_exposure = newExposure; 
        data.new_exposure = newExposure; 
        exposure_buffer[0] = data;
    }
}
