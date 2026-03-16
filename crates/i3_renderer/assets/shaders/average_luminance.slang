// assets/system/shaders/average_luminance.slang
#include "common.slangh"

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

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> histogram;
[[vk::binding(1, 0)]] RWStructuredBuffer<ExposureData> exposure_buffer;

groupshared float sharedWeightedLogLum[GROUP_SIZE];
groupshared uint sharedCount[GROUP_SIZE];

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gl_LocalInvocationID : SV_GroupThreadID)
{
    uint localIndex = gl_LocalInvocationID.x;
    
    uint count = histogram[localIndex];
    histogram[localIndex] = 0;
    
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
    
    for (uint cutoff = (GROUP_SIZE >> 1); cutoff > 0; cutoff >>= 1) {
        if (localIndex < cutoff) {
            sharedWeightedLogLum[localIndex] += sharedWeightedLogLum[localIndex + cutoff];
            sharedCount[localIndex] += sharedCount[localIndex + cutoff];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    if (localIndex == 0) {
        float sumWeightedLogLum = sharedWeightedLogLum[0];
        float totalValidPixels = float(max(sharedCount[0], 1));
        
        float avgLogLum = sumWeightedLogLum / totalValidPixels;
        avgLogLum = clamp(avgLogLum, pc.min_log_lum, pc.max_log_lum);
        
        float avgLum = exp2(avgLogLum);
        float keyValue = 0.18; 
        float targetExposure = keyValue / max(avgLum, 0.0001);
        
        targetExposure = clamp(targetExposure, 0.001, 1000.0);
        
        float prevExposure = exposure_buffer[0].new_exposure;
        if (prevExposure <= 0.0) {
            prevExposure = targetExposure;
        }
        
        float factor = exp(-pc.time_delta * pc.adaptation_rate);
        float newExposure = lerp(targetExposure, prevExposure, factor);
        
        ExposureData data;
        data.prev_exposure = newExposure; 
        data.new_exposure = newExposure; 
        exposure_buffer[0] = data;
    }
}
