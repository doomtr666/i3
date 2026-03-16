// assets/system/shaders/histogram_build.slang
#include "common.slangh"

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
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> histogram;
[[vk::binding(2, 0)]] StructuredBuffer<ExposureData> exposure_buffer;

float luminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

uint colorToBin(float3 color, float minLogLum, float maxLogLum) {
    float lum = luminance(color);
    if (lum < 0.0001) {
        return 0;
    }
    float logLum = log2(lum);
    float fBin = (logLum - minLogLum) / (maxLogLum - minLogLum);
    return uint(clamp(fBin, 0.0, 1.0) * float(HISTOGRAM_BINS - 1));
}

groupshared uint sharedHistogram[HISTOGRAM_BINS];

[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID, uint gl_LocalInvocationIndex : SV_GroupIndex)
{
    sharedHistogram[gl_LocalInvocationIndex] = 0;
    GroupMemoryBarrierWithGroupSync();

    uint2 texDimensions;
    hdrImage.GetDimensions(texDimensions.x, texDimensions.y);
    
    if (gl_GlobalInvocationID.x < texDimensions.x && gl_GlobalInvocationID.y < texDimensions.y) {
        float3 color = hdrImage.Load(int3(gl_GlobalInvocationID.xy, 0)).rgb;
        
        float prev_ex = exposure_buffer[0].prev_exposure;
        if (prev_ex == 0.0) { prev_ex = 1.0; }
        color /= prev_ex;
        
        uint binIndex = colorToBin(color, pc.min_log_lum, pc.max_log_lum);
        InterlockedAdd(sharedHistogram[binIndex], 1);
    }
    
    GroupMemoryBarrierWithGroupSync();
    InterlockedAdd(histogram[gl_LocalInvocationIndex], sharedHistogram[gl_LocalInvocationIndex]);
}
