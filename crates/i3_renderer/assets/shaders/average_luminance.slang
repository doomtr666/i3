// assets/system/shaders/average_luminance.slang
#include "common.slangh"

#define HISTOGRAM_BINS 256
#define GROUP_SIZE 256
// GROUP_SIZE / minimum_subgroup_size(8) — handles NVIDIA(32), AMD(32/64), Intel(8..32)
#define MAX_WAVES 32

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
[[vk::binding(1, 0)]] StructuredBuffer<ExposureData> history_buffer;
[[vk::binding(2, 0)]] RWStructuredBuffer<ExposureData> exposure_buffer;

groupshared float sharedWaveLum[MAX_WAVES];
groupshared uint  sharedWaveCount[MAX_WAVES];

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gl_LocalInvocationID : SV_GroupThreadID)
{
    uint localIndex = gl_LocalInvocationID.x;

    uint count = histogram[localIndex];
    histogram[localIndex] = 0;

    float weightedLogLum;
    uint validCount;
    if (localIndex == 0 || localIndex == 255) {
        weightedLogLum = 0.0;
        validCount = 0;
    } else {
        float fBin = float(localIndex) / float(HISTOGRAM_BINS - 1);
        float logLum = lerp(pc.min_log_lum, pc.max_log_lum, fBin);
        weightedLogLum = float(count) * logLum;
        validCount = count;
    }

    // Intra-wave reduction via hardware subgroup ops — no shared memory, no possible race.
    float waveLumSum   = WaveActiveSum(weightedLogLum);
    uint  waveCountSum = WaveActiveSum(validCount);

    // Each wave's first lane writes its partial sum to a distinct shared-memory slot.
    // Different waves write to different waveIdx values, so there is no conflict.
    if (WaveIsFirstLane()) {
        uint waveIdx = localIndex / WaveGetLaneCount();
        sharedWaveLum[waveIdx]   = waveLumSum;
        sharedWaveCount[waveIdx] = waveCountSum;
    }

    GroupMemoryBarrierWithGroupSync();

    // Thread 0 serially sums the per-wave partials — single accessor, no race possible.
    if (localIndex == 0) {
        uint numWaves = (GROUP_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount();
        float sumWeightedLogLum = 0.0;
        uint  totalValidPixels  = 0u;
        for (uint w = 0; w < numWaves; w++) {
            sumWeightedLogLum += sharedWaveLum[w];
            totalValidPixels  += sharedWaveCount[w];
        }

        float avgLogLum = sumWeightedLogLum / float(max(totalValidPixels, 1u));
        avgLogLum = clamp(avgLogLum, pc.min_log_lum, pc.max_log_lum);

        float avgLum = exp2(avgLogLum);
        float keyValue = 0.18;
        float targetExposure = keyValue / max(avgLum, 0.0001);
        targetExposure = clamp(targetExposure, 0.001, 1000.0);

        float prevExposure = history_buffer[0].new_exposure;
        if (prevExposure <= 0.0) {
            prevExposure = targetExposure;
        }

        float factor = exp(-pc.time_delta * pc.adaptation_rate);
        float newExposure = lerp(targetExposure, prevExposure, factor);

        ExposureData data;
        data.prev_exposure = prevExposure;
        data.new_exposure = newExposure;
        exposure_buffer[0] = data;
    }
}
