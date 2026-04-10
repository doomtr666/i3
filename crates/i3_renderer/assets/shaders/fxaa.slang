// assets/system/shaders/fxaa.slang
#include "common.slangh"
// FXAA 3.11 — Nvidia Timothy Lottes, adapted for Slang / Vulkan
// Input: LDR tonemapped image (R8G8B8A8_UNORM or swapchain format)

struct PushConstants
{
    float2 inv_screen_size; // 1/width, 1/height
    float  subpix;          // 0.0–1.0  (default 0.75 — subpixel AA strength)
    float  edge_threshold;  // 0.063–0.333 (default 0.125 — min local contrast to trigger)
    uint   enabled;         // 0 = passthrough blit, 1 = FXAA
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]] Texture2D   ldrInput;

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VertexID)
{
    VertexOutput o;
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    o.uv       = uv;
    o.position = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    return o;
}

// --- Helpers -----------------------------------------------------------------

// Perceptual luma: sqrt approximates gamma 2.0, matching FXAA's design assumption
// of operating on gamma-encoded (SRGB) input. On linear LDR, this restores correct
// edge contrast in dark areas where linear luma deltas are too small.
float luma(float3 c) { return sqrt(dot(c, float3(0.299, 0.587, 0.114))); }

float3 sampleOffset(Texture2D tex, SamplerState s, float2 uv, float2 off)
{
    return tex.SampleLevel(s, uv + off * pc.inv_screen_size, 0).rgb;
}

// --- FXAA --------------------------------------------------------------------

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    float2 uv  = input.uv;
    float2 inv = pc.inv_screen_size;

    // Luma of center + 4 neighbours
    float3 cC  = ldrInput.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uv, 0).rgb;

    if (pc.enabled == 0)
        return float4(cC, 1.0);

    float3 cN  = sampleOffset(ldrInput, bindlessSamplers[SAMPLER_LINEAR], uv, float2( 0,-1));
    float3 cS  = sampleOffset(ldrInput, bindlessSamplers[SAMPLER_LINEAR], uv, float2( 0, 1));
    float3 cE  = sampleOffset(ldrInput, bindlessSamplers[SAMPLER_LINEAR], uv, float2( 1, 0));
    float3 cW  = sampleOffset(ldrInput, bindlessSamplers[SAMPLER_LINEAR], uv, float2(-1, 0));

    float lumaC  = luma(cC);
    float lumaN  = luma(cN);
    float lumaS  = luma(cS);
    float lumaE  = luma(cE);
    float lumaW  = luma(cW);

    float rangeMin = min(lumaC, min(min(lumaN, lumaS), min(lumaE, lumaW)));
    float rangeMax = max(lumaC, max(max(lumaN, lumaS), max(lumaE, lumaW)));
    float range    = rangeMax - rangeMin;

    // Early exit: below contrast threshold → return unfiltered
    if (range < max(0.0312, rangeMax * pc.edge_threshold))
        return float4(cC, 1.0);

    // Diagonal neighbours for direction + subpixel blend
    float3 cNW = sampleOffset(ldrInput, bindlessSamplers[SAMPLER_LINEAR], uv, float2(-1,-1));
    float3 cNE = sampleOffset(ldrInput, bindlessSamplers[SAMPLER_LINEAR], uv, float2( 1,-1));
    float3 cSW = sampleOffset(ldrInput, bindlessSamplers[SAMPLER_LINEAR], uv, float2(-1, 1));
    float3 cSE = sampleOffset(ldrInput, bindlessSamplers[SAMPLER_LINEAR], uv, float2( 1, 1));

    float lumaNW = luma(cNW); float lumaNE = luma(cNE);
    float lumaSW = luma(cSW); float lumaSE = luma(cSE);

    // Subpixel blend factor
    float lumaAvg = (lumaN + lumaS + lumaE + lumaW) * 0.25;
    float subpixBlend = abs(lumaAvg - lumaC) / range;
    subpixBlend = smoothstep(0.0, 1.0, subpixBlend);
    subpixBlend = subpixBlend * subpixBlend * pc.subpix;

    // Edge direction: horizontal vs vertical
    float edgeH = abs(lumaN  + lumaS  - 2.0*lumaC) * 2.0
                + abs(lumaNE + lumaSE - 2.0*lumaE)
                + abs(lumaNW + lumaSW - 2.0*lumaW);
    float edgeV = abs(lumaE  + lumaW  - 2.0*lumaC) * 2.0
                + abs(lumaNE + lumaNW - 2.0*lumaN)
                + abs(lumaSE + lumaSW - 2.0*lumaS);

    bool isHorizontal = (edgeH >= edgeV);

    // Step perpendicular to the edge, pick positive/negative direction
    float luma1 = isHorizontal ? lumaN  : lumaE;
    float luma2 = isHorizontal ? lumaS  : lumaW;
    float grad1 = abs(luma1 - lumaC);
    float grad2 = abs(luma2 - lumaC);

    bool isPos = (grad1 >= grad2);
    float stepDir = isHorizontal ? (isPos ? -inv.y : inv.y)
                                 : (isPos ?  inv.x : -inv.x);

    float2 uvEdge = uv;
    if (isHorizontal) uvEdge.y += stepDir * 0.5;
    else              uvEdge.x += stepDir * 0.5;

    // Walk along the edge to find its ends (up to 8 steps each side)
    float2 edgeStep = isHorizontal ? float2(inv.x, 0) : float2(0, inv.y);
    float lumaEdge = (luma1 + lumaC) * 0.5;
    float gradThreshold = max(grad1, grad2) * 0.25;

    float2 uvP = uvEdge + edgeStep;
    float2 uvN = uvEdge - edgeStep;
    float deltaP = luma(ldrInput.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uvP, 0).rgb) - lumaEdge;
    float deltaN = luma(ldrInput.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uvN, 0).rgb) - lumaEdge;
    bool doneP = abs(deltaP) >= gradThreshold;
    bool doneN = abs(deltaN) >= gradThreshold;

    [unroll]
    for (int i = 0; i < 7; ++i)
    {
        if (!doneP) { uvP += edgeStep; deltaP = luma(ldrInput.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uvP, 0).rgb) - lumaEdge; }
        if (!doneN) { uvN -= edgeStep; deltaN = luma(ldrInput.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uvN, 0).rgb) - lumaEdge; }
        doneP = doneP || abs(deltaP) >= gradThreshold;
        doneN = doneN || abs(deltaN) >= gradThreshold;
        if (doneP && doneN) break;
    }

    float distP = isHorizontal ? (uvP.x - uv.x) : (uvP.y - uv.y);
    float distN = isHorizontal ? (uv.x - uvN.x) : (uv.y - uvN.y);

    float spanLen = distP + distN;
    float pixelOff = -min(distP, distN) / spanLen + 0.5;

    // Only blend if end luma delta has same sign as center offset
    bool goodSpan = ((deltaP < 0.0) != (lumaC < lumaEdge)) ? (distP < distN)
                                                            : (distN < distP);
    float blendFinal = max(subpixBlend, goodSpan ? pixelOff : 0.0);

    float2 uvFinal = uv;
    if (isHorizontal) uvFinal.y += blendFinal * stepDir;
    else              uvFinal.x += blendFinal * stepDir;

    float3 result = ldrInput.SampleLevel(bindlessSamplers[SAMPLER_LINEAR], uvFinal, 0).rgb;
    return float4(result, 1.0);
}
