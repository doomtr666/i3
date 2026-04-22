[[vk::binding(0, 0)]] RWTexture2D<float> aoResolved;

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 dtid : SV_DispatchThreadID) {
    uint w, h;
    aoResolved.GetDimensions(w, h);
    if (dtid.x >= w || dtid.y >= h) return;
    aoResolved[dtid.xy] = 1.0f;
}
