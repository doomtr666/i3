// debug_draw.slang — Debug line rendering pass.
// Vertices are pulled from a StructuredBuffer using SV_VertexID (no vertex input).
// Each vertex carries position + RGBA color. Drawn as LINE_LIST.

struct DebugVertex {
    float3 pos;
    float  _pad;
    float4 col;  // linear RGBA, pre-multiplied alpha not required (alpha blend enabled)
};

[[vk::binding(0, 0)]]
StructuredBuffer<DebugVertex> vertices;

struct PushConstants {
    float4x4 view_proj;
};
[[vk::push_constant]] PushConstants pc;

// ─── Vertex shader ───────────────────────────────────────────────────────────

struct VSOut {
    float4 sv_pos : SV_Position;
    float4 col    : COLOR0;
};

[shader("vertex")]
VSOut vs_main(uint id : SV_VertexID)
{
    DebugVertex v = vertices[id];
    VSOut o;
    o.sv_pos = mul(pc.view_proj, float4(v.pos, 1.0));
    o.col    = v.col;
    return o;
}

// ─── Fragment shader ─────────────────────────────────────────────────────────

[shader("fragment")]
float4 fs_main(VSOut i) : SV_Target
{
    return i.col;
}
