#include "common.slangh"

// ─── GPU node buffer ──────────────────────────────────────────────────────────

// All fields are float4/uint4 → matches std430 alignment exactly (6 × 16 = 96 bytes).
struct GpuSdfNode
{
    uint4 header;        // x=tag, yzw=pad
    float4 pos_scale;    // xyz=position, w=scale
    float4 rotation;     // quaternion xyzw (w=real part)
    float4 geom;         // prim: geometry params; op: x=smooth_k
    float4 albedo_rough; // xyz=albedo, w=roughness
    float4 metal_pad;    // x=metallic, yzw=pad
};

// Tag constants (must match sdf_tree.rs)
static const uint TAG_SPHERE = 0u;
static const uint TAG_BOX = 1u;
static const uint TAG_CAPSULE = 2u;
static const uint TAG_CYLINDER = 3u;
static const uint TAG_TORUS = 4u;
static const uint TAG_TERRAIN = 5u;
static const uint TAG_UNION = 10u;
static const uint TAG_SUBTRACTION = 11u;
static const uint TAG_SMOOTH_UNION = 12u;

[[vk::binding(0, 0)]]
StructuredBuffer<GpuSdfNode> sdf_nodes;

struct PushConstants
{
    uint node_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::push_constant]]
PushConstants pc;

// ─── Transform helpers ────────────────────────────────────────────────────────

float3 rotateByQuat(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

float3 applyInvTransform(float3 world_p, GpuSdfNode n)
{
    float3 p = (world_p - n.pos_scale.xyz) / n.pos_scale.w;
    float4 invRot = float4(-n.rotation.xyz, n.rotation.w);
    return rotateByQuat(invRot, p);
}

// ─── Smooth operators (for terrain) ──────────────────────────────────────────

float sminT(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return lerp(b, a, h) - k * h * (1.0 - h);
}

float smaxT(float a, float b, float k)
{
    return -sminT(-a, -b, k);
}

// ─── Hash / value noise ───────────────────────────────────────────────────────

float hash21(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float valueNoise(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    return lerp(
        lerp(hash21(i + float2(0.0, 0.0)), hash21(i + float2(1.0, 0.0)), u.x),
        lerp(hash21(i + float2(0.0, 1.0)), hash21(i + float2(1.0, 1.0)), u.x),
        u.y);
}

// ─── fBm terrain SDF (IQ bounded composition) ────────────────────────────────
// geom.x = base scale  geom.y = amplitude  geom.z = octave count

float terrainFbmSdf(float3 p, float4 geom)
{
    float hscale = geom.x;
    float amp = geom.y;
    int oct = min(int(geom.z), 12);

    float d = p.y;
    float s = 1.0;            // octave scale — always starts at 1, not hscale
    float2 q = p.xz / hscale; // horizontal scale applied to coords, not to s

    const float ca = 0.80902, sa = 0.58779; // cos/sin 36°

    for (int i = 0; i < oct; i++)
    {
        float n = amp * s * valueNoise(q / s);
        float k = 0.3 * amp * s;
        n = smaxT(n, d - 0.1 * amp * s, k);
        d = sminT(n, d, k);
        s *= 0.5;
        q = float2(ca * q.x - sa * q.y, sa * q.x + ca * q.y);
    }
    return d;
}

// ─── SDF primitives ───────────────────────────────────────────────────────────

float evalPrimitive(uint tag, float3 p, float4 g)
{
    if (tag == TAG_SPHERE)
    {
        return length(p) - g.x;
    }
    else if (tag == TAG_BOX)
    {
        float3 q = abs(p) - g.xyz;
        return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
    }
    else if (tag == TAG_CAPSULE)
    {
        float3 cp = p;
        cp.y -= clamp(cp.y, -g.x, g.x);
        return length(cp) - g.y;
    }
    else if (tag == TAG_CYLINDER)
    {
        float2 d = abs(float2(length(p.xz), p.y)) - float2(g.y, g.x);
        return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
    }
    else if (tag == TAG_TORUS)
    {
        float2 q = float2(length(p.xz) - g.x, p.y);
        return length(q) - g.y;
    }
    else // TAG_TERRAIN
    {
        return terrainFbmSdf(p, g);
    }
}

// ─── RPN evaluator ────────────────────────────────────────────────────────────
// Evaluates the CSG tree (stored in post-order) at world-space point p.
// Returns float2(distance, winning_node_index_as_float).

float2 evalSdfWithMat(float3 p)
{
    float distStack[32];
    uint matStack[32]; // node index of the material to use
    uint sp = 0u;

    for (uint i = 0u; i < pc.node_count; i++)
    {
        GpuSdfNode n = sdf_nodes[i];

        if (n.header.x < 10u) // primitive
        {
            float3 local_p = applyInvTransform(p, n);
            float d = evalPrimitive(n.header.x, local_p, n.geom);
            distStack[sp] = d;
            matStack[sp] = i;
            sp++;
        }
        else if (n.header.x == TAG_UNION)
        {
            float db = distStack[sp - 1u];
            uint mb = matStack[sp - 1u];
            float da = distStack[sp - 2u];
            uint ma = matStack[sp - 2u];
            sp--;
            bool aWins = da < db;
            distStack[sp - 1u] = aWins ? da : db;
            matStack[sp - 1u] = aWins ? ma : mb;
        }
        else if (n.header.x == TAG_SUBTRACTION)
        {
            // result = max(solid, -hole)
            float hole = distStack[sp - 1u];
            float solid = distStack[sp - 2u];
            uint ma = matStack[sp - 2u]; // solid's material always wins
            sp--;
            distStack[sp - 1u] = max(solid, -hole);
            matStack[sp - 1u] = ma;
        }
        else if (n.header.x == TAG_SMOOTH_UNION)
        {
            float k = n.geom.x;
            float db = distStack[sp - 1u];
            uint mb = matStack[sp - 1u];
            float da = distStack[sp - 2u];
            uint ma = matStack[sp - 2u];
            sp--;
            float h = clamp(0.5 + 0.5 * (db - da) / k, 0.0, 1.0);
            float d = lerp(db, da, h) - k * h * (1.0 - h);
            // material: pick the one closer to the blend point
            uint m = (h > 0.5) ? ma : mb;
            distStack[sp - 1u] = d;
            matStack[sp - 1u] = m;
        }
    }

    if (sp == 0u)
        return float2(1e10, 0.0);
    return float2(distStack[0], asfloat(matStack[0]));
}

float evalSdf(float3 p)
{
    return evalSdfWithMat(p).x;
}

float3 calcNormal(float3 p)
{
    float2 e = float2(0.001, 0.0);
    return normalize(float3(
        evalSdf(p + e.xyy) - evalSdf(p - e.xyy),
        evalSdf(p + e.yxy) - evalSdf(p - e.yxy),
        evalSdf(p + e.yyx) - evalSdf(p - e.yyx)));
}

// ─── Fullscreen vertex ────────────────────────────────────────────────────────

[shader("vertex")]
float4 vsMain(uint id: SV_VertexID) : SV_Position
{
    float2 uv = float2(float((id << 1) & 2), float(id & 2));
    return float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
}

// ─── GBuffer output ───────────────────────────────────────────────────────────

struct GBufferOut
{
    float4 albedo : SV_Target0;
    float2 normal : SV_Target1;
    float2 roughMetal : SV_Target2;
    float3 emissive : SV_Target3;
    float hizDepth : SV_Target4;
    float depth : SV_Depth;
};

[shader("fragment")]
GBufferOut fsMain(float4 svPos: SV_Position)
{
    if (pc.node_count == 0u)
        discard;

    float2 screenSize = g_common.GetScreenSize();
    float2 uv = svPos.xy / screenSize;

    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float4 wFar = mul(g_common.invViewProjection, float4(ndc, 0.0, 1.0));
    wFar /= wFar.w;

    float3 ro = g_common.GetCameraPos();
    float3 rd = normalize(wFar.xyz - ro);

    // Sphere tracing
    float t = 0.001;
    bool hit = false;
    for (int i = 0; i < 256 && t < 120.0; i++)
    {
        float d = evalSdf(ro + rd * t);
        if (d < 0.0002 * t)
        {
            hit = true;
            break;
        }
        t += d * 0.85; // slight relaxation for near-SDF terrain
    }

    if (!hit)
        discard;

    float3 hitPos = ro + rd * t;
    float3 normal = calcNormal(hitPos);

    // Fetch material from winning node
    float2 sdfResult = evalSdfWithMat(hitPos);
    uint matIdx = asuint(sdfResult.y);
    GpuSdfNode matNode = sdf_nodes[min(matIdx, pc.node_count - 1u)];

    float4 clip = mul(g_common.viewProjection, float4(hitPos, 1.0));
    float ndcDepth = clip.z / clip.w;

    GBufferOut o;
    o.albedo = float4(matNode.albedo_rough.xyz, 1.0);
    o.normal = encodeOctahedral(normal);
    o.roughMetal = float2(matNode.albedo_rough.w, matNode.metal_pad.x);
    o.emissive = float3(0.0, 0.0, 0.0);
    o.hizDepth = ndcDepth;
    o.depth = ndcDepth;
    return o;
}
