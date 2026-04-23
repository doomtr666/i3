// Stochastic Screen-Space Reflections — Sample pass
//
// One ray per pixel per frame, direction importance-sampled from GGX.

#include "common.slangh"
#include "sampling.slangh"

struct SssrSamplePc
{
    float max_distance;
    float thickness; // max view-space depth gap for hit acceptance
    float roughness_cutoff;
    uint max_steps;
    uint enabled;
};
[[vk::push_constant]]
SssrSamplePc pc;

[[vk::binding(0, 0)]]
Texture2D<float> depthBuffer;
[[vk::binding(1, 0)]]
Texture2D<float> hizBuffer;
[[vk::binding(2, 0)]]
Texture2D<float2> gbufferNormal;
[[vk::binding(3, 0)]]
Texture2D<float2> gbufferRoughMetal;
[[vk::binding(4, 0)]]
RWTexture2D<float4> ssrOutput;

// ─── Helpers ─────────────────────────────────────────────────────────────────

float3 viewSpacePos(uint2 px, float depth)
{
    float2 uv = (float2(px) + 0.5f) / float2(g_common.GetScreenWidth(), g_common.GetScreenHeight());
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float4 vh = mul(g_common.invProjection, float4(ndc, depth, 1.0f));
    return vh.xyz / vh.w;
}

float3 projectViewToScreenSpace(float3 viewPos)
{
    float4 clip = mul(g_common.projection, float4(viewPos, 1.0f));
    clip.w = max(clip.w, 1e-6f);
    float2 ndc = clip.xy / clip.w;
    return float3(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f, clip.z / clip.w);
}

// Traçage Hi-Z optimisé pour Reverse-Z (1.0 = Proche, 0.0 = Loin)
// Retourne true si une intersection valide est trouvée, et stocke la position UVZ du hit.
bool traceHiZ(
    float3 startUVZ,     // Coordonnées de départ : UV [0,1] + Depth [0,1]
    float3 rayDirUVZ,    // Direction vectorielle (P_end - P_start projeté)
    float4x4 invProj,    // Matrice inverse pour le test d'épaisseur
    float maxThickness,  // Tolérance d'épaisseur en mètres
    out float3 outHitUVZ // Résultat en UVZ (si hit)
)
{
    const int MaxMipLevel = 5;
    float2 screenSize = g_common.GetScreenSize();

    // 1. SCALER EN ESPACE PIXEL
    float3 rayPos = float3(startUVZ.xy * screenSize, startUVZ.z);
    float3 rayDir = float3(rayDirUVZ.xy * screenSize, rayDirUVZ.z);

    float rayLength = length(rayDir.xy);
    if (rayLength < 0.001f)
        return false;

    // Pré-calculs de l'algorithme DDA (Protection NaN)
    float2 dir = rayDir.xy;
    dir.x = abs(dir.x) < 1e-5f ? (dir.x >= 0.0f ? 1e-5f : -1e-5f) : dir.x;
    dir.y = abs(dir.y) < 1e-5f ? (dir.y >= 0.0f ? 1e-5f : -1e-5f) : dir.y;
    float2 invDir = 1.0f / dir;

    int2 stepDir = int2(rayDir.x >= 0.0f ? 1 : -1, rayDir.y >= 0.0f ? 1 : -1);
    float2 crossStep = float2(stepDir.x > 0 ? 1.0f : 0.0f, stepDir.y > 0 ? 1.0f : 0.0f);

    float tEpsilon = 0.01f / rayLength;

    float t = 0.0f;
    int mipLevel = 0;
    const uint MAX_ITER = 100;

    for (uint i = 0; i < MAX_ITER; i++)
    {
        float3 currentPos = rayPos + rayDir * t;

        // Condition d'arrêt : Sortie d'écran
        if (currentPos.x < 0.0f || currentPos.x >= screenSize.x ||
            currentPos.y < 0.0f || currentPos.y >= screenSize.y ||
            currentPos.z < 0.0f || currentPos.z > 1.0f)
        {
            return false;
        }

        float cellSize = exp2(float(mipLevel));
        float2 cellIndex = floor((currentPos.xy + rayDir.xy * 0.01f) / cellSize);

        float2 distanceToBoundary = (cellIndex + crossStep) * cellSize - currentPos.xy;
        float2 tToBoundary = distanceToBoundary * invDir;

        float tAdvance = min(tToBoundary.x, tToBoundary.y);
        float nextT = t + tAdvance;

        // Test d'heuristique (Reverse-Z)
        float z1 = currentPos.z;
        float z2 = rayPos.z + rayDir.z * nextT;
        float minZ_segment = min(z1, z2);

        int2 mipCoords = int2(cellIndex);
        float cellMaxZ = hizBuffer.Load(int3(mipCoords, mipLevel)).r;

        if (minZ_segment <= cellMaxZ)
        {
            if (mipLevel == 0)
            {
                // --- TEST D'ÉPAISSEUR INTÉGRÉ ---
                float2 ndc_xy = (currentPos.xy / screenSize) * 2.0f - 1.0f;
                ndc_xy.y = -ndc_xy.y; // Inversion Vulkan

                float4 ray_clip = mul(invProj, float4(ndc_xy, currentPos.z, 1.0f));
                float ray_view_z = ray_clip.z / ray_clip.w;

                float4 geo_clip = mul(invProj, float4(ndc_xy, cellMaxZ, 1.0f));
                float geo_view_z = geo_clip.z / geo_clip.w;

                float thickness = abs(geo_view_z - ray_view_z);

                if (thickness > maxThickness)
                {
                    // Faux positif : on passe derrière un objet fin (ex: guirlande).
                    // On avance et on continue à chercher.
                    t = nextT;
                    mipLevel = min(mipLevel + 1, MaxMipLevel);
                    continue;
                }

                // Vrai contact
                outHitUVZ = float3(currentPos.xy / screenSize, currentPos.z);
                return true;
            }
            else
            {
                mipLevel--;
            }
        }
        else
        {
            t = nextT + tEpsilon;
            mipLevel = min(mipLevel + 1, MaxMipLevel);
        }
    }

    return false;
}

[numthreads(8, 8, 1)]
void computeMain(uint3 id: SV_DispatchThreadID)
{
    uint2 px = id.xy;
    if (any(px >= uint2(g_common.GetScreenWidth(), g_common.GetScreenHeight())))
        return;

    if (pc.enabled == 0u)
    {
        ssrOutput[px] = 0;
        return;
    }

    float depth = depthBuffer.Load(int3(px, 0)).r;
    if (depth <= 0.0f)
    {
        ssrOutput[int2(px)] = 0;
        return;
    }

    float roughness = gbufferRoughMetal.Load(int3(px, 0)).r;

    float3 P = viewSpacePos(px, depth);
    float3 N_world = decodeOctahedral(gbufferNormal.Load(int3(px, 0)).rg);
    float3x3 viewRot = float3x3(g_common.view[0].xyz, g_common.view[1].xyz, g_common.view[2].xyz);
    float3 N = normalize(mul(viewRot, N_world));
    float3 V = normalize(-P);

    float3 noise = pcg3f(uint3(px, g_common.GetFrameIndex()));

    // 1. Construire la base
    float3 T, B;
    buildBasis(N, T, B);

    // 2. Transformer V en espace tangent
    float3 V_tangent = float3(dot(V, T), dot(V, B), dot(V, N));

    // 3. Sampler avec le V tangentiel
    float3 H_local = sampleGGXVNDF(V_tangent, roughness, roughness, noise.x, noise.y);

    // 4. Repasser H en View Space
    float3 H = normalize(T * H_local.x + B * H_local.y + N * H_local.z);

    float3 D = reflect(-V, H);

    // --- Calcul de l'Indice de Confiance (Fade) ---
    float confidence = 1.0f;

    // Rejet si le rayon plonge sous la géométrie
    if (dot(D, N) <= 0.001f)
    {
        ssrOutput[px] = float4(0, 0, 0, 0);
        return;
    }

    // Fondu directionnel pour la caméra (évite les hard cuts sur les vitres)
    float fade_camera = smoothstep(0.1f, -0.1f, D.z);
    confidence *= fade_camera;

    // Calcul du point de fin du rayon (ex: 100 mètres)
    float3 P_end = P + D * 100.0f;

    float3 screen_P = projectViewToScreenSpace(P);
    float3 screen_P_end = projectViewToScreenSpace(P_end);

    float3 screen_RayDir = normalize(screen_P_end - screen_P);

    float2 screen_RayDir_Pixels = screen_RayDir.xy * g_common.GetScreenSize();
    float step_scale = 1.0f / max(abs(screen_RayDir_Pixels.x), abs(screen_RayDir_Pixels.y));
    float3 ray_step_screen = screen_RayDir * step_scale;

    // --- LE FIX (Ray Bump) ---
    // Offset pour éviter l'auto-intersection
    float start_offset = 4.0f + noise.z;
    float3 safe_start_P = screen_P + ray_step_screen * start_offset;

    // Lancer avec test d'épaisseur intégré
    float3 hit_UVZ;
    bool valid_hit = traceHiZ(safe_start_P, ray_step_screen, g_common.invProjection, pc.thickness, hit_UVZ);

    if (!valid_hit)
    {
        // On écrit du noir/vide pur pour que le TAA l'efface proprement
        ssrOutput[px] = float4(0, 0, 0, 0);
        return;
    }

    // --- Atténuations post-hit ---
    // 1. Vignettage écran
    float2 edge_dist = smoothstep(0.0f, 0.05f, hit_UVZ.xy) * smoothstep(1.0f, 0.95f, hit_UVZ.xy);
    confidence *= (edge_dist.x * edge_dist.y);

    // 2. Fondu sur la rugosité
    confidence *= 1.0f - smoothstep(0.6f, 0.8f, roughness);

    // 3. Validation Backface
    uint2 hit_px = uint2(hit_UVZ.xy * g_common.GetScreenSize());
    float3 hit_N_world = decodeOctahedral(gbufferNormal.Load(int3(hit_px, 0)).rg);
    float3 hit_N = normalize(mul(viewRot, hit_N_world)); // On le repasse en View Space !
    float N_dot_D = dot(hit_N, D);

    confidence *= smoothstep(0.1f, -0.1f, N_dot_D);

    ssrOutput[px] = float4(hit_UVZ, confidence);
}
