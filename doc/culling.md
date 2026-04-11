# G-Buffer + Hi-Z — Plan d'implémentation

> Référence : `doc/gpu_driven_plan.md` (OcclusionCullGroup, Phase 1b Niveau 2).  
> État actuel : GPU-driven frustum cull opérationnel (`FrustumCullPass`), `DepthBuffer` D32_FLOAT présent dans le render graph.  
> Objectif : pipeline G-Buffer + Hi-Z cohérent, servant à la fois l'occlusion culling two-pass et les effets screen-space (GTAO, SSR, light culling, transparents).

---

## Vue d'ensemble du pipeline par frame

```
┌──────────────────────────────────────────────────────────────────────┐
│ Frame N                                                              │
│                                                                      │
│  ┌──────────────────┐    ┌─────────────────────────┐                │
│  │ 1. CullingPass   │    │ Hi-Z N-1 (frame précéd.)│                │
│  │    (Compute)     │◄───│ depuis history buffer    │                │
│  └────────┬─────────┘    └─────────────────────────┘                │
│           │ IndirectBuffer (DrawCallBuffer + DrawCountBuffer)        │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 2. GBufferPass   │  → DepthBuffer N (D32_FLOAT)                  │
│  │    (Graphics)    │  → GBuffer_Albedo / Normal / RoughMetal /      │
│  │  early_fragment_ │     Emissive                                   │
│  │  tests activé   │                                                 │
│  └────────┬─────────┘                                                │
│           │ DepthBuffer N (complet)                                  │
│           ▼                                                          │
│  ┌──────────────────┐                                                │
│  │ 3. HiZBuildPass  │  SPD sur DepthBuffer N → HiZPyramid N         │
│  │    (Compute)     │  (mip0 = full-res, mip_k = 2^k × 2^k tiles)  │
│  └────────┬─────────┘                                                │
│           │ HiZPyramid N (image mip-chain)                           │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ 4. Passes "Accurate" (Compute)                               │    │
│  │    GTAO · SSR · LightCulling · CullingTransparents           │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  HiZPyramid N devient Hi-Z N+1 (history) pour le CullingPass N+1   │
└──────────────────────────────────────────────────────────────────────┘
```

**Invariant de frame :** la passe 1 lit Hi-Z N-1 (produit lors de la frame précédente). Les passes 4 lisent Hi-Z N (produit ce même frame). Il y a donc **une latence d'1 frame** pour le culling — acceptable (artefact "ghost" d'1 frame si un occludeur disparaît brutalement).

---

## Phase 1 — CullingPass (Compute)

### Rôle

Remplace / étend le `FrustumCullPass` existant. Ajoute le test d'occlusion via le Hi-Z N-1 temporal.

### Données d'entrée

| Ressource | Type | Source |
|---|---|---|
| `InstanceBuffer` | SSBO `GpuInstanceData[]` | `InstanceSyncPass` |
| `MeshDescriptorBuffer` | SSBO `GpuMeshDescriptor[]` | `MeshRegistrySyncPass` |
| `HiZPyramid` | Image (mip-chain, `R32_SFLOAT`) | Frame N-1 (history) |
| Push constants | `view_proj`, `prev_view_proj`, `instance_count`, `hiz_mip_count`, `screen_size` | CPU |

### Données de sortie

| Ressource | Type | Consommé par |
|---|---|---|
| `DrawCallBuffer` | `VkDrawIndirectCommand[]` | GBufferPass |
| `DrawCountBuffer` | `u32` | GBufferPass |
| `VisibleInstanceBuffer` | `u32[]` | GBufferPass VS |

### Algorithme shader (`cull.slang`)

```slang
// 1 thread par instance
uint i = gl_GlobalInvocationID.x;
if (i >= push.instance_count) return;

GpuInstanceData   inst = instances[i];
GpuMeshDescriptor mesh = meshes[inst.mesh_idx];

// --- Frustum cull (AABB world-space vs 6 plans) ---
float3 aabb_min = inst.world_aabb_min;
float3 aabb_max = inst.world_aabb_max;
if (!frustum_cull(aabb_min, aabb_max, push.view_proj)) return;

// --- Reprojection AABB → NDC frame N-1 (reverse-Z : near=1.0, far=0.0) ---
// Projeter les 8 coins de l'AABB avec prev_view_proj
float2 ndc_min, ndc_max;
float  ndc_z_max;   // depth MAX des coins = point le PLUS PROCHE de la caméra en reverse-Z
project_aabb_ndc(aabb_min, aabb_max, push.prev_view_proj,
                 ndc_min, ndc_max, ndc_z_max);

// Taille screen-space → niveau de mip
float2 screen_extent = (ndc_max - ndc_min) * 0.5 * float2(push.screen_size);
float  mip_level     = ceil(log2(max(screen_extent.x, screen_extent.y)));
mip_level            = clamp(mip_level, 0.0, float(push.hiz_mip_count - 1));

// --- Sample Hi-Z N-1 ---
// Hi-Z stocke le MAX de profondeur par tile.
// En reverse-Z, MAX = surface la plus proche de la caméra dans le tile = occludeur potentiel.
float2 uv_center  = (ndc_min + ndc_max) * 0.5 * 0.5 + 0.5;
float  hiz_max    = hiz_pyramid.SampleLevel(sampler_nearest, uv_center, mip_level).r;

// Test d'occlusion reverse-Z :
//   ndc_z_max < hiz_max  →  le coin le plus proche de l'objet est encore PLUS LOIN
//                            que l'occludeur du tile → l'objet est derrière → cull.
if (ndc_z_max < hiz_max) return;

// --- Émettre le draw ---
uint slot = atomicAdd(draw_count, 1u);
visible_ids[slot]  = i;
draws[slot].vertexCount   = 0;              // non utilisé (indexed)
draws[slot].instanceCount = 1;
draws[slot].firstVertex   = 0;
draws[slot].firstInstance = slot;           // → gl_BaseInstance dans VS
```

> **Note reverse-Z** : le moteur utilise `D32_FLOAT` avec projection reverse-Z ? À confirmer avec `GBufferPushConstants::view_projection`. Adapter le test d'occlusion en conséquence (closer = depth plus grand en reverse-Z).

### Struct Rust

```rust
// crates/i3_renderer/src/passes/culling.rs  (nouveau fichier ou remplacement de cull.rs)

pub struct CullingPass {
    instance_buffer:         BufferHandle,
    mesh_descriptor_buffer:  BufferHandle,
    hiz_pyramid:             ImageHandle,    // Hi-Z N-1 (history)
    draw_call_buffer:        BufferHandle,
    draw_count_buffer:       BufferHandle,
    visible_instance_buffer: BufferHandle,
    pipeline:                Option<BackendPipeline>,
}
```

### Déclarations FrameGraph

```rust
fn declare(&mut self, builder: &mut PassBuilder) {
    self.instance_buffer        = builder.resolve_buffer("InstanceBuffer");
    self.mesh_descriptor_buffer = builder.resolve_buffer("MeshDescriptorBuffer");
    // Hi-Z N-1 : history image (aliasé sur HiZPyramid du frame précédent)
    self.hiz_pyramid            = builder.resolve_image("HiZPyramid");

    self.draw_call_buffer        = builder.resolve_buffer("DrawCallBuffer");
    self.draw_count_buffer       = builder.resolve_buffer("DrawCountBuffer");
    self.visible_instance_buffer = builder.resolve_buffer("VisibleInstanceBuffer");

    builder.read_buffer(self.instance_buffer,        ResourceUsage::SHADER_READ);
    builder.read_buffer(self.mesh_descriptor_buffer, ResourceUsage::SHADER_READ);
    builder.read_image (self.hiz_pyramid,            ResourceUsage::SHADER_READ);

    builder.write_buffer(self.draw_call_buffer,        ResourceUsage::SHADER_WRITE);
    builder.write_buffer(self.draw_count_buffer,       ResourceUsage::SHADER_WRITE);
    builder.write_buffer(self.visible_instance_buffer, ResourceUsage::SHADER_WRITE);
}
```

> **History buffer :** le frame graph supporte déjà les ressources temporelles (nombre de copies géré dynamiquement selon le buffering en cours). Déclarer `HiZPyramid` comme ressource history via le mécanisme existant — le graph garantit que la frame N lit la copie produite en frame N-1.

---

## Phase 2 — GBufferPass (Graphics)

### Ce qui change par rapport à l'état actuel

Le `GBufferFillPass` existant est **quasi-conforme**. Les modifications nécessaires sont minimales :

#### 2a. Early Fragment Tests

Dans `gbuffer.slang` (fragment shader), ajouter en tête de fichier :

```slang
// Garantit que le depth test HW tue les fragments avant l'exécution du FS.
// Requis : pas d'écriture manuelle à gl_FragDepth dans le FS.
[earlydepthstencil]    // Slang attribute
```

En GLSL équivalent : `layout(early_fragment_tests) in;`

> **Prérequis absolu** : le FS ne doit jamais écrire `gl_FragDepth`. Si un discard conditionnel (alpha-test) est présent, `early_fragment_tests` est inutilisable — il faudra un Z-prepass séparé pour ces shaders.

#### 2b. `DepthBuffer` usage flags

Dans `GBufferPass::declare()`, `DepthBuffer` doit déclarer un usage composite pour être lu par `HiZBuildPass` :

```rust
builder.declare_image_output("DepthBuffer", ImageDesc {
    format: Format::D32_FLOAT,
    usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
         | ImageUsageFlags::SAMPLED,            // ← déjà présent
    mip_levels: 1,
    // ...
});
```

Le flag `SAMPLED` est **déjà présent** dans le code actuel (`gbuffer.rs:210`). Aucune modification nécessaire ici.

#### 2c. `draw_indirect` → `draw_indirect_count`

Le `GBufferFillPass::execute()` appelle déjà `ctx.draw_indirect_count(...)` ✓. Aucune modification.

### Point d'attention : initialisation de DrawCountBuffer

Le `DrawCountBuffer` doit être remis à zéro **avant** chaque `CullingPass`. Options :

1. Un `ClearBufferPass` trivial (compute : `draw_count = 0`) en tête de `CullGroup`.
2. Ou `vkCmdFillBuffer` via une commande dédiée dans `PassContext`.

> Recommandation : ajouter `fn fill_buffer(&mut self, buffer: BufferHandle, offset: u64, size: u64, value: u32)` dans le trait `PassContext`, mappé sur `vkCmdFillBuffer`. Simple, efficace, sans overhead.

---

## Phase 3 — HiZBuildPass (Compute)

### Rôle

Générer la pyramide Hi-Z à partir du `DepthBuffer` frais de la frame courante, pour alimenter :
- le `CullingPass` N+1 (frame suivante)
- les passes compute "accurate" (GTAO, SSR, LightCulling, CullingTransparents) dans la **même frame**

### Algorithme : Single Pass Downsampler (SPD)

La construction naïve (un dispatch par niveau de mip) est correcte mais sous-optimale. Utiliser l'algorithme **AMD SPD** (Single Pass Downsampler) :

- Un seul dispatch compute pour tous les niveaux de mip.
- Exploite la shared memory pour la réduction hiérarchique intra-workgroup.
- Réduit la pression sur le bandwidth mémoire (les mips intermédiaires restent en L2/SRAM).
- Référence publique : [GPUOpen SPD](https://gpuopen.com/fidelityfx-spd/).

> Pour i3 : implémenter une version simplifiée en Slang, sans la dépendance FidelityFX. Le cœur de l'algorithme tient en ~150 lignes de GLSL/Slang.

#### Opérateur de réduction

En **reverse-Z** (near=1.0, far=0.0), la surface la plus proche de la caméra a la valeur de profondeur la **plus grande**. Le Hi-Z stocke donc le **MAX** par tile — c'est l'occludeur potentiel le plus proche.

```slang
// Réduction d'un quad 2×2 : MAX des 4 profondeurs (occludeur le plus proche en reverse-Z)
float reduce_4(float4 s) {
    return max(max(s.x, s.y), max(s.z, s.w));
}
```

> **Cas particulier de la résolution impaire** : si `width` ou `height` n'est pas une puissance de 2, le dernier pixel de la ligne/colonne est doublé dans la réduction. Le sampler doit utiliser `CLAMP_TO_EDGE`.

### Format de HiZPyramid

```rust
// Image persistante dans GpuBuffers
pub hiz_pyramid: BackendImage,

// Description
ImageDesc {
    format:     Format::R32_SFLOAT,     // profondeur copiée depuis D32_FLOAT
    width:      next_pow2(screen_w),    // puissance de 2 supérieure
    height:     next_pow2(screen_h),
    mip_levels: ceil(log2(max(w, h))) + 1,
    usage: ImageUsageFlags::STORAGE       // écriture mip N par compute
         | ImageUsageFlags::SAMPLED,      // lecture par CullingPass et passes accurate
    view_type: ImageViewType::Type2D,
}
```

> Le format `R32_SFLOAT` est nécessaire car `D32_FLOAT` ne peut pas être utilisé comme `STORAGE_IMAGE` en Vulkan. La conversion depth → color est faite au mip0. Alternativement, utiliser une image `D32_FLOAT` + `imageAtomic` n'est pas supporté ; la copie vers R32_SFLOAT reste la solution canonique.

### Struct Rust

```rust
// crates/i3_renderer/src/passes/hiz_build.rs

pub struct HiZBuildPass {
    depth_buffer: ImageHandle,      // input : DepthBuffer N (D32_FLOAT)
    hiz_pyramid:  ImageHandle,      // output : HiZPyramid N (R32_SFLOAT, tous mips)
    mip_count:    u32,
    pipeline:     Option<BackendPipeline>,
}
```

### Déclarations FrameGraph

```rust
fn declare(&mut self, builder: &mut PassBuilder) {
    self.depth_buffer = builder.resolve_image("DepthBuffer");
    self.hiz_pyramid  = builder.resolve_image("HiZPyramid");

    builder.read_image (self.depth_buffer, ResourceUsage::SHADER_READ);
    // Chaque mip est écrit indépendamment ; le frame graph génère la barrière globale
    builder.write_image(self.hiz_pyramid,  ResourceUsage::SHADER_WRITE);
}
```

### Dispatch

```rust
fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
    let Some(pipeline) = self.pipeline else { return; };
    ctx.bind_pipeline_raw(pipeline);
    // SPD : un seul dispatch couvre tous les mips
    // Workgroup 8×8, couvre toute la surface au niveau mip0
    let wg_x = (self.width  + 63) / 64;
    let wg_y = (self.height + 63) / 64;
    ctx.dispatch(wg_x, wg_y, 1);
}
```

---

## Phase 4 — Passes "Accurate" (Compute)

Ces passes lisent `HiZPyramid N` produit dans la **même frame** par `HiZBuildPass`. Le frame graph garantit la barrière `SHADER_WRITE → SHADER_READ` entre les deux.

### 4a. GTAO (Ground Truth Ambient Occlusion)

Accès Hi-Z : échantillonnage multi-directions en screen-space.

```
HiZPyramid N → GTAO Compute → AO buffer (R8_UNORM)
```

Le Hi-Z permet de **rejeter rapidement** les ray-marching steps derrière la géométrie — réduction du nombre de samples nécessaires.

### 4b. SSR (Screen-Space Reflections)

Accès Hi-Z : ray-marching hiérarchique ("Hi-Z tracing").

```slang
// Hi-Z tracing : avance le ray au niveau de mip le plus grossier possible
// tant qu'il n'intersecte pas, puis descend de mip pour affiner.
// Complexité : O(log(screen_size)) au lieu de O(screen_size).
while (ray_active && mip >= 0) {
    float hiz_depth = hiz.SampleLevel(smp, uv, mip).r;
    if (ray_depth_behind(ray.z, hiz_depth)) {
        mip--;  // raffiner
    } else {
        ray.uv += ray.step_at_mip(mip);  // avancer
        mip     = min(mip + 1, max_mip);  // remonter
    }
}
```

### 4c. Light Occlusion Culling (pré-cluster)

Le `LightCullPass` clustered affecte les lumières aux clusters visibles. Avant ce travail, on peut éliminer en amont les lumières dont **toute la sphère d'influence est derrière la géométrie visible** — elles ne peuvent illuminer aucun pixel à l'écran.

C'est un test **par lumière** (pas par cluster), effectué dans une passe compute légère avant `LightCullPass`.

#### Algorithme (`light_occlusion_cull.slang`)

```slang
// 1 thread par lumière active
uint light_idx = gl_GlobalInvocationID.x;
if (light_idx >= push.light_count) return;

GpuLight light = lights[light_idx];

// Projeter la sphère d'influence du light en NDC (prev_view_proj ou view_proj selon usage)
float3 center = light.world_position;
float  radius = light.influence_radius;

// Projeter le centre + calculator le footprint 2D
float4 ndc_center = mul(push.view_proj, float4(center, 1.0));
if (ndc_center.w <= 0.0) return;   // derrière la caméra
ndc_center.xyz /= ndc_center.w;

// Approximation screenspace foot : on projette deux points sur l'axe caméra-lumière
// séparés de radius pour estimer la taille en pixels
float  proj_radius = radius / ndc_center.w;   // approx NDC radius
float2 screen_size  = float2(push.screen_w, push.screen_h);
float2 uv_center    = ndc_center.xy * 0.5 + 0.5;

// Mip correspondant au footprint de la sphère
float  footprint_px = proj_radius * max(screen_size.x, screen_size.y);
float  mip_level    = ceil(log2(max(footprint_px, 1.0)));
mip_level           = clamp(mip_level, 0.0, float(push.hiz_mip_count - 1));

// Profondeur NDC du front-face de la sphère (le plus proche de la caméra)
// En reverse-Z : point le plus proche = depth MAX.
// ndc_center.z est la depth du centre. On ajoute la contribution du radius en NDC.
float light_front_depth = ndc_center.z + proj_radius;  // approximatif, suffit pour le cull
light_front_depth = saturate(light_front_depth);

// Sample Hi-Z N (même frame, déjà produit)
float hiz_max = hiz_pyramid.SampleLevel(smp_nearest, uv_center, mip_level).r;

// Test occlusion reverse-Z :
//   light_front_depth < hiz_max → le front de la sphère est plus loin que l'occludeur → cull
if (light_front_depth < hiz_max) return;   // lumière entièrement occultée

// Lumière visible → la conserver dans la liste active
uint slot = atomicAdd(visible_light_count, 1u);
visible_light_ids[slot] = light_idx;
```

#### Intégration dans le frame graph

```
HiZBuildPass (Compute)
    ↓ HiZPyramid N
LightOcclusionCullPass (Compute)   ← NOUVEAU, pré-cluster
    IN : LightBuffer, HiZPyramid N
    OUT: VisibleLightBuffer (indices), VisibleLightCountBuffer
    ↓
ClusterBuildPass (Compute)         ← inchangé
LightCullPass (Compute)            ← lit VisibleLightBuffer au lieu de LightBuffer
```

Le `LightCullPass` n'itère plus tous les lights, mais uniquement `visible_light_ids[]`. Sur une scène avec beaucoup de lumières ponctuelles derrière des murs, le gain peut être significatif.

> **Limites** : ce test est conservateur — une lumière dont la sphère touche un mur mais éclaire des surfaces devant lui ne sera pas cullée (correct). Il ne remplace pas le per-cluster light assignment, il le précède.

### 4d. Culling des Transparents

La frame graph sequence du culling transparent :

```rust
// Dans TransparentCullGroup::declare()
self.hiz_pyramid = builder.resolve_image("HiZPyramid");
builder.read_image(self.hiz_pyramid, ResourceUsage::SHADER_READ);
// → même texture que celle produite par HiZBuildPass, barrière automatique
```

Algorithme : identique au `CullingPass` opaque, mais sans émettre de draw_indexed_indirect — les transparents sont triés et dessinés en forward. Le cull réduit la liste de candidats avant le tri CPU (ou GPU).

---

## Données persistantes dans `GpuBuffers`

### Extensions requises

```rust
pub struct GpuBuffers {
    // ... existant (Phase 1 gpu_driven_plan.md) ...

    // Hi-Z — géré comme ressource history par le frame graph (N copies selon buffering)
    pub hiz_pyramid: BackendImage,   // R32_SFLOAT, mip-chain
}
```

### Tailles

| Resource | Dimensions | Format | Taille estimée (1080p) |
|---|---|---|---|
| `HiZPyramid` | 2048×1024 mip-chain | R32_SFLOAT | ~11 Mo (toutes mips) |

---

## Ordonnancement dans `DefaultRenderGraph::declare()`

```
Frame N :

1. SyncGroup
   └── MeshRegistrySyncPass
   └── InstanceSyncPass
   └── MaterialSyncPass

2. CullGroup
   └── ClearDrawCountPass     ← vkCmdFillBuffer(DrawCountBuffer, 0)
   └── CullingPass (Compute)  ← lit HiZPyramid N-1, écrit DrawCallBuffer
       IN : InstanceBuffer, MeshDescriptorBuffer, HiZPyramid (history)
       OUT: DrawCallBuffer, DrawCountBuffer, VisibleInstanceBuffer

3. GBufferPass (Graphics)
   └── GBufferFillPass        ← draw_indirect_count, early_fragment_tests
       IN : DrawCallBuffer (INDIRECT_READ), VisibleInstanceBuffer, etc.
       OUT: DepthBuffer N, GBuffer_*

4. HiZBuildPass (Compute)    ← SPD sur DepthBuffer N
   IN : DepthBuffer N (SHADER_READ)
   OUT: HiZPyramid N (tous mips, SHADER_WRITE)
   → Publie "HiZPyramid" dans le symbol table pour les passes suivantes

5. LightOcclusionCullPass (Compute)  ← pré-cluster, filtre lumières occultées
   IN : LightBuffer, HiZPyramid N
   OUT: VisibleLightBuffer, VisibleLightCountBuffer

6. ClusterGroup
   └── ClusterBuildPass       ← inchangé
   └── LightCullPass          ← lit VisibleLightBuffer (sous-ensemble non occludé)

7. GTAOPass (Compute)         ← lit HiZPyramid N

8. SSRPass (Compute)          ← lit HiZPyramid N (Hi-Z tracing)

9. DeferredResolvePass        ← lit GBuffer_* + AO + SSR

10. TransparentCullPass (Compute)  ← lit HiZPyramid N
11. ForwardTransparentPass        ← draw forward triés

12. PostProcessGroup
    └── FxaaPass
    └── TonemapPass

13. EguiPass
14. PresentPass

→ HiZPyramid N devient la copie history lue par le CullingPass de la frame N+1 (géré par le frame graph).
```

---

## Plan de fichiers

### Nouveaux fichiers

| Fichier | Contenu |
|---|---|
| `crates/i3_renderer/src/passes/culling.rs` | `CullingPass` (frustum + Hi-Z occlusion) |
| `crates/i3_renderer/src/passes/hiz_build.rs` | `HiZBuildPass` (SPD) |
| `crates/i3_renderer/src/passes/light_occlusion_cull.rs` | `LightOcclusionCullPass` |
| `crates/i3_renderer/assets/shaders/culling.slang` | Shader compute du CullingPass |
| `crates/i3_renderer/assets/shaders/hiz_build.slang` | Shader SPD compute |
| `crates/i3_renderer/assets/shaders/light_occlusion_cull.slang` | Shader pré-cluster light cull |

### Fichiers modifiés

| Fichier | Modification |
|---|---|
| `crates/i3_renderer/src/gpu_buffers.rs` | Ajouter `hiz_pyramid: BackendImage` |
| `crates/i3_renderer/src/render_graph.rs` | Intégrer `HiZBuildPass` + `LightOcclusionCullPass` |
| `crates/i3_renderer/assets/shaders/gbuffer.slang` | Ajouter `[earlydepthstencil]` |
| `crates/i3_gfx/src/graph/backend.rs` | Ajouter `fn fill_buffer(...)` dans `PassContext` |
| `crates/i3_vulkan_backend/src/commands.rs` | Implémenter `fill_buffer` → `vkCmdFillBuffer` |

---

## Pièges et points résolus

### 1. Convention de profondeur — Reverse-Z ✓

| Item | Valeur |
|---|---|
| Near plane depth | 1.0 |
| Far plane depth | 0.0 |
| Surface la plus proche | depth MAX |
| Opérateur SPD Hi-Z | **MAX** |
| Test d'occlusion instance | `aabb_max_depth_ndc < hiz_max` → occludé |
| Test d'occlusion light | `light_front_depth < hiz_max` → occludé |

### 2. History Hi-Z — Frame graph ✓

Le frame graph gère déjà les ressources temporelles sans nombre fixe de copies. `HiZPyramid` est déclaré comme ressource history — le graph résout automatiquement la copie N-1 selon le buffering en cours (double ou triple).

### 3. Résolution de HiZPyramid — Puissance de 2

> Décision : puissance de 2 supérieure (`next_pow2(w) × next_pow2(h)`).

Simplifie le SPD (pas de gestion des résolutions non-PoT), au prix de ~10% de mémoire sur-allouée par rapport à la résolution exacte.

### 4. Compatibilité D32_FLOAT → R32_SFLOAT

`vkCmdCopyImage` entre `D32_FLOAT` et `R32_SFLOAT` est **illégale** en Vulkan. Le mip0 du `HiZBuildPass` est implémenté comme un **blit compute** : lecture du depth via `sampler2D` (binding `SAMPLED`), écriture en `imageStore` dans `R32_SFLOAT`. Les réductions SPD suivantes travaillent entièrement dans `R32_SFLOAT`.

### 5. Alpha-test / earlydepthstencil

Le GBuffer actuel utilise un `discard` pour l'alpha-test. En conséquence, `[earlydepthstencil]` **ne peut pas** être activé sur ce shader (le GPU lirait le depth avant que le discard ne soit évalué). Le GBuffer sera rendu sans early-Z pour les matériaux alpha-tested — acceptable à court terme.

> Z-prepass dédié reporté à plus tard.

---

## Métriques de performance attendues

| Étape | GPU budget cible (1080p, mid-range) |
|---|---|
| `CullingPass` (frustum + Hi-Z) | ~0.2–0.5 ms |
| `GBufferPass` (après cull) | −30 à −70 % overdraw → gain dépendant de la scène |
| `HiZBuildPass` (SPD) | ~0.1–0.2 ms |
| GTAO avec Hi-Z tracing | ~1.0–1.5 ms (remplacement de ~2–3 ms sans Hi-Z) |
| SSR avec Hi-Z tracing | ~0.5–1.0 ms (vs ~2 ms linéaire) |

---

## Références

- [AMD FidelityFX SPD](https://gpuopen.com/fidelityfx-spd/) — algorithme de référence
- [Hi-Z Screen Space Cone Traced AO — Laine & Karras](https://research.nvidia.com/publication/2012-06_two-level-cone-step-mapping)
- [Hierarchical-Z map based occlusion culling — Johansson](https://www.rastergrid.com/blog/2010/10/hierarchical-z-map-based-occlusion-culling/)
- [Practical Hi-Z occlusion culling — GPU Gems](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-6-hardware-occlusion-queries-made-useful)
- `doc/gpu_driven_plan.md` — Phase 1b Niveau 2 : OcclusionCullGroup
