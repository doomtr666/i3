# Tâches d'implémentation : Two-Phase Temporal Visibility Culling

> Design de référence : `doc/culling.md`.  
> Ancré sur l'état du code au 2026-04-13.  
> Chaque tâche est indépendante, testable, et modifie au maximum 2–3 fichiers.
>
> **État stabilisé (2026-04-14)** : occlusion test désactivé dans `draw_call_gen.slang` (commenté). Seul le frustum cull est actif pendant la transition vers la nouvelle architecture.

---

## Conventions

- **Test de compilation** : `.\tools\i3-cargo.ps1 check` passe sans erreurs ni warnings.
- **Test runtime** : l'application démarre, affiche la scène, aucune erreur Vulkan Validation Layer.
- Les shaders Slang sont buildés par l'asset pipeline. La tâche crée le `.slang`, le build produit le `.i3b`.
- **État actuel** : `draw_call_gen.slang` + `HiZBuildPass` opérationnels. Frustum cull + HiZ occlusion single-pass fonctionnels mais instables (reprojection temporelle).

---

## Statut global

```
M0 — Infrastructure visibilité temporelle        [ ] en attente
M1 — PreZ Cull + PreZ Pass                       [ ] en attente
M2 — HiZ Build paramétrable                      [ ] en attente
M3 — Occlusion Cull (vue courante)               [ ] en attente
M4 — GBuffer avec early-Z PreZ                   [ ] en attente
M5 — HiZ Build #2 + history final                [ ] en attente
M6 — Intégration frame graph complète            [ ] en attente
M7 — Debug GPU-driven (AABB visibles)            [ ] en attente
M8 — Light cull HiZ (clusters + sphères)         [ ] en attente
```

---

## M0 — Infrastructure : VisibleInstanceBitset

### T0.1 — Créer le VisibleInstanceBitset dans `GpuBuffers`

**Fichier :** `crates/i3_renderer/src/gpu_buffers.rs`

Ajouter deux buffers temporels (current + history) pour stocker la visibilité frame N-1 :

```rust
pub struct GpuBuffers {
    // ... champs existants ...

    /// Bitset des instances visibles à la frame courante (1 bit = 1 instance).
    /// Taille : ceil(max_instances / 32) * 4 bytes.
    pub visible_bitset: BackendBuffer,

    /// Bitset des instances visibles à la frame N-1 (history).
    pub visible_bitset_history: BackendBuffer,
}
```

Dans `GpuBuffers::new()` :
```rust
let bitset_size = (MAX_INSTANCES + 31) / 32 * 4;
let visible_bitset = backend.create_transient_buffer(&BufferDesc {
    size: bitset_size as u64,
    usage: BufferUsageFlags::STORAGE | BufferUsageFlags::TRANSFER_DST,
    memory_type: MemoryType::GpuOnly,
    label: Some("VisibleInstanceBitset".into()),
});
let visible_bitset_history = backend.create_transient_buffer(&BufferDesc {
    size: bitset_size as u64,
    usage: BufferUsageFlags::STORAGE | BufferUsageFlags::TRANSFER_DST,
    memory_type: MemoryType::GpuOnly,
    label: Some("VisibleInstanceBitset_History".into()),
});
```

**Test :** compilation passe.

---

### T0.2 — Swap des bitsets en fin de frame

**Fichier :** `crates/i3_renderer/src/render_graph.rs`

À la fin de `DefaultRenderGraph::render()`, après `execute_compiled_graph`, swapper les deux buffers :

```rust
std::mem::swap(
    &mut self.gpu_buffers.visible_bitset,
    &mut self.gpu_buffers.visible_bitset_history,
);
```

La prochaine frame, `visible_bitset_history` contient les instances visibles de la frame précédente.

**Test :** compilation passe. Pas de changement visuel encore.

---

### T0.3 — Ajouter un second DrawCallBuffer pour le PreZ

**Fichier :** `crates/i3_renderer/src/gpu_buffers.rs`

```rust
pub struct GpuBuffers {
    // ... existant ...
    pub draw_call_buffer:      BackendBuffer,   // existant : pour le GBuffer
    pub draw_count_buffer:     BackendBuffer,   // existant
    pub draw_call_buffer_prez: BackendBuffer,   // NOUVEAU : pour le PreZ
    pub draw_count_buffer_prez: BackendBuffer,  // NOUVEAU
}
```

Même `BufferDesc` que les buffers existants. Initialiser dans `GpuBuffers::new()`.

**Test :** compilation passe.

---

## M1 — PreZ Cull Pass

### T1.1 — Shader `prez_cull.slang`

**Fichier (nouveau) :** `crates/i3_renderer/assets/shaders/prez_cull.slang`

Compute shader 1 thread/instance. Émet un draw si l'instance passe le frustum ET était visible à N-1.

```slang
#include "common.slangh"

struct DrawIndirectCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct PreZCullPushConstants {
    float4x4 viewProjection;
    uint instanceCount;
    uint maxDrawCalls;
    uint2 _pad;
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuMeshDescriptor>    meshDescriptors;
[[vk::binding(1, 0)]] StructuredBuffer<GpuInstanceData>      instances;
[[vk::binding(2, 0)]] RWStructuredBuffer<DrawIndirectCommand> drawCalls;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>               drawCount;
[[vk::binding(4, 0)]] StructuredBuffer<uint>                 visibilityBitset; // history N-1

[[vk::push_constant]] PreZCullPushConstants pc;

bool wasVisible(uint idx) {
    uint word = visibilityBitset[idx / 32];
    return (word >> (idx % 32u)) & 1u != 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    if (idx >= pc.instanceCount) return;

    // Frustum cull (réutilise la fonction de draw_call_gen.slang)
    GpuInstanceData instance = instances[idx];
    if (!frustumTest(instance.worldAabbMin, instance.worldAabbMax, pc.viewProjection))
        return;

    // Filtre temporel : seulement les instances visibles à N-1
    if (!wasVisible(idx))
        return;

    GpuMeshDescriptor mesh = meshDescriptors[instance.meshIdx];
    uint drawIdx;
    InterlockedAdd(drawCount[0], 1, drawIdx);
    if (drawIdx < pc.maxDrawCalls) {
        DrawIndirectCommand cmd;
        cmd.vertexCount   = mesh.indexCount;
        cmd.instanceCount = mesh.indexCount > 0 ? 1 : 0;
        cmd.firstVertex   = 0;
        cmd.firstInstance = idx;
        drawCalls[drawIdx] = cmd;
    }
}
```

> Note : `frustumTest` est dupliqué depuis `draw_call_gen.slang`. Extraire dans `common.slangh` lors d'un refactor ultérieur.

**Test :** compilation shader passe (`slangc`).

---

### T1.2 — Rust : `passes/prez_cull.rs`

**Fichier (nouveau) :** `crates/i3_renderer/src/passes/prez_cull.rs`

Structure calquée sur le `DrawCallGenPass` existant. Bindings identiques, sauf :
- binding 4 : `visibilityBitset` (history) au lieu de `hizPyramid`.
- Push constants : `PreZCullPushConstants` (sans `prevViewProjection`, sans hiz params).

Déclarer dans le frame graph :
- Read : `InstanceBuffer`, `MeshDescriptorBuffer`, `VisibleInstanceBitset_History`
- Write : `DrawCallBuffer_PreZ`, `DrawCountBuffer_PreZ`

**Test :** compilation passe.

---

### T1.3 — PreZ Graphics Pass

**Fichier (nouveau) :** `crates/i3_renderer/src/passes/prez_pass.rs`

Pass graphics depth-only. Shader minimal :
- Vertex shader : lit `GpuInstanceData` via `firstInstance`, transforme la position avec `worldTransform`.
- Fragment shader : vide (ou absent — depth only).
- Pipeline state : color attachments = aucun, depth attachment = `DepthPreZ`.

```rust
pub struct PreZPass {
    draw_call_buffer:  BufferHandle,
    draw_count_buffer: BufferHandle,
    depth_prez:        ImageHandle,
    pipeline:          Option<BackendPipeline>,
}
```

Déclarer dans le frame graph :
- Read : `DrawCallBuffer_PreZ`, `DrawCountBuffer_PreZ`
- Write (depth) : `DepthPreZ` (D32_FLOAT, même dimensions que DepthBuffer)

**Test :** runtime — le PreZ rend les objets visibles N-1 dans le frustum courant. Pas encore utilisé pour le culling.

---

## M2 — HiZ Build paramétrable

### T2.1 — Adapter `HiZBuildPass` pour accepter un input configurable

**Fichier :** `crates/i3_renderer/src/passes/hiz_build.rs`

Le `HiZBuildPass` existant lit toujours `DepthBuffer`. Le paramétrer pour accepter soit `DepthPreZ` soit `DepthBuffer` :

```rust
pub struct HiZBuildPass {
    input_depth_name:  String,   // "DepthPreZ" ou "DepthBuffer"
    output_hiz_name:   String,   // "HiZPreZ" ou "HiZFinal"
    // ... autres champs inchangés
}

impl HiZBuildPass {
    pub fn new_prez() -> Self {
        Self::with_names("DepthPreZ", "HiZPreZ")
    }
    pub fn new_final() -> Self {
        Self::with_names("DepthBuffer", "HiZFinal")
    }
}
```

Dans `declare()`, utiliser `self.input_depth_name` et `self.output_hiz_name` pour les resolves.

**Test :** deux instances de `HiZBuildPass` peuvent être ajoutées au frame graph avec des noms différents.

---

## M3 — Occlusion Cull (vue courante)

### T3.1 — Adapter `draw_call_gen.slang`

**Fichier :** `crates/i3_renderer/assets/shaders/draw_call_gen.slang`

Remplacer la logique temporelle par un test contre le `HiZPreZ` courant :

1. Renommer `prevViewProjection` → `viewProjection` dans les push constants (supprimer `prevViewProjection`).
2. Dans `occlusionTest`, remplacer le paramètre `float4x4 prevVp` par `float4x4 vp`.
3. Supprimer la guard `hizMipCount == 0` (le HiZPreZ est toujours valide — si PreZ vide, le HiZ est tout 0 = far, test conservatif passe).
4. Ajouter un output : écrire dans `VisibleInstanceBitset` :

```slang
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> visibleBitset;

// Après la décision "visible", avant d'émettre le draw :
uint word_idx = idx / 32;
uint bit_idx  = idx % 32;
InterlockedOr(visibleBitset[word_idx], 1u << bit_idx);
```

**Push constants mis à jour :**
```slang
struct DrawCallGenPushConstants {
    float4x4 viewProjection;   // vue courante (plus de prevViewProjection)
    float2   screenSize;
    uint     hizMipCount;
    uint     instanceCount;
    uint     maxDrawCalls;
    uint     _pad;
};
```

**Test :** compilation shader. Le culling se fait contre le HiZPreZ (même frame, même caméra).

---

### T3.2 — Rust : mettre à jour `DrawCallGenPass`

**Fichier :** `crates/i3_renderer/src/passes/draw_call_gen.rs`

1. Supprimer le push de `prevViewProjection`.
2. Ajouter binding 5 : `visibleBitset` (write).
3. Déclarer en write dans le frame graph : `VisibleInstanceBitset` (reset en début de frame).
4. Lire `HiZPreZ` au lieu de l'image HiZ temporelle.

**Test :** runtime — le culling est stable (pas de reprojection).

---

## M4 — GBuffer avec early-Z PreZ

### T4.1 — Configurer le GBuffer pour utiliser `DepthPreZ`

**Fichier :** `crates/i3_renderer/src/passes/gbuffer.rs`

Le depth attachment du GBuffer peut réutiliser `DepthPreZ` comme depth buffer de départ. Les fragments déjà testés par le PreZ seront rejetés gratuitement.

Option simple : le GBuffer déclare `DepthBuffer` comme output, initialisé depuis `DepthPreZ` via une copie ou en l'utilisant directement :

```rust
// Dans declare() :
// Lire DepthPreZ comme input depth initial
// Écrire dans DepthBuffer (ou réutiliser DepthPreZ en R/W)
builder.read_image(self.depth_prez, ResourceUsage::DEPTH_READ);
builder.write_image(self.depth_buffer, ResourceUsage::DEPTH_WRITE);
```

Si le frame graph supporte le `LOAD_OP_LOAD` sur le depth attachment, charger `DepthPreZ` comme état initial. Sinon, effectuer une copie `vkCmdCopyImage` de `DepthPreZ` → `DepthBuffer` avant le GBuffer.

**Test :** runtime — le depth buffer final contient les profondeurs PreZ + GBuffer. Vérifier en RenderDoc qu'il n'y a pas de régression dans le G-Buffer.

---

## M5 — HiZ Build #2 + History

### T5.1 — Ajouter `HiZBuildPass::new_final()` dans le frame graph

**Fichier :** `crates/i3_renderer/src/render_graph.rs`

Instancier une seconde `HiZBuildPass` après le GBuffer :

```rust
let hiz_build_final = HiZBuildPass::new_final(); // lit DepthBuffer → écrit HiZFinal
```

`HiZFinal` est stocké via `TemporalRegistry` comme history pour la frame N+1 (le `PreZCullPass` de la frame suivante n'en a pas besoin directement, mais les passes screen-space et les transparents le lisent).

**Test :** runtime — `HiZFinal` contient la pyramide complète post-GBuffer.

---

### T5.2 — Screen-space passes utilisent `HiZFinal`

**Fichiers :** passes GTAO, SSR, LightCull, TransparentCull

Remplacer toute référence à l'ancien HiZ temporel par `HiZFinal`. Ces passes bénéficient du Hi-Z complet (tous objets visibles, pas seulement PreZ).

**Test :** GTAO/SSR visuellement identiques ou améliorés.

---

## M6 — Intégration frame graph complète

### T6.1 — Séquence dans `DefaultRenderGraph::declare()`

**Fichier :** `crates/i3_renderer/src/render_graph.rs`

Ordre final des passes :

```
1. SyncGroup
   ├── MeshRegistrySyncPass
   ├── InstanceSyncPass
   └── MaterialSyncPass

2. PreZCullGroup
   ├── ClearDrawCountPass (DrawCountBuffer_PreZ → 0)
   └── PreZCullPass (frustum ∩ visible_N1 → DrawCallBuffer_PreZ)

3. PreZGroup
   └── PreZPass (depth-only → DepthPreZ)

4. HiZBuild1Group
   └── HiZBuildPass::new_prez() (DepthPreZ → HiZPreZ)

5. OcclusionCullGroup
   ├── ClearDrawCountPass (DrawCountBuffer → 0)
   ├── ClearBitsetPass (VisibleInstanceBitset → 0)
   └── DrawCallGenPass (frustum + occlusion vs HiZPreZ → DrawCallBuffer + VisibleBitset)

6. GBufferGroup
   └── GBufferFillPass (DrawCallBuffer, DepthPreZ → DepthBuffer, GBuffer_*)

7. HiZBuild2Group
   └── HiZBuildPass::new_final() (DepthBuffer → HiZFinal → history)

8. ScreenSpaceGroup
   ├── GTAOPass (HiZFinal)
   ├── SSRPass (HiZFinal)
   └── LightCullPass

9. DeferredResolvePass

10. ForwardTransparentGroup
    ├── TransparentCullPass (HiZFinal)
    └── ForwardTransparentPass

11. PostProcessGroup + EguiPass + PresentPass
```

**Test :** runtime — scène complète rendue sans artefacts, pas d'erreur Vulkan Validation Layer, culling stable sans blink.

---

### T6.2 — Supprimer l'ancienne logique temporelle de `draw_call_gen.slang`

**Fichier :** `crates/i3_renderer/assets/shaders/draw_call_gen.slang`

- Supprimer `prevViewProjection` des push constants.
- Supprimer la guard `hizMipCount == 0` (remplacée par la sémantique PreZ vide).
- Nettoyer les commentaires obsolètes sur la reprojection.

**Fichier :** `crates/i3_renderer/src/passes/draw_call_gen.rs`

- Supprimer le champ `prev_view_projection` et son upload.
- Adapter `DrawCallGenPushConstants` en Rust.

**Test :** compilation passe. Pas de régression visuelle.

---

## M7 — Debug GPU-driven : AABB des instances visibles

### T7.1 — Compute : générer les lignes AABB depuis le DrawCallBuffer

**Fichier (nouveau) :** `crates/i3_renderer/assets/shaders/debug_cull_vis.slang`

1 thread par draw call émis. Lit `DrawCallBuffer[i].firstInstance` → index d'instance → `worldAabbMin/Max` → émet 12 segments de ligne dans le `DebugLineBuffer` du `DebugDrawPass`.

```slang
[[vk::binding(0, 0)]] StructuredBuffer<DrawIndirectCommand> drawCalls;
[[vk::binding(1, 0)]] StructuredBuffer<GpuInstanceData>     instances;
[[vk::binding(2, 0)]] RWStructuredBuffer<DebugLine>         debugLines;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>              debugLineCount;

struct PushConstants { uint drawCount; uint maxLines; float4 color; };
[[vk::push_constant]] PushConstants pc;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.drawCount) return;
    uint instIdx = drawCalls[tid.x].firstInstance;
    float3 lo = instances[instIdx].worldAabbMin;
    float3 hi = instances[instIdx].worldAabbMax;
    // Émettre 12 lignes de l'AABB (4 arêtes × 3 axes) dans debugLines[]
    // via InterlockedAdd sur debugLineCount
}
```

**Test :** activer en debug → les AABBs des objets retenus par le GPU s'affichent en overlay.

---

### T7.2 — Intégrer dans `DebugDrawPass`

**Fichier :** `crates/i3_renderer/src/passes/debug_draw.rs`

Ajouter une méthode `enable_cull_vis(enabled: bool)`. Quand activé, un compute `DebugCullVisPass` tourne après `DrawCallGenPass` et peuple le `DebugLineBuffer`. Le toggle s'expose dans l'UI egui (touche dédiée ou checkbox).

---

## M8 — Light Cull avec HiZ (future)

Deux niveaux de culling des lumières après `HiZBuildFinal` :

### T8.1 — Culling par cluster AABB

Avant l'affectation lumière/cluster : tester l'AABB du cluster contre `HiZFinal`. Un cluster entièrement derrière la géométrie n'a pas besoin d'être traité.

Test identique à l'occlusion cull d'instance (projeter AABB, tester maxZ vs HiZ mip correspondant).

### T8.2 — Culling par sphère de lumière

Pour chaque lumière ponctuelle/spot : projeter la sphère d'influence, tester le front-face depth contre `HiZFinal`. Si le front de la sphère est derrière l'occludeur à tous les coins UV → cull.

```slang
// Front de la sphère en reverse-Z = depth MAX
float sphereFrontZ = ndc_center.z + proj_radius; // approximation NDC
if (sphereFrontZ + kBias < hizSample) return; // lumière entièrement occultée
```

**Non bloquant** — implémentable une fois `HiZFinal` stable et les passes screen-space migrées.

---

## M9 — SPD (future, non bloquant)

### T9.1 — Implémenter Single Pass Downsampler

Remplacer les N dispatches itératifs de `HiZBuildPass` par un SPD en un seul dispatch.

Référence : [AMD FidelityFX SPD](https://gpuopen.com/fidelityfx-spd/).

Requis : `globallycoherent` image writes + atomic counter inter-workgroup. Non bloquant pour la correction — le rendu est correct avec les dispatches itératifs.

---

## Ordre de priorité suggéré

```
M0 → M1 (PreZ infra) → M2 (HiZ param) → M3 (OcclusionCull current VP)
→ M4 (GBuffer early-Z) → M5 (HiZ Final) → M6 (integration) → M7 (SPD)
```

M3 peut être testé en isolation (remplacer juste la VP dans draw_call_gen) avant que PreZ/HiZPreZ ne soient prêts, en utilisant temporairement le HiZ existant avec `viewProjection` au lieu de `prevViewProjection`. Cela seul élimine le problème de reprojection et peut être un point de validation intermédiaire utile.
