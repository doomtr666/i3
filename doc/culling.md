# Culling Architecture — Two-Phase Temporal Visibility

> Design décidé le 2026-04-13 en remplacement de l'approche Hi-Z temporel single-pass.  
> Référence tâches : `doc/culling_tasks.md`.

---

## Problème avec l'ancienne approche

L'ancienne approche reprojetait le Hi-Z du frame N-1 (produit avec la caméra précédente) pour culler les objets du frame N. Deux problèmes fondamentaux :

1. **Mouvement de caméra** : le Hi-Z N-1 ne correspond pas à la vue courante. Après reprojection, les occludeurs sont décalés → faux culls ou objets qui clignotent.
2. **Complexité** : gestion de `prevViewProjection`, de la reprojection UV, du border sampler, du texel snapping — tout cela pour un résultat fragile.

---

## Nouvelle approche : visibilité temporelle deux phases

### Principe

On maintient un **history buffer de visibilité** : la liste des indices d'instances qui étaient visibles à la frame N-1. Cette liste est l'input du PreZ.

```
Frame N-1                          Frame N
─────────────────────────────      ──────────────────────────────────────────────────────
                                   1. PreZ (Graphics)
                                      → rend { frustum_visible ∩ visible_N1 }
                                      → DepthPreZ (vue courante, camera N)

                                   2. HiZ Build #1 (Compute)
                                      → HiZPreZ (depuis DepthPreZ)

                                   3. Occlusion Cull (Compute)
                                      → tous les objets frustum_visible
                                      → testés contre HiZPreZ
                                      → produit visible_set_N

                                   4. GBuffer (Graphics)
                                      → rend visible_set_N
                                      → early-Z depuis DepthPreZ (gratuit)
                                      → DepthGBuffer (complet)

                                   5. HiZ Build #2 (Compute)
                                      → HiZFinal (depuis DepthGBuffer)
                                      → stocké en history pour le PreZ N+1

                                   6. Save visible_set_N → history pour frame N+1
```

### Pourquoi ça marche

- Le Hi-Z utilisé pour le culling (HiZPreZ) est construit **dans la même frame**, avec la même caméra. Pas de reprojection, pas de décalage.
- Les occludeurs du PreZ sont exactement les objets qui étaient visibles N-1 **et** sont dans le frustum courant. C'est l'ensemble le plus fiable disponible sans rayon de visibility supplémentaire.
- Le GBuffer bénéficie du depth buffer du PreZ : les fragments cachés sont rejetés gratuitement par le depth test hardware (early-Z).
- La convergence est rapide : après quelques frames, `visible_set_N1` est stable pour les scènes statiques.

### Cas limite : premier frame / reset

Si `visible_set_N1` est vide (démarrage, téléportation) :
- Le PreZ ne rend rien → DepthPreZ vide → HiZPreZ = tout 0 (far).
- L'occlusion cull passe tout (test conservatif : 0 = far → objet toujours visible).
- Le GBuffer rend tout ce qui est dans le frustum.
- Après une frame, `visible_set_N` est peuplé correctement.

→ La frame 0 est conservative (pas de cull), les frames suivantes sont correctes.

### Cas limite : objet nouvellement visible

Un objet qui entre dans le frustum est absent de `visible_set_N1` → absent du PreZ → absent du Hi-Z. Son test d'occlusion retournera `visible` (aucun occludeur dans le Hi-Z à sa position). Il sera rendu dans le GBuffer. Correct.

### Cas limite : occludeur qui disparaît

Un objet qui occultait d'autres objets sort du frustum à la frame N. Le PreZ ne le rend plus. Le HiZPreZ ne contient plus sa contribution. Les objets qu'il cachait passent le test d'occlusion et sont ajoutés à `visible_set_N`. Ils apparaissent au plus 1 frame après la disparition de l'occludeur. Acceptable.

---

## Détail de chaque phase

### Phase 1 — PreZ (Graphics, Depth Only)

**Input :** `DrawCallBuffer_PreZ` (produit par le cull de visibilité temporelle)  
**Output :** `DepthPreZ` (D32_FLOAT, résolution écran complète)

Le cull PreZ est un simple compute :
```
pour chaque instance i dans InstanceBuffer :
    si frustum_cull(i) ET visible_N1[i] :
        émettre draw
```

Shader depth-only minimal, sans fragment shader (ou FS vide). Le rasterizer écrit uniquement le depth.

### Phase 2 — HiZ Build #1 (Compute)

**Input :** `DepthPreZ`  
**Output :** `HiZPreZ` (R32_SFLOAT, mip-chain)

Réduction MAX (reverse-Z). Même algorithme que le HiZ Build #2, appliqué à `DepthPreZ`.

> Cible future : SPD (Single Pass Downsampler) pour les deux builds.

### Phase 3 — Occlusion Cull (Compute)

**Input :** `InstanceBuffer`, `MeshDescriptorBuffer`, `HiZPreZ`  
**Output :** `DrawCallBuffer_GBuffer`, `DrawCountBuffer`, `VisibleInstanceBitset_N`

```
pour chaque instance i dans le frustum :
    si occlusionTest(i.worldAabb, HiZPreZ, viewProjection) :
        émettre draw
        marquer visible_bitset[i] = true
```

Identique au `draw_call_gen.slang` actuel, sauf :
- Utilise `viewProjection` (frame courante) au lieu de `prevViewProjection`.
- Le Hi-Z est celui du PreZ courant, pas l'historique.
- Pas de `hizMipCount == 0` guard nécessaire (le PreZ vide → Hi-Z tout 0 → test conservatif passe).

### Phase 4 — GBuffer (Graphics)

**Input :** `DrawCallBuffer_GBuffer`, `DepthPreZ` (comme depth attachment en read-only pour early-Z)  
**Output :** `DepthGBuffer`, GBuffer targets (Albedo, Normal, RoughMetal, Emissive)

Le depth attachment est configuré en **depth test + depth write** normaux. Le hardware rejette automatiquement les fragments derrière le PreZ grâce à l'early-Z.

> Option avancée : attacher `DepthPreZ` en lecture seule et écrire dans un `DepthGBuffer` séparé. Selon le hardware et les drivers, le PreZ peut être promu en depth-prepass natif. Commencer simplement : un seul depth buffer, PreZ et GBuffer partagent le même attachment.

### Phase 5 — HiZ Build #2 (Compute)

**Input :** `DepthGBuffer` (depth complet de la frame)  
**Output :** `HiZFinal` → stocké comme history pour frame N+1

Ce Hi-Z contient la contribution de **tous** les objets visibles, pas seulement ceux du PreZ. C'est l'occluder set le plus complet disponible pour le culling de la frame suivante.

> Note : `HiZFinal` n'est pas utilisé pour le culling de la frame courante — seulement pour N+1. Les passes screen-space (GTAO, SSR) de la frame courante utilisent `HiZFinal` ou `HiZPreZ` selon leur besoin de complétude vs disponibilité.

### Phase 6 — History Update

**Input :** `VisibleInstanceBitset_N`  
**Output :** `VisibleInstanceBitset_N` → copié/swappé vers history pour frame N+1

Le bitset est une image ou buffer de taille `ceil(instance_count / 32)` uint32. Stocké dans `TemporalRegistry` pour être lu par le PreZ cull de la frame suivante.

---

## Structures de données

### VisibleInstanceBitset

```rust
// Taille : ceil(max_instances / 32) * 4 bytes
// Ex : 65536 instances → 2048 bytes = 2 Ko
BufferDesc {
    size: (max_instances + 31) / 32 * 4,
    usage: BufferUsageFlags::STORAGE | BufferUsageFlags::TRANSFER_DST,
    memory_type: MemoryType::GpuOnly,
}
```

Lecture dans le shader PreZ Cull :
```slang
bool wasVisible(uint instanceIdx) {
    uint word = visibilityBitset[instanceIdx / 32];
    return (word >> (instanceIdx % 32)) & 1u != 0;
}
```

### DrawCallBuffer (deux instances)

- `DrawCallBuffer_PreZ` : produit par le PreZ Cull compute
- `DrawCallBuffer_GBuffer` : produit par le Occlusion Cull compute

Même struct `DrawIndirectCommand`, même layout. Deux buffers séparés pour éviter les dépendances entre passes.

### HiZPreZ et HiZFinal

Deux images distinctes (ou réutilisation de la même selon l'architecture du frame graph) :

```rust
ImageDesc {
    format: Format::R32_SFLOAT,
    width:  screen_width,
    height: screen_height,
    mip_levels: floor(log2(max(w, h))) + 1,
    usage: ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
}
```

---

## Séquence dans le frame graph

```
Frame N :

1. SyncGroup
   ├── MeshRegistrySyncPass
   ├── InstanceSyncPass
   └── MaterialSyncPass

2. PreZCullGroup (Compute)
   ├── ClearDrawCountPass_PreZ          ← reset DrawCountBuffer_PreZ à 0
   └── PreZCullPass                     ← frustum_cull ∩ visible_N1 → DrawCallBuffer_PreZ

3. PreZGroup (Graphics, Depth Only)
   └── PreZPass                         ← draw_indirect depuis DrawCallBuffer_PreZ
                                           → DepthPreZ

4. HiZBuild1 (Compute)
   └── HiZBuildPass                     ← SPD sur DepthPreZ → HiZPreZ

5. OcclusionCullGroup (Compute)
   ├── ClearDrawCountPass_GBuffer       ← reset DrawCountBuffer_GBuffer à 0
   ├── ClearVisibilityBitset            ← reset VisibleInstanceBitset_N
   └── OcclusionCullPass                ← frustum + occlusion vs HiZPreZ
                                           → DrawCallBuffer_GBuffer + VisibleInstanceBitset_N

6. GBufferGroup (Graphics)
   └── GBufferFillPass                  ← draw_indirect_count depuis DrawCallBuffer_GBuffer
                                           early-Z depuis DepthPreZ
                                           → DepthGBuffer, GBuffer_*

7. HiZBuild2 (Compute)
   └── HiZBuildPass                     ← SPD sur DepthGBuffer → HiZFinal (→ history N+1)

8. ScreenSpaceGroup (Compute)           ← lit HiZFinal ou HiZPreZ selon besoin
   ├── GTAOPass
   ├── SSRPass
   └── LightCullPass

9. DeferredResolvePass

10. ForwardTransparentGroup
    ├── TransparentCullPass             ← lit HiZFinal
    └── ForwardTransparentPass

11. PostProcessGroup
    ├── FxaaPass
    └── TonemapPass

12. EguiPass / PresentPass

→ HiZFinal → history pour PreZCull N+1
→ VisibleInstanceBitset_N → history pour PreZCull N+1
```

---

## Comparaison avec l'ancienne approche

| Critère | Ancienne (Hi-Z temporel) | Nouvelle (visibilité temporelle) |
|---|---|---|
| Référentiel du Hi-Z | Frame N-1, vue N-1 | Frame N, vue N (PreZ) |
| Reprojection | Oui (`prevViewProjection`) | Non |
| Stabilité sur mouvement | Fragile (blink) | Stable |
| Latence de convergence | 1 frame | 1 frame |
| Nombre de Hi-Z builds | 1 | 2 |
| Passes GPU supplémentaires | 0 | +PreZ, +Cull PreZ, +HiZ1 |
| Coût estimé overhead | — | ~0.3–0.5 ms (PreZ + HiZ1) |
| Bénéfice early-Z GBuffer | Non | Oui (gratuit) |

Le coût du PreZ est partiellement compensé par le gain early-Z dans le GBuffer (moins de fragments exécutés dans le FS).

---

## Implémentation — État actuel

Le shader `draw_call_gen.slang` existant correspond au **OcclusionCullPass (Phase 5)**. Il utilise encore `prevViewProjection` et `hizMipCount` pour la logique temporelle — cela sera remplacé par `viewProjection` + `HiZPreZ` dans la nouvelle architecture.

Les passes existantes :
- `HiZBuildPass` → réutilisable pour les deux builds (paramétré par input depth image).
- `draw_call_gen` → à bifurquer en `PreZCullPass` et `OcclusionCullPass`.

---

## Références

- [Temporal Visibility — Sebastian Aaltonen (GPU Gems style)](https://advances.realtimerendering.com/)
- [Two-pass occlusion culling — Ubisoft / IW Engine](https://www.gdcvault.com/play/1022699/Optimizing-the-Graphics-Pipeline-With)
- [AMD FidelityFX SPD](https://gpuopen.com/fidelityfx-spd/) — pour les HiZ builds
- `doc/culling_tasks.md` — tâches d'implémentation détaillées
