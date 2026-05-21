# Revue Technique — Moteur i3

> Auteur : Claude Sonnet 4.6  
> Date : 2026-05-21  
> Base de code : branche `main`, commit `2f7529b`

---

## Résumé exécutif

Le moteur i3 est un projet de rendu Vulkan 1.3 en Rust ambitieux, en développement actif. Il vise explicitement le segment "haut de gamme open source" que Godot ne couvre pas. L'architecture globale est saine et cohérente avec l'état de l'art (Frostbite Frame Graph, GPU-driven rendering, deferred clustered shading).

**Niveau de maturité :** Prototype avancé / Alpha. Le cœur du frame graph fonctionne. Le renderer deferred complet est opérationnel (GBuffer, GTAO, SSSR, Bloom, tone-mapping, clustering, sky). Le pipeline brickmap SDF clipmap (GPU-driven bake sur 10 niveaux) est récemment implémenté et fonctionnel. Il manque encore la GPU culling pass, le ZPrePass, le forward transparent et l'aliasing mémoire.

**Points forts :** Architecture frame graph propre, séparation backend abstraite robuste, documentation design-first de qualité, pipeline brickmap GPU innovant, shaders Slang bien structurés.

**Risques principaux :** ~~Unsafe dans `compiled.rs` (SyncPtr)~~ (corrigé), ~~fuite domaine Vulkan `SYNC-02`~~ (corrigé), ~~barrières AS manquantes `SYNC-06`~~ (corrigé), panics non contrôlées dans les chemins critiques du frame graph, aliasing mémoire manquant, `frame_graph_design.md` significativement obsolète.

---

## Architecture

### Forces architecturales

**Séparation HRI (Hardware Rendering Interface) exemplaire.** La frontière `i3_gfx` ↔ `i3_vulkan_backend` est respectée : aucun type `ash` ne traverse vers `i3_gfx`. Les translations abstraites (`AbstractTransition` → `vk::ImageMemoryBarrier2`) sont confinées dans le backend. Cette propreté est notable et maintenue avec discipline.

**Frame graph à trois phases bien implémenté.** Le cycle `declare → compile → execute` est fonctionnel. La topologie s'aplatit via `flatten_recursive`, le DAG se construit par analyse des dépendances (RAW, WAW, WAR), le tri topologique (Kahn) est correct. Les nœuds parallèles sont détectés par niveau et exécutés via `rayon`.

**Scoped Symbol Table (blackboard).** L'injection de dépendances CPU/GPU via `publish`/`consume`/`resolve_buffer`/`import_buffer` est cohérente. La règle `import_buffer` (master) vs `resolve_buffer` (consommateur) est effectivement respectée dans le code. `FrameBlackboard` pour les données per-frame est un ajout élégant.

**Pipeline brickmap.** L'approche GPU-driven bake (brickmap_bake.slang, dispatch indirect, LDS 729 voxels par workgroup) est architecturalement solide. La séparation CPU state (`BrickmapClipmapState`) / GPU buffers (`ClipmapGpuBuffers`) / passes de rendu est propre. Le système de 10 niveaux de clipmap (3.2m à 2km de couverture) avec toroidal addressing est bien conçu.

**Conventions cohérentes.** Reverse-Z, column-major, right-handed, CCW compensé par FrontFace=CW dans Vulkan — tout est documenté dans `engine_conventions.md` et respecté dans le code (e.g. construction de la projection, culling HiZ en MAX).

**Rust 2024, warnings = deny.** L'interdiction des warnings à la racine du workspace est une bonne pratique de discipline de code. L'utilisation de `thiserror` pour les erreurs structurées est correcte.

### Incohérences architecturales

Le `doc/frame_graph_design.md` contient des APIs qui ne correspondent plus au code (voir section Documentation). Cette divergence crée une confusion pour quiconque lit la doc avant le code.

~~**SYNC-02** : `queue_family: u32` dans `ResourceState` du module `i3_gfx` — fuite du domaine Vulkan dans la couche abstraite.~~ **Corrigé.** `queue_family: u32` remplacé par `queue: QueueType` dans la struct abstraite (`i3_gfx/graph/sync.rs`). Le seeding traduit désormais `img.last_queue_family` (u32 Vulkan réel) → `QueueType` via `get_queue_type_from_family_info` ; le commit fait le chemin inverse via `get_queue_family`. Le bug de cross-contamination (planner écrivait 0/1/2 dans `last_queue_family` qui était censé contenir un vrai VkQueueFamilyIndex) est éliminé.

---

## État des documentations

| Fichier | État | Remarque |
|---|---|---|
| `engine_hld.md` | **Partiel** | Structure workspace correcte mais liste `i3_dx12_backend` (inexistant), absents `i3_brickmap`, `i3_voxel`, `i3_math`, `i3_egui` |
| `frame_graph_design.md` | **Obsolète (partiel)** | Nombreuses divergences connues (voir `workplan.md §5`). APIs `PassBuilder`, `PassContext`, domaine, `graph.setup()` ne correspondent plus au code. Sections timing/memory restent valides conceptuellement |
| `renderer_design.md` | **Obsolète (partiel)** | Structure de passes décrite (~correct), mais `SceneProvider` trait non implémenté tel quel, skinning/BLAS absent. Utile comme vision |
| `engine_conventions.md` | **À jour** | Conventions coord, reverse-Z, roughness/alpha PBR — cohérentes avec le code |
| `baker_design.md` | **À jour** | Détaillé et correspond au code existant. Formats binaires i3mesh/i3scene/i3skeleton/i3animation corrects |
| `sdf.md` | **Partiellement dépassé** | Plan d'implémentation correctement suivi, mais quelques détails diffèrent (ex: 8 niveaux doc vs 10 niveaux code) |
| `culling.md` | **À jour (design)** | Architecture deux phases décrite, non encore implémentée. Cohérente avec l'existant |
| `ao.md` | **À jour** | GTAO implémenté et correspond à la description |
| `bloom.md` | **À jour** | Bloom implémenté et correspond |
| `ssr.md` | **Obsolète** | Remplacé par SSSR (voir `sssr.md`). Le code `passes/ssr.rs` n'existe plus (remplacé par `sssr.rs`) |
| `sssr.md` | **À jour** | Correspond à l'implémentation en cours de `passes/sssr.rs` |
| `spd.md` | **À jour (design)** | SPD documenté, non encore implémenté (`HiZBuildPass` multi-dispatch toujours en place) |
| `gpu_driven_plan.md` | **À jour (design)** | GPU culling pipeline non implémenté — correspond à ce qui manque |
| `graph_optimizer.md` | **À jour (design)** | Spécification technique de l'oracle de sync et du scheduler HEFT (non implémenté) |
| `workplan.md` | **À jour** | Le document de tracking le plus précis. Bien maintenu, statuts DONE corrects |
| `baked_noise.md` | Non lu (hors périmètre) | — |

---

## Revue par sous-système

### Frame Graph (`crates/i3_gfx/src/graph/`)

**Forces :**

- `compiler.rs` est clair et bien délimité (~560 LOC). La refactorisation GFX-03 a bien fonctionné.
- `sync_planner.rs` abstrait : les fonctions `get_image_state` et `get_buffer_state` couvrent correctement les cas d'usage (COLOR_ATTACHMENT, DEPTH_STENCIL, SHADER_READ/WRITE, TRANSFER, INDIRECT, ACCEL_STRUCT). La logique `needs_barrier` est conservative mais correcte (tout write déclenche une barrière, tout changement de layout aussi).
- L'exportation Mermaid `to_mermaid()` est un outil de debug utile.
- Le système de scratchpad (`scratch_images`, `scratch_buffers`) dans `SyncPlanner` pour éviter les allocations par frame est une bonne optimisation.

**Points d'attention :**

- ~~**SYNC-04 : `image_seed`/`buffer_seed` sont `pub`** dans `SyncPlanner`.~~ **Corrigé.** Champs passés privés ; API `seed_image(id, state)` / `seed_buffer(id, state)` / `clear_seeds()` exposée. `backend.rs` mis à jour en conséquence.
- ~~**GFX-04 : `consume` panique** au lieu de retourner `Result`.~~ **Partiellement corrigé.** `resolve_image()` et `resolve_buffer()` n'utilisent plus `consume` paniquant : elles appellent `try_consume` et retournent `ImageHandle::INVALID` / `BufferHandle::INVALID` avec `tracing::error!` sur symbole manquant. Le moteur continue à tourner. `consume()` générique reste paniquant (erreur de programmation intentionnelle, pour la consommation de services au init-time).
- **GFX-06 : aliasing mémoire absent.** Toutes les ressources transientes sont allouées indépendamment à chaque frame. La pression mémoire GPU n'est pas optimisée.
- **GFX-08 : dead node elimination absente.** Tous les nœuds déclarés sont exécutés même si leur output n'est pas consommé. `is_output` est en place mais non utilisé pour le culling.
- **Détection de cycle :** `topological_sort_levels` détecte un cycle via `order.len() != n` et log une erreur, mais continue à exécuter le graphe partiellement trié. En présence de cycle, le comportement est indéfini (pas de panic, pas de Result, juste un log).
- **SyncPtr unsafe dans `compiled.rs` :** Le `SyncPtr<T>` qui implémente `Send + Sync` sur un raw pointer est utilisé pour permettre l'accès parallèle aux `NodeStorage` depuis rayon. La sûreté repose sur l'invariant que chaque pass_index pointe vers un nœud distinct — ce qui est vrai par construction (le HashMap node_id → NodeStorage est unique), mais n'est pas vérifiable statiquement. Ce code a survécu sans issue visible, mais il s'agit d'un `unsafe` non trivial.
- **`add_pass` transmute lifetime :** `pass.rs:242` utilise `std::mem::transmute` pour caster un `&mut dyn RenderPass` en `'static`. Le commentaire `// Safety: The pass must outlive...` est correct mais non vérifiable. Une restructuration avec `&'graph mut dyn RenderPass` serait préférable à terme.

---

### Vulkan Backend (`crates/i3_vulkan_backend/src/`)

**Forces :**

- Timeline semaphores multi-queue correctement implémentés (`submission.rs`). Les `begin_frame` per-queue waitent les valeurs correctes avant de reset les pools.
- Synchronization2 (`VK_KHR_synchronization2`) utilisé correctement via `vk::ImageMemoryBarrier2` et `vk::BufferMemoryBarrier2`.
- `get_queue_family` dans `sync.rs` utilise un fallback `unwrap_or(graphics_family)` qui est safe (GFX-MQ-02 résolu).
- `sanitize_stages` émet des warnings tracés en cas de fallback (GFX-MQ-03 résolu).
- `VK_KHR_dynamic_rendering` utilisé, supprimant VkRenderPass/VkFramebuffer. Correct.

**Points d'attention :**

- ~~**TODO acceleration structures dans `sync_planner.rs` :** `// TODO: Handle AS transitions (memory barriers, no layout)`.~~ **Corrigé** (2026-05-21). Le planner abstrait suit désormais les lectures/écritures d'AS dans `accel_struct_reads/writes` sur `FlatPass`/`NodeStorage`. `import_acceleration_structure` enregistre une intention d'écriture + dépendance data ; `read_acceleration_structure` enregistre la lecture. La dépendance topologique BlasUpdate→TlasRebuild→shader est garantie par le DAG. Le `translate_plan` émet un `Barrier::Memory(VkMemoryBarrier2)` avec `srcStage=AS_BUILD, srcAccess=AS_WRITE, dstStage=AS_BUILD|RAY_TRACING_SHADER, dstAccess=AS_READ` pour couvrir le hazard TLAS-write→shader-read. `record_barriers` passe le memory barrier via `VkDependencyInfo.pMemoryBarriers`.
- **VK-03 : audit format conversion** non fait. `convert.rs` liste les conversions Format → VkFormat. Avec les ajouts récents (texture compressées, formats atlas brickmap), un audit est nécessaire.
- **GFX-12 : `VkBufferView` absent.** `UniformTexelBuffer` et `StorageTexelBuffer` ne sont pas supportés dans le backend. Cependant, **aucune passe du renderer n'utilise ces binding types** (audit 2026-05-21) — toutes les passes utilisent `storage_buffer` / `sampled_image` / `storage_image`. Dette purement hypothétique, non bloquante.
- ~~**GFX-13 : sous-ressources (mip/layer views) absentes.**~~ **Non-issue** (audit 2026-05-21). Les passes `HdrMipGen`, `HiZBuild`, `BloomDown/Up/Composite` gèrent les mips via `DescriptorWrite::storage_image_mip` / `sampled_image_mip` directement dans `execute()` — aucun contournement, aucune fuite d'abstraction. Le mécanisme existant couvre les besoins réels.
- **GFX-14 : readback GPU→CPU absent.** `OutputKind::Readback` dans `types.rs` est déclaré mais non implémenté. Bloque le bake IBL GPU-side, les debug dumps et les captures de frame.

---

### Renderer (`crates/i3_renderer/src/`)

**Forces :**

- Pipeline complet et opérationnel : GBuffer (albedo, normal oct-encodée, roughmetal, emissive, depth), GTAO, SSSR (stochastique), Bloom (Jimenez dual-pass), HiZ, clustering (tile 64px, 24 slices log), tone-mapping, FXAA, sky, debug viz.
- `CommonData` UBO centralisé correctement synchronisé une fois par frame — évite la duplication de matrices dans chaque pass.
- Les passes utilisent correctement `import_buffer` (master) et `resolve_buffer` (consommateur). La règle frame graph est respectée.
- `GBufferVertex` avec `tangent: [f32; 4]` — le normal mapping est opérationnel (RN-02 résolu).
- Architecture des groupes (GtaoGroup, RtaoGroup) propre.

**Points d'attention :**

- **RN-04 / RN-05 : GPU culling et ZPrePass absents.** Le pipeline actuel fait des draw directs CPU-driven. Sur des scènes denses, c'est un goulot d'étranglement potentiel significatif. `culling.md` documente l'architecture cible mais elle n'est pas implémentée.
- **RN-06 : forward transparent absent.** Aucun support des matériaux transparents/translucides.
- ~~**RN-11 : `unsafe ptr::copy_nonoverlapping` dans `sync.rs`** sans vérification de bornes.~~ **Corrigé** (2026-05-21). Deux guards ajoutés dans le loop de remplissage de `flat_descriptors` : `idx >= MAX_MESHES` → `tracing::warn!` + skip ; `idx >= flat_descriptors.len()` → `tracing::warn!` + skip. L'OOB silencieux est éliminé.
- ~~**RN-09 : `LightData` padding**.~~ **Corrigé** (2026-05-21). Le pipeline GPU utilise `GpuLightData` (`render_graph.rs`) comme intermédiaire — le layout GPU était déjà correct. Côté API : `LightType` a reçu `#[repr(u32)]` (discriminant garanti = 4 octets) et les champs de `LightData` ont été réordonnés pour correspondre à la définition HLSL (`position, radius, color, intensity, direction, light_type`).
- **GFX-15 : CommonData UBO** implémenté mais chaque pass qui lit les matrices doit encore consommer `CommonBuffer` explicitement. Ce n'est pas un bug mais crée de la verbosité.
- ~~**EG-I02 : scissoring absent dans egui.**~~ **Déjà implémenté** : `EguiPass::execute()` appelle `ctx.set_scissor()` par primitive depuis `clipped_primitive.clip_rect`, avec clamping à la taille de la fenêtre.
- **EG-I03 : VB/IB egui réalloués chaque frame.** Performance sous-optimale pour les UIs actives.

---

### Brickmap Pipeline (`crates/i3_brickmap/src/`, `examples/sdf/src/`)

**Forces :**

- Architecture CPU/GPU bien séparée. `BrickmapClipmapState` gère le state CPU (origins, giant_jump, invalidation spheres) ; `ClipmapGpuBuffers` gère les buffers GPU.
- Le baker Rayon (`bake_all`) parallélise correctement sur les bricks CPU. Le skip `center_sdf.abs() > half_diag * 2.0` est une bonne heuristique d'élimination rapide.
- La marge 9³ au lieu de 8³ (BRICK_SIZE+1) pour le trilinear sampling cross-brick est correcte et bien documentée dans le code.
- Le système de 10 niveaux (0.0125m à 8m/voxel) avec toroidal addressing est flexible.
- Les shaders brickmap sont propres : `brickmap_common.slang` centralise le lookup de brick et l'accès aux atlas.

**Points d'attention :**

- **Divergence doc/code sur le nombre de niveaux :** `doc/sdf.md` décrit 8 niveaux, le code (`clipmap.rs`) en implémente 10. C'est une évolution non documentée.
- **`BrickmapData` (chemin CPU) est du code legacy.** Le struct `BrickmapData` avec `page_table: Vec<u16>` / `sdf_atlas: Vec<u16>` représente l'ancien pipeline CPU-bake. Il n'est plus uploadé vers le GPU (le buffer `page_table_buf` est `GpuOnly`, non mappable). `BrickmapData` n'est utilisé que dans `brickmap_validate.rs` pour un test de timing CPU — le code est de facto mort côté pipeline GPU et devrait être nettoyé ou clairement marqué comme `#[cfg(debug_assertions)]` uniquement.
- **`gpu_scene.rs::pack_bvh`** : la conversion du BVH CPU vers le format GPU (`GpuBvhNode`) n'a pas de test. Si le layout ne correspond pas exactement au shader, le bake GPU produit des résultats incorrects silencieusement.
- **MAX_BRICKS_PER_LEVEL = 8192, GRID_VOL = 32768.** Il y a 4x plus de cellules que de bricks maximum par niveau. La gestion de l'overflow (atlas plein) : `alloc_slot` retourne `None` silencieusement si `brick_count >= max`. Dans ce cas, les bricks ne sont pas bakées — pas de message d'erreur visible à l'utilisateur.
- **`free_list` pour les bricks évincées** est en place mais dépend du système d'invalidation GPU (`brickmap_dirty.slang`) pour marquer les bricks libérables. L'intégration doit être auditée pour éviter les fuites d'atlas.
- **Shader `brickmap_bake.slang`** : la phase LDS est correcte (groupshared uint lds_sdf[729]) mais ne comporte pas de `GroupMemoryBarrierWithGroupSync()` entre la phase d'écriture LDS et la phase de lecture pour le store final. Si le store depuis LDS vers `sdf_atlas` se fait thread par thread sans sync, ce n'est pas un problème (chaque thread écrit son propre slot). À confirmer.

---

### Shaders (`examples/sdf/assets/shaders/`, `crates/i3_renderer/assets/shaders/`)

**Forces :**

- Utilisation correcte de Slang avec les extensions Vulkan (`[[vk::binding]]`, `[[vk::push_constant]]`).
- `brickmap_common.slang` factorise bien le lookup brick/atlas — réutilisé dans les passes de debug et de rendu.
- `brickmap_clipmap.slang` : la marche DDA hiérarchique (coarse brick par brick puis fine intra-brick) avec `safeRcp` pour éviter les divisions par zéro est correcte.
- `brickmap_bake.slang` : la normalisation SDF par `half_diag` avant stockage u8 est correcte.
- Les shaders renderer (`gtao_main.slang`, `sssr_sample.slang`) suivent les conventions reverse-Z et le Y-flip NDC documentées.

**Points d'attention :**

- **`PAGE_EMPTY` : pas de mismatch dans le pipeline GPU.** `brickmap_clipmap.slang` déclare `PAGE_EMPTY = 0xFFFFFFFFu` et lit la page table comme `StructuredBuffer<uint>` (u32). Le buffer GPU (`page_table_buf`) est `GpuOnly`, initialisé par `BrickmapInitPass` via `clear_buffer(..., 0xFFFF_FFFFu32)`, et écrit/lu exclusivement en u32 par les passes compute. L'ancien `BrickmapData.page_table: Vec<u16>` (valeur vide = `u16::MAX`) appartient au chemin CPU-only (`brickmap_validate.rs`) et n'est jamais uploadé vers ce buffer. Le pipeline GPU est cohérent.
- ~~**`globallycoherent` absent dans `brickmap_bake.slang`**~~ **Non-issue** (2026-05-21). Chaque workgroup écrit à un `atlas_offset` distinct (une brick par workgroup) — pas de conflits intra-dispatch. La lecture de l'atlas par les passes GBuffer se fait lors d'une frame ultérieure ou d'une passe séparée couverte par des barrières Vulkan compute→graphics. `globallycoherent` n'est nécessaire que pour des accès RAW intra-dispatch sans barrière, ce qui n'est pas le cas ici.

---

### Baker (`crates/i3_baker/src/`)

**Forces :**

- Architecture Importer/Extractor bien conçue et conforme au design doc.
- Formats binaires (`i3mesh`, `i3scene`, `i3skeleton`, `i3animation`) bien définis avec `repr(C)` et padding explicite.
- Bake incrémental via mtime check.
- Sémantique texture automatisée (BC7_SRGB pour albedo, BC5 pour normales, etc.).

**Points d'attention :**

- **BK-01 : `PipelineNode` abstraction** dans le baker déclarée morte dans `workplan.md` — code à auditer/supprimer.
- **BK-05 : tangent recalculation absente** si Assimp ne fournit pas de tangentes. Le GBuffer pass attend un `tangent: [f32; 4]` — les meshes sans tangentes auront des normales incorrectes.
- Pas de test d'intégration baker end-to-end visible dans le workspace (le plan prévoyait `integration_test.rs`).

---

### i3_io

**Forces :**

- `AssetHandle::get()` et `wait_loaded()` retournent des `Arc<T>` (IO-01 résolu) — pas de lifetime problème.
- `bytemuck::pod_read_unaligned` pour les casts mémoire (IO-03 résolu) — pas d'UB.
- Architecture VFS propre : raw files + bundle backend via `mmap`.

**Points d'attention :**

- Pas de test unitaire visible sur le VFS.

---

## Bugs connus et risques

Liste priorisée par impact potentiel :

### Critique

1. **Détection de cycle sans propagation d'erreur.** Un cycle dans le DAG (déclarations incohérentes) produit un `tracing::error!` mais continue à exécuter un graphe partiellement ordonné. Sur un bug de déclaration, le comportement est indéfini. Devrait retourner une `GraphError`.

3. ~~**SyncPtr unsafe dans `compiled.rs`.**~~ **Corrigé.** `*mut T` → `*const T`, `&mut *ptr` → `&*ptr` dans tous les sites de déréférencement. L'invariant de sûreté (node_ids uniques, lecture seule, tree stable pendant execute) est maintenant documenté dans le commentaire du type.

### Élevé

4. ~~**Panics non contrôlées dans les chemins critiques.**~~ **Partiellement corrigé.** `resolve_buffer()` et `resolve_image()` retournent désormais un handle INVALID + `tracing::error!` au lieu de crasher. `consume()` générique reste paniquant (usage intentionnel à l'init). Risque résiduel : un handle INVALID passé à un descriptor set peut produire un rendu silencieusement incorrect plutôt qu'un crash — surveiller les logs `tracing::error` en cas de régression visuelle.

5. **Aliasing mémoire GPU absent (GFX-06).** Sans aliasing, les ressources transientes (GBuffer, AO temporaire, SSR buffers) consomment de la VRAM en parallèle alors qu'elles pourraient partager de la mémoire. Sur GPU avec 8 GB de VRAM, ce n'est pas bloquant aujourd'hui mais le sera avec des scènes plus riches.

~~6. **`unsafe ptr::copy_nonoverlapping` sans bounds check** dans `sync.rs::MeshRegistrySyncPass`.~~ **Corrigé** (2026-05-21) — voir RN-11 ci-dessus.

### Moyen

~~7. **`globallycoherent` possiblement manquant** dans `brickmap_bake.slang`.~~ **Non-issue** (2026-05-21) — voir section shaders.

~~8. **LightData padding non audité (RN-09).**~~ **Corrigé** (2026-05-21) — `LightType` reçoit `#[repr(u32)]`, champs `LightData` réordonnés. Le GPU lit via `GpuLightData` qui avait déjà le bon layout.

~~9. **Scissoring egui absent (EG-I02).**~~ **Déjà implémenté** — `set_scissor()` appelé par primitive dans `EguiPass::execute()`.

~~10. **Overflow silencieux de l'atlas brickmap.**~~ **Corrigé** (2026-05-21). `alloc_slot` émet un `tracing::warn!` once-per-process via `AtomicBool` à la première saturation de l'atlas.

### Faible

~~11. **SYNC-05 : champ `layout` dans `ResourceState` pour buffers.**~~ **Documenté** (2026-05-21). Commentaire ajouté dans `ResourceState` : "`layout` n'a de sens que pour les images ; toujours `Undefined` pour les buffers et les AS — ces ressources n'ont pas de concept de layout en Vulkan."

12. **EG-I03 : VB/IB egui réalloués chaque frame.** Performance.

---

## Dettes techniques

### Documentées dans workplan.md

- **GFX-06** : aliasing mémoire (AliasingPlan)
- **GFX-08** : dead node elimination
- **GFX-11** : `DescriptorSetWriter` fluent (partiellement fait, struct existe)
- **GFX-12** : VkBufferView (UniformTexelBuffer, StorageTexelBuffer) — **non utilisé dans le renderer, dette hypothétique**
- ~~**GFX-13** : sous-ressources via ImageViewDesc~~ **Non-issue** : `DescriptorWrite::storage_image_mip` / `sampled_image_mip` couvrent les besoins réels
- **GFX-14** : readback GPU→CPU
- **GFX-15** : CommonData UBO (implémenté, mais la verbosité reste)
- **RN-04** : GPU culling
- **RN-05** : ZPrePass
- **RN-06** : forward transparent
- ~~**SYNC-02** : `queue_family: u32` dans la couche abstraite~~ **Corrigé** (2026-05-21)
- **SYNC-03** : `load_ops` dans PassSyncData (couplage renderpass/sync)
- ~~**SYNC-04** : `image_seed`/`buffer_seed` pub → méthodes~~ **Corrigé** (2026-05-21)
- ~~**SYNC-05** : `layout` dans ResourceState pour buffers~~ **Documenté** (2026-05-21)
- ~~**SYNC-06** : barrières manquantes pour acceleration structures~~ **Corrigé** (2026-05-21)

### Non documentées, identifiées lors de cette revue

- **`BrickmapData` legacy** — `Vec<u16>` page_table et sdf_atlas du chemin CPU-bake ne sont plus utilisés dans le pipeline GPU ; à nettoyer ou isoler en `#[cfg(debug_assertions)]`.
- **Mise à jour de `doc/sdf.md`** : nombre de niveaux (8 doc vs 10 code), détails du format atlas (f16 vs u8-packed).
- ~~**`doc/ssr.md` obsolète**~~ **Supprimé** (2026-05-21).
- **`doc/engine_hld.md` manque** : `i3_brickmap`, `i3_voxel`, `i3_math`, `i3_egui` dans le schéma workspace.
- **`doc/frame_graph_design.md`** : nécessite une révision majeure pour correspondre au code actuel (APIs, mécanismes d'exécution, `FrameBlackboard`, `declare_image_output`, etc.).
- **Tests d'intégration baker manquants** (`integration_test.rs` planifié mais absent).
- **Tests VFS manquants.**
- ~~**`BK-01` : PipelineNode** dead code dans le baker.~~ **Déjà absent** : grep confirme qu'aucun `PipelineNode` n'existe dans `crates/i3_baker/src/` — probablement retiré lors d'un refactor antérieur.

---

## Recommandations

### Priorité 1 — Sécurité immédiate

~~**Remplacer les panics critiques par Result dans le frame graph.**~~ **Partiellement fait** : `resolve_image()` et `resolve_buffer()` retournent `INVALID` + error log au lieu de crasher. Reste : `consume()` générique (init-time, panic acceptable), et le cas du handle INVALID utilisé en downstream (descriptor set invalide) qui ne crash pas mais produit un rendu incorrect silencieux.

### Priorité 2 — Stabilisation architecture

**Mettre à jour `doc/frame_graph_design.md`.** Ce document est la première référence pour comprendre le frame graph. Sa divergence avec le code nuit à la maintenabilité. La mise à jour devrait couvrir : les APIs réelles de `PassBuilder`, le mécanisme `FrameBlackboard`, `declare_image_output`/`import_buffer`/`resolve_buffer`, la non-implémentation de l'aliasing mémoire.

**Implémenter le ZPrePass et la GPU culling (RN-04/RN-05).** L'architecture `culling.md` est prête. Sans ZPrePass, le fill rate des fragments est inutilement élevé sur des scènes denses. Sans GPU culling, `iter_objects()` CPU-side devient le bottleneck sur les grandes scènes. Ces deux features sont le prochain seuil de scalabilité.

### Priorité 3 — Qualité long terme

**Implémenter l'aliasing mémoire (GFX-06).** La conception est documentée dans `frame_graph_design.md` et `graph_optimizer.md`. Le `MemoryPool` avec lifetime tracking des ressources transientes réduirait significativement la pression VRAM.

**Ajouter les tests d'intégration baker et VFS.** Le baker produit des formats binaires qui sont le pont entre le tooling et le runtime. Un test end-to-end (source glTF → bake → load → vérification header) éviterait des régressions silencieuses lors des évolutions de format.

~~**Auditer et documenter l'unsafe dans `compiled.rs`.**~~ Fait : `SyncPtr` passe de `*mut` à `*const`, tous les sites utilisent `&*ptr` (lecture seule), invariants de sûreté documentés dans le commentaire du type.

~~**Corriger SYNC-02 : `queue_family: u32` dans la couche abstraite.**~~ Fait : `ResourceState.queue_family: u32` → `queue: QueueType` dans `i3_gfx`. Pont seeding/commit dans `backend.rs` traduit correctement les family indices Vulkan ↔ `QueueType`. Plus de risque de cross-contamination entre les index abstraits (0/1/2) et les vrais `VkQueueFamilyIndex`.

~~**Corriger SYNC-06 : barrières acceleration structures manquantes.**~~ Fait (2026-05-21) : `write_acceleration_structure` / `import_acceleration_structure` alimentent désormais `accel_struct_writes` + `data_writes` ; `read_acceleration_structure` alimentent `accel_struct_reads`. Le sync_planner abstrait maintient un `as_flow` HashMap et émet `ResourceKind::AccelStruct` transitions. Le `translate_plan` Vulkan génère `Barrier::Memory(VkMemoryBarrier2)` avec `srcStage=AS_BUILD | dstStage=AS_BUILD|RAY_TRACING_SHADER`. `record_barriers` passe ces barrières via `VkDependencyInfo.pMemoryBarriers`. Le hazard TLAS-write→ray-tracing-shader-read est désormais couvert. Audit des passes (2026-05-21) : `DeferredResolvePass` et `RtaoPass` corrigés de `SHADER_READ` → `ACCEL_STRUCT_READ` dans `read_acceleration_structure` pour les accès TLAS.

---

*Fin du rapport — i3 Engine Review 2026-05-21*
