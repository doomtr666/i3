# SDF → Brickmap avec clipmap multi-niveaux

> **Note d'implémentation (2026-05-21)** — Ce document est le plan de conception original.
> Il a été largement suivi mais présente plusieurs divergences avec le code actuel :
> - **10 niveaux** implémentés (L0..L9), non 8.
> - **Format atlas : u8-packed** (4 octets/DWORD). Les samples f16 du chemin CPU legacy (`BrickmapData`) ne sont plus uploadés vers le GPU.
> - **Brick voxels : 9³ = 729** (BRICK\_SIZE=8 + 1 voxel de recouvrement pour le trilinear cross-brick), non 8³=512.
> - **BRICK\_DWORDS = 183** (ceil(729/4)), non 128.
> - **Grille par niveau : 32³ = 32 768 cellules** (CLIPMAP\_GRID=[32,32,32]). Couverture = 256 × voxel\_size par axe.
> - Le **bake GPU compute** (`brickmap_bake.slang`) est implémenté et dispatché en indirect. Les passes `dirty` (détection) et `cull+alloc` (allocation GPU) sont la prochaine étape (plan GPU-driven dans `gpu_driven_plan.md`).
> - La **DDA traversal** (`brickmap_clipmap.slang`) est implémentée et fonctionnelle.
> - La **page\_table GPU** est un buffer `u32` (`PAGE_EMPTY = 0xFFFFFFFF`), initialisé par `BrickmapInitPass`.

## Contexte

L'exemple SDF actuel évalue toutes les primitives par ray marching analytique (évaluateur RPN). Avec ~10 primitives c'est correct, mais le passage à des centaines de primitives + un terrain fBm détaillé rend cette approche trop coûteuse.

L'objectif est de pré-calculer le champ de distance sur une grille hiérarchique (brickmap clipmap) et de faire le ray marching par DDA sur cette grille plutôt que par évaluation analytique à chaque pas.

## Architecture cible

```
CPU : SdfScene + BVH des primitives
        │
        ▼  BrickmapBaker (par frame, budget N bricks)
        │
GPU : Page table 3D (u16 → brick index)   ← Clipmap L0..L7
      + Atlas 3D SDF (f16, 8×8×8 bricks)
      + Atlas 3D material ID (u8)
        │
GPU Shader : DDA traversal → trilinear sample
             Fallback analytique (RPN) au-delà du clipmap
```

---

## Étape 1 — BVH sur les primitives SDF (fondation CPU)

**Pourquoi :** Le baker doit, pour chaque brick, trouver quelles primitives l'influencent. Le scan linéaire O(n) actuel de `SdfScene::get_nodes()` ne scale pas. Un BVH réduit ça à O(log n) et bénéficie aussi au voxel octree existant.

**Fichier : `crates/i3_voxel/src/sdf.rs`**

```rust
struct BvhNode {
    aabb: AABB,
    left: u32,              // 0xFFFFFFFF = feuille
    right: u32,
    primitive_index: u32,   // index dans nodes[] si feuille
}

// Ajouté à SdfScene :
bvh_nodes: Vec<BvhNode>,

impl SdfScene {
    pub fn build_bvh(&mut self);
    pub fn get_nodes_bvh(&self, query: &AABB) -> impl Iterator<Item = &SdfNode>;
}
```

Build : **median split** sur l'axe le plus long de l'AABB courant.  
Remplace l'appel dans `octree.rs::OctreeNode::generate_mesh()`.

**Vérification :** rendu voxel identique avant/après, benchmark temps génération mesh avec 100+ primitives.

---

## Étape 2 — Brickmap baker CPU (1 niveau uniforme) — **implémenté**

**Crate `crates/i3_brickmap/`**

```
Brick = 9×9×9 = 729 samples (BRICK_SIZE=8 + 1 voxel de recouvrement trilinear)
Sample SDF GPU : u8, normalisé par half_diag, packé 4 voxels/DWORD (183 DWORDs/brick)
Sample mat GPU : u8, packé 4 voxels/DWORD
Page table GPU : u32 flat [z][y][x], 0xFFFFFFFF = brick vide (GpuOnly buffer)

// Chemin CPU legacy (BrickmapData, utilisé uniquement dans les tests) :
Sample SDF CPU : f16 bits
Page table CPU : Vec<u16>, 0xFFFF = vide
```

**Types clés :**

```rust
pub struct BrickmapBaker {
    pub brick_size:   u32,         // 8
    pub grid_dims:    [u32; 3],    // nombre de bricks par axe
    pub world_origin: [f32; 3],
    pub voxel_size:   f32,         // taille monde d'un voxel
}

pub struct BrickmapData {
    pub page_table:  Vec<u16>,     // grid_dims.x × y × z
    pub sdf_atlas:   Vec<u16>,     // brick_count × 512, f16 bits
    pub mat_atlas:   Vec<u8>,      // brick_count × 512
    pub brick_count: u32,
}

impl BrickmapBaker {
    pub fn bake_all(&self, sdf: &SdfScene) -> BrickmapData;
    pub fn bake_batch(&self, sdf: &SdfScene, state: &mut BakeState, budget: usize);
}
```

**Algorithme bake d'une brick `(bx, by, bz)` :**
1. Calculer l'AABB monde + marge (2 × voxel_size)
2. `sdf.get_nodes_bvh(aabb)` → primitives candidates
3. Évaluer SDF au **centre** de la brick ; si `|sdf| > brick_half_diagonal × 2` → skip
4. Évaluer les 8³ points, quantifier en f16 normalisé, écrire dans atlas
5. Mettre à jour page_table

**Dépendances `Cargo.toml` :**
```toml
i3_math  = { path = "../i3_math" }
i3_voxel = { path = "../i3_voxel" }
half     = "2"
rayon    = { workspace = true }
```

**Vérification :** dump atlas → slices PNG, isosurface d=0 visible pour sphère + terrain.

---

## Étape 3 — Rendu GPU par DDA traversal (1 niveau)

**Nouveau `examples/sdf/src/brickmap_pass.rs`**

**Buffers GPU uploadés depuis `BrickmapData` :**
```
page_table_buf : StorageBuffer u16[], grid_dims.x×y×z × 2 bytes
sdf_atlas_buf  : StorageBuffer u16[], brick_count × 512 × 2 bytes
mat_atlas_buf  : StorageBuffer u8[],  brick_count × 512 bytes
```

**Push constants :**
```slang
struct BrickmapPC {
    float3 world_origin;  float voxel_size;
    uint3  grid_dims;     uint  brick_count;
};
```

**Shader `brickmap_gbuffer.slang` — algorithme :**
```
1. DDA coarse : avance brick par brick (t_delta = brick_world_size / |rd.axis|)
2. Pour chaque brick :
   a. page_table[bx,by,bz] == 0xFFFF ? → skip, DDA step suivant
   b. DDA fine 8×8×8 à l'intérieur de la brick
      - Interpolation trilinéaire du SDF
      - Hit si sdf < 0.001
3. Normal : 6 taps sur l'atlas, différences finies (step = voxel_size)
4. Material : mat_atlas[brick_base + local_idx]
5. Hors grille : fallback évaluateur RPN analytique existant
```

**Vérification :** même visuel qu'analytique pour sphère + terrain, mesurer gain FPS.

---

## Étape 4 — Clipmap multi-niveaux (**10 niveaux — implémenté**)

Chaque brick = 8³ voxels + 1 voxel de recouvrement trilinear = 9³ = 729 samples. La couverture double à chaque niveau. 10 niveaux couvrent du détail millimétrique (1.25 cm) jusqu'à 2 km, avec fallback analytique au-delà.

Grille : **32×32×32 bricks/niveau**. Couverture = 32 × 8 × voxel\_size = 256 × voxel\_size par axe.

| Niveau | Voxel size | Brick size | Couverture (côté) | MAX\_BRICKS |
|--------|-----------|-----------|-------------------|-------------|
| L0     | 0.0125 m  | 0.10 m    | 3.2 m             | 8192        |
| L1     | 0.025 m   | 0.20 m    | 6.4 m             | 8192        |
| L2     | 0.05 m    | 0.40 m    | 12.8 m            | 8192        |
| L3     | 0.10 m    | 0.80 m    | 25.6 m            | 8192        |
| L4     | 0.20 m    | 1.60 m    | 51.2 m            | 8192        |
| L5     | 0.40 m    | 3.20 m    | 102 m             | 8192        |
| L6     | 0.80 m    | 6.40 m    | 205 m             | 8192        |
| L7     | 1.60 m    | 12.8 m    | 410 m             | 8192        |
| L8     | 4.00 m    | 32.0 m    | 1024 m            | 8192        |
| L9     | 8.00 m    | 64.0 m    | 2048 m            | 8192        |

**Mémoire GPU :**
- Page tables : 10 × 32³ × 4 bytes (u32) = **10 MB**
- Atlas SDF (u8-packed) : 10 × 8192 bricks × 183 DWORDs × 4 bytes = **60 MB**
- Atlas material (u8-packed) : identique = **60 MB**
- Total atlas : **≈ 130 MB** (constant, pas d'éviction LRU)

**`crates/i3_brickmap/src/clipmap.rs` (implémenté) :**
```rust
pub const NUM_LEVELS: usize = 10;
pub const CLIPMAP_GRID: [u32; 3] = [32, 32, 32];
pub const LEVEL_VOXEL_SIZES: [f32; NUM_LEVELS] =
    [0.0125, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 4.00, 8.00];
pub const MAX_BRICKS_PER_LEVEL: usize = 8192;
pub const GRID_VOL: usize = 32 * 32 * 32; // 32 768

pub struct BrickmapClipmapState {
    pub levels:               Vec<ClipmapLevel>,
    pub invalidation_spheres: Vec<InvalidationSphere>,
}
```

**Shader :** sélection de niveau par distance camera → brick, blend aux transitions de niveau.

---

## Étape 5 — Mise à jour dynamique + hybride analytique

**Dynamisme :**
- Primitive change → invalider bricks dont l'AABB intersecte l'ancienne **ou** nouvelle AABB de la primitive
- Re-enqueue dans `BakeState` avec priorité haute (`Dirty` state)
- Rendu utilise le dernier état valide pendant le rebake

**Hybride brickmap / analytique :**
- `t > coverage_L7` → bascule vers RPN evaluator analytique existant
- Sous-ensemble filtré par emprise visuelle pour limiter le coût analytique

**Matériaux :**
- `GpuMaterial[256]` : albedo + roughness + metallic, buffer partagé
- Bricks stockent un `u8` material ID → lookup au hit

---

## Étape Debug — Visualisation brickmap & clipmap

Réutilise `debug_draw_pass`, `RendererDebugGui`, `DebugChannel` existants.

### Canaux GPU (nouveaux `DebugChannel`)

| Canal | Description |
|-------|-------------|
| `BrickmapLevel` | Couleur par niveau actif au pixel (L0=bleu → L7=rouge) |
| `BrickmapSdfError` | Différence SDF brickmap vs analytique |
| `BrickmapAllocated` | Vert = allouée, noir = vide/fallback |
| `BrickmapFreshness` | Blanc = rebakée ce frame, rouge = ancienne |

### Overlays fil-de-fer (debug_draw_pass)

- Bricks allouées L0 : bleu clair
- Bricks en cours de rebake : jaune
- Bricks invalidées : rouge (1 frame)

### Statistiques GUI (extra closure)

```
Brickmap — L0: 12 341 / 16K  [=====......] 128 rebaked/frame
           ...
Atlas : 48 231 / 262 144 bricks (18%)  ~48 MB SDF
Brick budget : [slider 8..256]/frame
```

### Dump PNG (debug uniquement)

```rust
#[cfg(debug_assertions)]
pub fn dump_slice_png(&self, level: usize, y_world: f32, path: &str);
```

---

## Test de validation — Déformation dynamique

### Creuser (clic gauche)
- Raycaste contre la surface brickmap
- Ajoute une soustraction sphérique `r = 2.0 m` au point de contact
- Bricks dont l'AABB intersecte la sphère → `Dirty`, rebake prioritaire
- **Attendu :** cratère visible en < 5 frames

### Ajouter de la matière (clic droit)
- Même logique, union sphérique `r = 1.5 m`

### GUI
```
Tool radius : [slider 0.5..10.0 m]
[ ] Dig (clic gauche)    [ ] Fill (clic droit)
Dirty bricks : 1 247  →  rebaked this frame : 128
```

### Critères

| Mesure | Cible |
|--------|-------|
| Latence visuelle creuser r=2m | < 5 frames |
| Creuser r=10m | < 15 frames |
| Trou net, pas de gradient corrompu | ✓ |
| Frontière brickmap/analytique cohérente | ✓ |
| Déformation persistante après déplacement caméra | ✓ |

---

## Ordre d'implémentation

1. `doc/sdf.md` — ce document ✓
2. **Étape 1** : BVH dans `crates/i3_voxel/src/sdf.rs`
3. **Étape 2** : Crate `crates/i3_brickmap/`, baker CPU, dump PNG
4. **Étape Debug** : overlays + stats GUI (en parallèle des étapes suivantes)
5. **Étape 3** : `brickmap_pass.rs` + `brickmap_gbuffer.slang`, 1 niveau
6. **Étape 4** : Clipmap 8 niveaux + mise à jour incrémentale
7. **Étape 5** : Dynamisme + hybride analytique/brickmap

## Critères de validation globaux

- Visuellement identique au ray marcher analytique pour la scène de référence
- FPS ×2 minimum sur terrain fBm dense
- Caméra à 200 m/s sur 20 km², transitions de niveau invisibles
- Creuser r=2m → visible en < 5 frames
- Debug GUI : stats live + canaux `BrickmapLevel`, `BrickmapAllocated`, `BrickmapFreshness`
