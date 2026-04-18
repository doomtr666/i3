# Single Pass Downsampler (SPD)

> Référence : "FidelityFX Single Pass Downsampler" — AMD GPUOpen, 2020.  
> Paper : "Optimized Down-Sampling on the GPU" — Jasper St. Pierre, GDC 2020.  
> Code de référence : https://github.com/GPUOpen-Effects/FidelityFX-SPD

---

## 1. Problème

Le downsampler multi-dispatch actuel (`HiZBuildPass`) émet **N dispatches séquentiels** (un par mip) avec une barrière pipeline entre chaque. Pour une image 1920×1080 → 11 mips, c'est 10 round-trips GPU, chaque dispatch étant sous-occupé sur les petits mips.

Le même problème se pose pour la génération des mips de `SceneLit` (R16G16B16A16_SFLOAT) nécessaires à SSR.

---

## 2. Principe SPD

SPD génère **toute la chaîne de mips en un seul dispatch compute**, quelle que soit la taille source (jusqu'à 4096×4096, soit 12 mips).

### Deux phases dans un seul dispatch

```
Input (mip 0)
    │
    ├── Phase 1 : chaque threadgroup (16×16 = 256 threads) traite
    │             une tuile 64×64 de mip 0 et produit les mips 1→5
    │             en mémoire locale (LDS). Aucune synchronisation globale.
    │
    │   [atomic fetch_add sur un compteur global]
    │   Le dernier threadgroup à finir est détecté → il prend en charge
    │
    └── Phase 2 : un unique threadgroup ré-réduit les résultats de
                  tous les threadgroups (mip 5) pour produire mips 6→12
                  via LDS. Toujours dans le même dispatch.
```

### La clé : le compteur global

Un buffer de `N` u32 (un par image à downsample) est réinitialisé à 0 avant chaque dispatch. Chaque threadgroup fait `atomicAdd(counter, 1)` en fin de phase 1. Le threadgroup qui obtient la valeur `numGroups - 1` sait qu'il est le dernier → il seul exécute la phase 2.

### LDS layout

Chaque threadgroup alloue `~64 × 64 / 4 = 1024 floatN` en LDS pour les niveaux intermédiaires. Pour RGBA16F → `4 × 1024 × 4 bytes = 16 KB` par threadgroup (bien dans les 48–64 KB disponibles sur RDNA/Turing+).

---

## 3. Design générique dans i3

### Trait `SpdReduction` (Slang)

```slang
// Interface de réduction — instanciée statiquement à la compilation
interface ISpdReduction {
    associatedtype T;               // type de donnée (float, float4, ...)
    static T reduce(T a, T b);      // opération 2×2 → 1
    static T load(Texture2D<T> src, uint2 coord);
    static void store(RWTexture2D<T> dst, uint2 coord, T value);
}

// Instance MAX pour HiZ (reverse-Z)
struct SpdMaxReduction : ISpdReduction {
    typedef float T;
    static float reduce(float a, float b) { return max(a, b); }
    // ...
}

// Instance AVG pour SceneLit (couleur HDR)
struct SpdAvgReduction : ISpdReduction {
    typedef float4 T;
    static float4 reduce(float4 a, float4 b) { return (a + b) * 0.5; }
    // ...
}
```

### Rust : `SpdPass<R>`

```rust
pub struct SpdPass {
    pub input_name:  &'static str,
    pub output_name: &'static str,
    reduction:       SpdReduction,   // enum { Max, Avg }
    pipeline:        Option<BackendPipeline>,
    counter_buffer:  Option<BufferHandle>,  // u32[1], réinitialisé chaque frame
}

pub enum SpdReduction { Max, Avg }

impl SpdPass {
    pub fn new_hiz_final() -> Self {
        Self { input_name: "DepthBuffer", output_name: "HiZFinal", reduction: SpdReduction::Max, ... }
    }
    pub fn new_hiz_prez() -> Self {
        Self { input_name: "DepthPreZ", output_name: "HiZPreZ", reduction: SpdReduction::Max, ... }
    }
    pub fn new_scene_lit_mips() -> Self {
        Self { input_name: "SceneLit", output_name: "SceneLit", reduction: SpdReduction::Avg, ... }
    }
}
```

> `new_scene_lit_mips()` lit et écrit le même nom — le mip 0 est en lecture seule, les mips 1..N sont écrits.

---

## 4. Ressources nécessaires

| Ressource         | Format             | Notes                                             |
|-------------------|--------------------|---------------------------------------------------|
| Image source      | quelconque         | Mip 0, lecture seule                              |
| Image output      | même format        | Mips 1..N, storage write                          |
| `SpdCounterBuffer`| R32_UINT           | 1 u32, reset à 0 avant chaque dispatch, persistant|

---

## 5. Pipeline de passes (remplacement)

```
Avant (HiZ) :
  HiZBlit → HiZReduce[0] → HiZReduce[1] → ... → HiZReduce[N-1]
  (N barriers GPU)

Après (SPD) :
  SpdPass  (1 seul dispatch, 0 barrier)
```

---

## 6. Tâches

---

### M0 — Shader SPD générique

#### T0.1 — `spd.slang` : shader SPD paramétré

**Fichier (nouveau) :** `assets/shaders/spd.slang`

Le shader est écrit une fois. La réduction est sélectionnée via une **spécialisation constante** (`[[vk::constant_id(0)]] const uint REDUCTION_MODE`) ou via deux entry points distincts (`spd_max` / `spd_avg`).

**Recommandation :** deux entry points dans le même fichier — plus simple à déboguer et évite les spécialisations de pipeline.

Push constants :
```slang
struct SpdPushConstants {
    uint  mip_count;        // nombre de mips à générer (max 12)
    uint  num_work_groups;  // nombre total de threadgroups = ceil(w/64) * ceil(h/64)
    uint2 work_group_offset;// (0,0) sauf pour des sous-régions
    uint2 src_size;         // taille du mip 0
};
```

Bindings (set 0) :
```slang
[[vk::binding(0, 0)]] Texture2D<float>    srcDepth;           // HiZ only
[[vk::binding(0, 0)]] Texture2D<float4>   srcColor;           // SceneLit only
[[vk::binding(1, 0)]] globallycoherent RWTexture2D<float>  dstMips[12];   // par mip
[[vk::binding(2, 0)]] globallycoherent RWStructuredBuffer<uint> spdCounter;
```

> `globallycoherent` est **obligatoire** sur les RWTextures et le buffer counter — sans ça, les writes ne sont pas visibles entre threadgroups sur certains hardware.

LDS :
```slang
groupshared float4 g_lds[16][16];  // stockage intermédiaire phase 1→2
// Pour le format float (HiZ) : groupshared float g_lds[16][16];
```

Structure de l'entry point :
```
[numthreads(256, 1, 1)]
void spd_max(uint3 gid : SV_GroupID, uint lid : SV_GroupIndex):
    1. Calculer la tuile couverte par ce threadgroup (64×64 pixels de mip 0)
    2. Chaque thread réduit un bloc 4×4 → écrit 1 valeur en mip 1 et LDS
    3. Réduction LDS en arbre : mips 2, 3, 4, 5 (chaque étape divise par 2)
    4. DeviceMemoryBarrier() — flush des writes mip 1..5 vers mémoire globale
    5. atomicAdd(spdCounter[0], 1, old_val)
    6. Si old_val == num_work_groups - 1 :
        // Ce threadgroup est le dernier
        7. Relire mip 5 depuis la mémoire globale → charger dans LDS
        8. Réduction LDS : mips 6, 7, 8, 9, 10, 11, 12
        9. Remettre spdCounter[0] à 0 pour le prochain frame
```

**Points tricky :**

- **`DeviceMemoryBarrier()`** entre phase 1 et phase 2 : sans ça, le dernier threadgroup lirait des données de phase 1 non committées par les autres. Slang → `AllMemoryBarrier()` ou `DeviceMemoryBarrier()`.
- **Thread mapping** : 256 threads par group (16×16), chaque thread traite **4 pixels** au step de mip 1 (donc couvre 32×32 pixels au total), puis chaque groupe de 4 threads collabore pour mip 2, etc.
- **Dimensions non-POT** : SPD gère nativement les tailles non-power-of-two — les indices out-of-bounds sont clampés.
- **Mip count < 12** : les étapes qui produiraient des mips > mip_count sont skippées avec un early-out conditionnel.

#### T0.2 — Pipelines `.ron`

```ron
// spd_max.ron
PipelineConfig(name: "spd_max", shader: Path("../shaders/spd.slang"), entry: "spd_max", graphics: None)

// spd_avg.ron
PipelineConfig(name: "spd_avg", shader: Path("../shaders/spd.slang"), entry: "spd_avg", graphics: None)
```

---

### M1 — Rust : `SpdPass`

#### T1.1 — Struct et `declare()`

```rust
fn declare(&mut self, builder: &mut PassBuilder) {
    let src = builder.resolve_image(self.input_name);
    let desc = builder.get_image_desc(src);
    let mip_count = compute_mip_count(desc.width, desc.height);  // min(12, floor(log2(max(w,h)))+1)

    // Pour un nouvel output (HiZ) : declare_image_output avec mip_levels = mip_count
    // Pour in-place (SceneLit) : l'image existe déjà avec ses mips

    let num_groups_x = (desc.width  + 63) / 64;
    let num_groups_y = (desc.height + 63) / 64;
    self.num_work_groups = num_groups_x * num_groups_y;

    // Lire mip 0 en input
    builder.read_image(src, ResourceUsage::SHADER_READ);
    // Écrire mips 1..N en storage
    for mip in 1..mip_count {
        builder.write_image_mip(dst, mip, ResourceUsage::SHADER_WRITE);
    }
}
```

#### T1.2 — `execute()` : reset counter + dispatch

```rust
fn execute(&self, ctx: &mut dyn PassContext, _frame: &FrameBlackboard) {
    // Reset atomic counter à 0 (fill buffer)
    ctx.fill_buffer(self.counter_buffer, 0);
    ctx.pipeline_barrier(/* buffer write → shader read */);

    ctx.bind_pipeline_raw(self.pipeline);
    // Binder src, dst mips (array de RWTextures), counter
    // Push constants : mip_count, num_work_groups, src_size
    ctx.dispatch(self.num_work_groups, 1, 1);
}
```

---

### M2 — Migration HiZ

#### T2.1 — Remplacer `HiZBuildPass` par `SpdPass::new_hiz_*()`

Supprimer `HiZBlitSubPass`, `HiZReduceSubPass` et le shader `hiz_build.slang`.  
Dans `render_graph.rs` :
```rust
// Avant :
builder.add_pass(&mut self.hiz_build_final);
// Après :
builder.add_pass(&mut self.spd_hiz_final);  // SpdPass::new_hiz_final()
```

**Test :** valider que le HiZ produit exactement les mêmes valeurs (RenderDoc, comparaison mip par mip). La réduction MAX doit être identique.

---

### M3 — Migration SceneLit

#### T3.1 — Remplacer `SceneLitMipPass` multi-dispatch par `SpdPass::new_scene_lit_mips()`

```rust
// Avant :
builder.add_pass(&mut self.scene_lit_mip_pass);  // multi-dispatch
// Après :
builder.add_pass(&mut self.spd_scene_lit);        // SpdPass::new_scene_lit_mips()
```

---

## 7. Points de vigilance

| Risque                              | Mitigation                                                                      |
|-------------------------------------|---------------------------------------------------------------------------------|
| `globallycoherent` manquant         | Obligatoire sur toutes les RWTextures et le counter — sinon corruption silencieuse |
| Counter non remis à zéro            | Le dernier threadgroup doit reset le counter (ou faire un `fill_buffer` CPU-side avant dispatch) |
| Mip array en SPIR-V                 | Vérifier que Slang génère bien un descriptor array pour les 12 RWTextures — sinon binder mip par mip |
| Dimensions non-POT                  | Tester explicitement sur 1920×1080 (non-POT) et 1024×512 (POT)                  |
| LDS overflow                        | 16×16×4 floats = 4KB (float) ou 16KB (float4) — dans les limites, mais à vérifier sur GPU cible |
| Validation layer warnings           | `globallycoherent` génère parfois des warnings sur certains drivers — documenter si ça arrive |

---

## 8. Ordre de priorité

```
T0.1 (shader spd.slang, entry spd_max seulement)
    → T0.2 (pipeline spd_max.ron)
    → T1 (SpdPass Rust, test avec HiZFinal)
    → Validation RenderDoc vs HiZ multi-dispatch
    → T2 (migration HiZ complète — suppression HiZBuildPass)
    → T0.1 bis (entry spd_avg)
    → T3 (migration SceneLit)
```

Commencer par `spd_max` seul permet de valider la logique SPD sur un cas simple (scalaire float) avant d'attaquer `spd_avg` sur float4.
