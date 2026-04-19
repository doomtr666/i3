# Stochastic Screen-Space Reflections (SSSR)

## Motivation

Le SSR déterministe HiZ actuel souffre de problèmes structurels difficiles à corriger :

| Problème | Cause racine |
|---|---|
| Halo gris autour des objets | Auto-intersection au départ du rayon |
| Reflets noirs sur surfaces horizontales | Rayon vers la caméra, frustum clip agressif |
| Bandes à distance | Interpolation profondeur écran-espace |
| Pas de roughness réel | Blur heuristique sur le HDR mip, pas de BRDF |
| Composite double-compte l'IBL | Additif pur, l'IBL est déjà dans HDR |

Le SSSR résout ces problèmes en échantillonnant **un seul rayon par pixel par frame** depuis la distribution GGX, et en débruitant sur plusieurs frames par accumulation temporelle. La roughness est gérée naturellement : surfaces lisses → variance faible → convergence rapide ; surfaces rugueuses → variance élevée → flou temporel.

**Référence** : Tomasz Stachowiak, "Stochastic Screen-Space Reflections", Frostbite / GDC 2015.

---

## Vue d'ensemble du pipeline

```
GBuffer + HDR_Target
        │
        ▼
[SssrSamplePass]      1 rayon GGX/pixel/frame — marche linéaire
        │ SSR_Raw  (RGBA16F : rgb=couleur×poids, a=conf)
        ▼
[SssrTemporalPass]    reprojection + EMA
        │ SSR_Resolved (RGBA16F : rgb=couleur, a=confiance)
        ▼
[SssrCompositePass]   remplace IBL specular par SSR là où conf > 0
        │
        ▼
    HDR_Target  (correct, sans double-compte)
```

**Position dans le pipeline :** après `deferred_resolve`, avant bloom.

---

## Algorithme détaillé

### Phase 1 — Échantillonnage GGX (SssrSamplePass)

```
Pour chaque pixel (roughness ≤ cutoff) :

1. Bruit bleu déterministe par frame :
     (u1, u2) = IGN(px) rotaté d'angle doré × frameIndex
     (u2 décalé de 0.5 pour diversifier l'axe azimutal)

2. Importance-sampling GGX dans l'espace tangent :
     α       = roughness²                          // GGX width
     cosθ²   = (1 − u2) / (1 + (α²−1)·u2)        // Trowbridge-Reitz
     sinθ    = sqrt(1 − cosθ²)
     φ       = 2π·u1
     H_local = (sinθ·cos(φ), sinθ·sin(φ), cosθ)

3. Transformer H en espace vue (TBN depuis N_view) :
     H_view = TBN · H_local
     R      = reflect(−V, H_view)               // direction de réflexion

4. Marche linéaire en écran-espace (deux niveaux) :
     — Coarse : 32 pas de taille fixe (ray_px / 32)
     — Dès qu'on passe derrière la surface : 8 pas de bisection
     — Profondeur perspective-correcte : rayDepth = lerp(cz,cz_e,t)/lerp(cw,cw_e,t)
     — Frustum clip si clip_e.w ≤ 0 (rayon vers caméra)

5. Sur hit :
     hit_color = hdrTarget.SampleLevel(LINEAR, hit_uv, 0)
     // Poids GGX importance-sampled (voir §Poids ci-dessous)
     weight = schlick_fresnel(V, H_view, F0) * G2(V, R, N, α)
              · dot(V, H_view) / (dot(N, H_view) · dot(N, V))
     conf   = edge_fade(hit_uv) · (1 − hit_t²)
     SSR_Raw[px] = float4(hit_color × weight × conf, conf)

6. Sur miss :
     SSR_Raw[px] = float4(0, 0, 0, 0)
```

**Poids simplifié** : si l'importance-sampling GGX est appliqué correctement, le terme
`D(H)·cos(θ_H)` du numérateur et du dénominateur PDF se simplifient, et le poids résiduel est :

```
weight = F(V,H) · G2(V,L,N,α) · dot(V,H) / (dot(N,H) · dot(N,V))
```

où `G2` est le terme de masquage-ombrage Smith height-correlated (cf. `lighting.slangh`).

### Phase 2 — Débruitage temporel (SssrTemporalPass)

Identique au pass temporel SSR actuel, avec :
- Reprojection via profondeur + matrice view/proj précédente
- Rejet de rehistory si la profondeur reprojetée diverge (> 0.1 linéaire)
- EMA avec alpha configurable (0.9 = stable, 0.0 = pas d'accumulation) :
  ```
  ssrResolved = lerp(history, current, 1 − α)   // sur les pixels avec hit
  ```
- Conservation du canal alpha (confiance) pour le composite

### Phase 3 — Composite correct (SssrCompositePass)

**Problème actuel** : le composite additif `hdr += ssr × weight` double-compte l'IBL specular déjà présent dans HDR.

**Solution** : le composite doit *remplacer* l'IBL specular par le SSR, pas s'y additionner.

L'IBL specular calculée dans `deferred_resolve.slang` vaut :
```
ibl_spec = prefiltered_env(R, roughness) × (F0·brdf.x + brdf.y)
```

Le composite recalcule cette valeur (les textures IBL sont bindless) et fait :
```
hdr[px] = hdr[px]
        + conf · ssr_color · brdf_weight      // contribution SSR
        − conf · ibl_spec                     // retire ce que l'IBL avait mis
```

Soit en une ligne :
```
hdr[px] += conf · (ssr_color · brdf_weight − ibl_spec)
```

Quand `conf = 0` : hdr inchangé (IBL seul). Quand `conf = 1` : SSR remplace exactement l'IBL.

Le composite a besoin de :
- `ssrResolved`   (rgb = couleur×poids, a = conf)
- `gbufferAlbedo`, `gbufferRoughMetal`, `gbufferNormal`
- IBL prefiltered + BRDF LUT (via bindless, index dans push constants)

---

## Fichiers à créer

### Shaders (`crates/i3_renderer/assets/shaders/`)

| Fichier | Rôle |
|---|---|
| `sssr_sample.slang` | Échantillonnage GGX + marche linéaire, écriture SSR_Raw |
| `sssr_temporal.slang` | EMA temporelle sur SSR_Raw → SSR_Resolved (peut réutiliser ssr_temporal.slang comme base) |
| `sssr_composite.slang` | Remplace IBL spec par SSR (nouveau, remplace ssr_composite.slang) |

### Pipelines (`crates/i3_renderer/assets/pipelines/`)

`sssr_sample.i3p`, `sssr_temporal.i3p`, `sssr_composite.i3p`

### Rust (`crates/i3_renderer/src/passes/sssr.rs`)

```rust
pub struct SssrPass {
    pub enabled:          bool,
    pub alpha:            f32,     // accumulation temporelle (0.0 = désactivé)
    pub roughness_cutoff: f32,
    pub intensity:        f32,
    pub max_steps:        u32,
    pub max_distance:     f32,
    pub thickness:        f32,
    // IBL bindless indices (copiés depuis le deferred pass pour le composite)
    pub ibl_lut_index:    u32,
    pub ibl_pref_index:   u32,
    pub ibl_intensity:    f32,
    ...
}
```

Structure de sous-passes identique au pattern BloomPass :
- `SssrSampleSubPass`
- `SssrTemporalSubPass`
- `SssrCompositeSubPass`

---

## Fichiers à supprimer

| Fichier | Raison |
|---|---|
| `assets/shaders/ssr_main.slang` | Remplacé par `sssr_sample.slang` |
| `assets/shaders/ssr_temporal.slang` | Remplacé par `sssr_temporal.slang` |
| `assets/shaders/ssr_composite.slang` | Remplacé par `sssr_composite.slang` |
| `assets/pipelines/ssr_main.i3p` | idem |
| `assets/pipelines/ssr_temporal.i3p` | idem |
| `assets/pipelines/ssr_composite.i3p` | idem |
| `src/passes/ssr.rs` | Remplacé par `sssr.rs` |

**À conserver** : `hiz_build.slang` et la pyramide HiZ — utilisés pour le HiZ culling GPU-driven (pas spécifique au SSR).

---

## Fichiers à modifier

### `src/passes/mod.rs`
```rust
pub mod sssr;
// supprimer : pub mod ssr;
```

### `src/render_graph.rs`
- Remplacer `ssr_pass: SsrPass` par `pub sssr_pass: SssrPass`
- Dans `record_lighting()` : remplacer les 3 sous-passes SSR par les 3 sous-passes SSSR
- Passer les IBL indices au SssrPass depuis les paramètres existants du deferred pass

### `examples/viewer/src/main.rs`
- Remplacer le bloc GUI SSR par un bloc SSSR avec les mêmes sliders
- Mettre à jour la référence dans le `DebugChannel::SsrResolved` (inchangé)

---

## Push constants

### sssr_sample.slang
```c
struct SssrSamplePc {
    float4x4 inv_projection;
    float4x4 projection;
    float4x4 view;
    float4x4 prev_view_proj;   // pour reprojection dans le temporal
    float2   screen_size;
    float    max_distance;
    float    thickness;
    float    roughness_cutoff;
    uint     max_steps;
    uint     frame_index;
    uint     enabled;
};
```

### sssr_temporal.slang
```c
struct SssrTemporalPc {
    float4x4 inv_projection;
    float4x4 prev_view_proj;
    float2   screen_size;
    float    alpha;
    uint     frame_index;
};
```

### sssr_composite.slang
```c
struct SssrCompositePc {
    float    intensity;
    uint     ibl_lut_index;
    uint     ibl_pref_index;
    float    ibl_intensity;
};
```

---

## Ordre d'implémentation

1. **`sssr_sample.slang` + `.i3p`** — GGX sampling + marche linéaire (peut commencer sans composite)
2. **`sssr_temporal.slang` + `.i3p`** — EMA temporelle (adapter ssr_temporal.slang)
3. **`sssr_composite.slang` + `.i3p`** — composite avec replace IBL
4. **`sssr.rs`** — câblage Rust (sous-passes, declare_image_output pour SSR_Raw et SSR_Resolved)
5. **Câblage `render_graph.rs`** — remplacer ssr_pass par sssr_pass
6. **Supprimer les anciens fichiers SSR** (shaders, pipelines, passes/ssr.rs)
7. **GUI viewer** — mise à jour des sliders

---

## Vérification

1. `cargo build -p i3_renderer -p viewer` — zéro erreur
2. Activer SSSR, roughness_cutoff = 1.0, alpha = 0.9
3. Scène 3 cubes sur sol miroir : reflets corrects, pas de halo, pas de bandes
4. Monter roughness : reflets flous (temporel accumule des directions variées)
5. alpha = 0.0 : voir le bruit brut 1 spp — normal, fortement bruité
6. alpha = 0.9 : convergence en ~20 frames — lisse
7. Debug SSR_Resolved : couleurs cohérentes avec la scène
8. IBL seul (SSSR désactivé) : scène identique à avant
9. SSSR activé sur sol miroir (roughness=0) : IBL specular est remplacée (pas additionnée) — vérifier en comparant l'intensité spéculaire avec/sans SSSR
