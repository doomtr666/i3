# Bloom

> Référence algorithmique : "Next Generation Post Processing in Call of Duty: Advanced Warfare" — Jorge Jimenez, SIGGRAPH 2014.  
> Référence complémentaire : "Physically Based and Unified Volumetric Rendering in Frostbite" — Sébastien Hillaire, SIGGRAPH 2015.

---

## 1. Principe

Le bloom simule la diffusion de la lumière dans les optiques : les zones très lumineuses de l'image HDR "saignent" sur leurs voisins. L'algorithme opère en trois phases :

1. **Préfiltre** — Extraire les pixels dépassant un seuil de luminance (avec soft-knee pour éviter les coupures brutales). Résultat à demi-résolution.
2. **Downsample** — Réduire progressivement par un filtre 13-tap bilinéaire jusqu'à un minimum (mip N).
3. **Upsample** — Remonter mip par mip avec un filtre tent 9-tap en accumulant, puis additionner sur `HDR_Target`.

L'ensemble opère en espace HDR linéaire avant le tonemap, ce qui est physiquement correct (la lumière s'additionne avant la courbe de compression).

```
HDR_Target mip 0
      │
      ▼
[BloomPrefilter]     → Bloom_Buffer mip 0  (W/2 × H/2, threshold + soft-knee)
      │
      ▼ × (N-1) descentes
[BloomDownsample]    → Bloom_Buffer mip 1 … mip N  (filtre 13-tap)
      │
      ▼ × (N-1) remontées
[BloomUpsample]      → Bloom_Buffer mip N-1 … mip 0  (tent 9-tap, accumulatif)
      │
      ▼
[BloomComposite]     → HDR_Target mip 0  +=  Bloom_Buffer mip 0 × intensity
```

---

## 2. Ressources

| Ressource         | Format               | Usage            | Notes                          |
|-------------------|----------------------|------------------|--------------------------------|
| `HDR_Target`      | R16G16B16A16_SFLOAT  | SAMPLED + STORAGE | Source + destination finale   |
| `Bloom_Buffer`    | R16G16B16A16_SFLOAT  | STORAGE + SAMPLED | Transient, W/2 × H/2, 5 mips |

`Bloom_Buffer` est alloué comme image transiente dans le render graph (recréé si la résolution change).

---

## 3. Position dans le pipeline

```
DeferredResolvePass
HdrMipsPass            ← génère les mips HDR pour SSR
SsrMainPass
SsrTemporalPass
SsrCompositePass       ← SSR additionné sur HDR_Target
──────────────────
BloomPass              ← ICI (avant tonemap, en HDR)
──────────────────
PostProcessGroup
  HistogramBuildPass
  AverageLuminancePass
  TonemapPass
  FxaaPass
```

---

## 4. Paramètres

| Paramètre   | Type  | Plage     | Défaut | Rôle                                              |
|-------------|-------|-----------|--------|---------------------------------------------------|
| `enabled`   | bool  | —         | true   | Active/désactive le pass (coût zéro si désactivé) |
| `threshold` | f32   | 0.5 – 4.0 | 1.0    | Luminance à partir de laquelle le bloom commence  |
| `knee`      | f32   | 0.0 – 1.0 | 0.5    | Largeur de la zone de transition douce            |
| `intensity` | f32   | 0.0 – 2.0 | 0.5    | Force de l'addition finale                        |

---

## 5. Algorithme détaillé

### 5.1 Préfiltre (bloom_prefilter.slang)

Bindings :
- `set 0 binding 0` — `hdr_src` (Texture2D, sampled)
- `set 0 binding 1` — `bloom_dst` (RWTexture2D, mip 0)

```hlsl
// Demi-résolution → bilinéaire gratuit
uv = (px + 0.5) / (src_size * 0.5)
col = hdr_src.SampleLevel(LINEAR_CLAMP, uv, 0)
lum = dot(col.rgb, float3(0.2126, 0.7152, 0.0722))

// Soft-knee — courbe quadratique autour du seuil
rq = clamp(lum - threshold + knee, 0, 2 * knee)
rq = (rq * rq) / (4 * knee + 1e-6)
weight = max(rq, lum - threshold) / max(lum, 1e-6)

bloom_dst[px] = float4(col.rgb * weight, 1.0)
```

### 5.2 Downsample 13-tap (bloom_downsample.slang)

Filtre Jimenez — 4 bilinéaires à mi-pixel (coin) + 4 bilinéaires centre-bord + 1 centre, pondérés 0.5 / 0.125 / 0.25 :

```
·  ·  ·  ·  ·
·  A  ·  B  ·
·  ·  C  ·  ·
·  D  ·  E  ·
·  ·  ·  ·  ·
```

Réduit le ringing et les artefacts de répétition par rapport à un simple 2×2.

### 5.3 Upsample tent 9-tap (bloom_upsample.slang)

Filtre tent 3×3, pondéré (coins 1, bords 2, centre 4) / 16.  
L'écriture est **accumulative** : le mip de destination est lu puis réécrit (`bloom_dst[px] += tent_result`), ce qui superpose les contributions de chaque niveau de la remontée.

### 5.4 Composite (bloom_composite.slang)

Lecture + écriture in-place sur `HDR_Target` mip 0 (storage RW, pattern identique à `ssr_composite.slang`) :

```hlsl
hdr[coord] = float4(hdr[coord].rgb + bloom_mip0[coord].rgb * intensity, hdr[coord].a)
```

---

## 6. Structure Rust

### `crates/i3_renderer/src/passes/bloom.rs`

```
BloomPass (impl RenderPass)
  pub enabled: bool
  pub threshold: f32
  pub knee: f32
  pub intensity: f32

  prefilter_pipeline:  Option<BackendPipeline>
  downsample_pipeline: Option<BackendPipeline>
  upsample_pipeline:   Option<BackendPipeline>
  composite_pipeline:  Option<BackendPipeline>

  init()     → charge les 4 pipelines via AssetLoader
  declare()  → déclare Bloom_Buffer, inject les sous-passes
```

#### Sous-passes (add_owned_pass, pattern HdrMipsPass)

| Sous-passe              | Nombre     |
|-------------------------|-----------|
| `BloomPrefilterSubPass` | 1         |
| `BloomDownSubPass`      | mip_count − 1 |
| `BloomUpSubPass`        | mip_count − 1 |
| `BloomCompositeSubPass` | 1         |

#### Push constants

```rust
// prefilter
#[repr(C)] struct BloomPrefilterPc { src_size: [u32;2], threshold: f32, knee: f32 }

// downsample / upsample
#[repr(C)] struct BloomMipPc { src_size: [u32;2], src_mip: u32, _pad: u32 }

// composite
#[repr(C)] struct BloomCompositePc { intensity: f32, _pad: [f32;3] }
```

---

## 7. Fichiers impliqués

| Type    | Fichier                                                              |
|---------|----------------------------------------------------------------------|
| Rust    | `crates/i3_renderer/src/passes/bloom.rs` (nouveau)                  |
| Rust    | `crates/i3_renderer/src/passes/mod.rs` (ajouter `pub mod bloom`)    |
| Rust    | `crates/i3_renderer/src/render_graph.rs` (champ + câblage)          |
| Rust    | `examples/viewer/src/main.rs` (GUI sliders)                         |
| Shader  | `crates/i3_renderer/assets/shaders/bloom_prefilter.slang`           |
| Shader  | `crates/i3_renderer/assets/shaders/bloom_downsample.slang`          |
| Shader  | `crates/i3_renderer/assets/shaders/bloom_upsample.slang`            |
| Shader  | `crates/i3_renderer/assets/shaders/bloom_composite.slang`           |
| Pipeline| `crates/i3_renderer/assets/pipelines/bloom_prefilter.i3p`           |
| Pipeline| `crates/i3_renderer/assets/pipelines/bloom_downsample.i3p`          |
| Pipeline| `crates/i3_renderer/assets/pipelines/bloom_upsample.i3p`            |
| Pipeline| `crates/i3_renderer/assets/pipelines/bloom_composite.i3p`           |

---

## 8. Vérification

1. `cargo build -p i3_renderer -p viewer` — compilation propre
2. Lancer le viewer, activer Bloom — threshold ~1.0, intensity ~0.5
3. Sources lumineuses / zones surexposées → halo visible
4. Threshold > 3.0 → bloom disparaît (aucune contribution)
5. SSR et GTAO inchangés (passes indépendantes)
6. Debug viz `SsrResolved` inchangée
7. Pas de régression tonemap/FXAA
