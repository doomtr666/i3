# Ground Truth Ambient Occlusion (GTAO)

> Référence algorithmique : "Practical Real-Time Strategies for Accurate Indirect Occlusion" — Jimenez et al., SIGGRAPH 2016.  
> Implémentations de référence : XeGTAO (Intel), FidelityFX CACAO (AMD).

---

## 1. Principe

GTAO est une AO basée sur l'intégrale horizon. Pour chaque pixel :

1. Reconstruire la position view-space depuis le depth buffer.
2. Lire la normale view-space depuis le GBuffer.
3. Pour **S directions** azimutales réparties sur le demi-cercle :
   - Marcher dans la direction projetée en screen-space.
   - Trouver les angles d'horizon `h₁` (côté +) et `h₂` (côté −) en échantillonnant le depth.
   - Calculer la contribution AO par tranche : `AO_slice = 1 − cos((h₁ + h₂) / 2 + γ)` où `γ` est l'angle de la normale projetée dans le plan de coupe.
4. Moyenner sur toutes les directions.

La formulation donne un résultat qui converge vers le vrai AO hémisphérique quand S → ∞.

---

## 2. Ressources nécessaires

| Ressource         | Format           | Source                    | Notes                                  |
|-------------------|------------------|---------------------------|----------------------------------------|
| `DepthBuffer`     | D32_FLOAT        | GBuffer pass              | Depth scène complet                    |
| `HiZFinal`        | R32_SFLOAT (mips)| HiZBuildPass::new_final() | Mips pour horizon search + large steps |
| `GBuffer_Normal`  | RG16_SFLOAT      | GBuffer pass              | Normale octaédrique encodée            |
| `AO_Raw`          | R8_UNORM         | GTAO Main pass (output)   | Transient                              |
| `AO_Blurred`      | R8_UNORM         | GTAO Denoise pass         | Transient                              |
| `AO_History`      | R8_UNORM         | TemporalRegistry          | Accumulation temporelle                |

---

## 3. Pipeline de passes

```
DepthBuffer + HiZFinal + GBuffer_Normal
        │
        ▼
  ┌─────────────┐
  │  GTAO Main  │  (compute, 1 thread/pixel, interleaved 2×2)
  └──────┬──────┘
         │ AO_Raw
         ▼
  ┌─────────────────┐
  │  Spatial Denoise│  (compute, bilateral blur 3×3 ou 5×5)
  └──────┬──────────┘
         │ AO_Blurred
         ▼
  ┌─────────────────────┐
  │  Temporal Accumulate│  (compute, reprojection + blend)
  └──────┬──────────────┘
         │ AO_History (write N) + AO_Final
         ▼
  DeferredResolve (lit AO_Final * ambient)
```

---

## 4. Tâches

---

### M0 — Infrastructure

#### T0.1 — Déclarations de ressources dans le frame graph

**Fichier :** `render_graph.rs`

```rust
// AO_History — temporel, double-buffered
builder.declare_image_history_output("AO_History", ImageDesc {
    format: Format::R8_UNORM,
    usage: SAMPLED | STORAGE,
    // dimensions = screen
});
```

`AO_Raw` et `AO_Blurred` sont transients (declare_image_output dans leurs passes respectives).

**Test :** compilation passe.

---

#### T0.2 — Passes Rust squelettes

**Fichiers (nouveaux) :** `passes/gtao.rs`

Quatre structs :
- `GtaoMainPass` (leaf compute)
- `GtaoDenoisePass` (leaf compute)
- `GtaoTemporalPass` (leaf compute)
- `GtaoGroup` (group, déclare les 3 enfants)

Exposer `GtaoGroup` dans `passes/mod.rs` et l'enregistrer dans `DefaultRenderGraph`.

**Test :** compilation passe, passes no-op.

---

### M1 — Pass principale GTAO

#### T1.1 — Shader `gtao_main.slang`

**Fichier (nouveau) :** `assets/shaders/gtao_main.slang`

Push constants :
```slang
struct GtaoPushConstants {
    float4x4 view;
    float4x4 projection;      // pour reconstruire view-space pos
    float4x4 invProjection;
    float2   screenSize;
    float    radius;          // rayon monde (ex. 0.5m)
    float    falloff;         // puissance de la falloff (ex. 2.0)
    uint     sliceCount;      // nb de directions azimutales (ex. 2 ou 3)
    uint     stepCount;       // pas par direction (ex. 4–6)
    uint     frameIndex;      // pour rotation temporelle du bruit
    uint     _pad;
};
```

Bindings (set 0) :
```slang
[[vk::binding(0, 0)]] Texture2D<float>   depthBuffer;   // D32_FLOAT samplé comme R32
[[vk::binding(1, 0)]] Texture2D<float2>  gbufferNormal; // RG16 oct-encoded
[[vk::binding(2, 0)]] Texture2D<float>   hizPyramid;    // HiZFinal pour large steps
[[vk::binding(3, 0)]] RWTexture2D<float> aoOutput;      // AO_Raw
```

Algorithme :
1. Reconstruire position view-space `P` depuis depth + `invProjection`.
2. Décoder la normale view-space `N` depuis GBuffer.
3. Rotation des directions par `frameIndex * goldenAngle` (bruit spatio-temporel).
4. Pour chaque slice :
   - Direction screen-space `d = rotate(vec2(1,0), slice_angle)`.
   - Marcher `stepCount` pas dans `+d` et `-d`, augmentant exponentiellement (mip HiZ).
   - À chaque pas : lire depth à la position marchée, calculer angle horizon en view-space.
   - Garder `maxH1`, `maxH2`.
   - Contribution : `ao_slice = 1 - cos((maxH1 + maxH2) * 0.5 + gamma)` clampé à `[0,1]`.
5. Écrire `ao = mean(ao_slice[0..S])` dans `aoOutput`.

**Bruit :** Séquence de Hilbert (16 valeurs) + rotation dorée par frame pour TAA.

**Test :** AO visible dans RenderDoc sur `AO_Raw`.

---

#### T1.2 — Pipeline `gtao_main.i3p`

```
PipelineConfig(
    name: "gtao_main",
    shader: Path("../shaders/gtao_main.slang"),
    graphics: None,
)
```

---

#### T1.3 — Rust : `GtaoMainPass::execute()`

Résoudre les handles et binder :
- `depthBuffer` → `builder.resolve_image("DepthBuffer")`
- `gbufferNormal` → `builder.resolve_image("GBuffer_Normal")`
- `hizPyramid` → `builder.resolve_image("HiZFinal")`
- Déclarer `AO_Raw` en output via `declare_image_output`

Dispatch : `ceil(width/8) × ceil(height/8)` groupes de `[8, 8, 1]` threads.

---

### M2 — Dénoise spatial

#### T2.1 — Shader `gtao_denoise.slang`

Filtre bilatéral **3×3** sur `AO_Raw` :
- Poids basé sur la différence de depth (σ_depth) et de normale (σ_normal).
- Depth et normale lus directement pour le poids — pas besoin de resampler le GBuffer.
- Séparable (H puis V) ou un seul pass 3×3 selon le budget.

```slang
[[vk::binding(0, 0)]] Texture2D<float>   aoRaw;
[[vk::binding(1, 0)]] Texture2D<float>   depthBuffer;
[[vk::binding(2, 0)]] Texture2D<float2>  gbufferNormal;
[[vk::binding(3, 0)]] RWTexture2D<float> aoBlurred;
```

**Test :** `AO_Blurred` visible dans RenderDoc, bruit réduit vs `AO_Raw`.

---

### M3 — Accumulation temporelle

#### T3.1 — Shader `gtao_temporal.slang`

```slang
[[vk::binding(0, 0)]] Texture2D<float>   aoCurrent;    // AO_Blurred
[[vk::binding(1, 0)]] Texture2D<float>   aoHistory;    // AO_History N-1
[[vk::binding(2, 0)]] Texture2D<float>   depthBuffer;  // pour reprojection
[[vk::binding(3, 0)]] RWTexture2D<float> aoOutput;     // AO_History N (write)
```

Push constants : `viewProjection`, `prevViewProjection`, `screenSize`.

Algorithme :
1. Reprojeter le pixel courant via `prevVP * invVP` → UV historique.
2. Lire `aoHistory` au UV reprojeté (bilinéaire).
3. Validity check : comparer depth reprojeté vs depth historique (seuil ε).
4. Blend : `ao_final = lerp(aoCurrent, aoHistory, alpha)` avec `alpha = 0.9` (valide) ou `0.0` (invalide = reset).
5. Écrire dans `aoOutput`.

**Test :** l'AO est stable en mouvement statique, se remet à jour quand la caméra bouge vite.

---

### M4 — Intégration dans le Deferred Resolve

#### T4.1 — Brancher `AO_Final` dans `deferred_resolve.slang`

```slang
[[vk::binding(N, 0)]] Texture2D<float> aoTexture;

// Dans le calcul de lighting :
float ao = aoTexture.Sample(samplerLinear, uv).r;
float3 ambient = ibl_ambient * albedo * ao;  // ou * ao² selon goût artistique
```

`AO_Final` = sortie du pass temporal = `AO_History` frame N.

**Test :** l'AO s'applique visuellement sur la scène, visible dans le canal Lit.

---

## 5. Paramètres exposés

| Paramètre      | Valeur défaut | Description                                     |
|----------------|---------------|-------------------------------------------------|
| `radius`       | 0.5 m         | Rayon monde de recherche AO                     |
| `sliceCount`   | 2             | Directions azimutales (2=perf, 3=qualité)       |
| `stepCount`    | 5             | Pas par direction                               |
| `falloff`      | 2.0           | Puissance de la falloff distance                |
| `alpha`        | 0.90          | Facteur d'accumulation temporelle               |
| `aoStrength`   | 1.0           | Intensité finale (0 = désactivé, 1 = full)      |

---

## 6. Notes d'implémentation

### Reconstruction position view-space

```slang
float depth = depthBuffer.Load(int3(px, 0)).r; // reverse-Z [0,1]
float2 uv   = (float2(px) + 0.5) / screenSize;
float2 ndc  = uv * 2.0 - 1.0;
// Vulkan Y-Up avec negative viewport → flip Y pour passer en clip space texture
ndc.y = -ndc.y;
float4 clip = float4(ndc, depth, 1.0);
float4 view = mul(invProjection, clip);
float3 P    = view.xyz / view.w;
```

### Rotation temporelle des directions

```slang
// Séquence de Hilbert 16 frames, rotation dorée entre frames
static const float kGoldenRatio = 0.6180339887;
float angleOffset = frac(float(frameIndex) * kGoldenRatio) * PI;
float sliceAngle  = (float(slice) / float(sliceCount)) * PI + angleOffset;
```

### Utilisation du HiZ pour les grands pas

Aux premiers pas (proches), lire `depthBuffer` directement (mip 0, précision max).  
Aux pas lointains, lire `hizPyramid` à un mip croissant — un seul tap couvre une zone plus large, sans aliasing.

```slang
float stepLen  = radius / float(stepCount);
float mipLevel = log2(max(stepLen * screenSize.x, 1.0));
float d = hizPyramid.SampleLevel(samplerNearest, sampleUV, mipLevel).r;
```

### Encodage GBuffer Normal

Les normales sont encodées en octaédrique dans `GBuffer_Normal` (RG16_SFLOAT, espace monde).  
Dans GTAO, transformer en view-space : `N_view = normalize(mul(float3x3(view), N_world))`.

---

## 7. Ordre de priorité

```
T0.1 (resources) → T0.2 (squelettes) → T1 (main pass)
→ Validation visuelle RenderDoc → T2 (denoise) → T3 (temporal) → T4 (apply)
```

Le dénoise et le temporal peuvent être skippés pour un premier test visuel — `AO_Raw` peut être appliqué directement dans le deferred resolve pendant le développement.
