# Screen Space Reflections (SSR)

> Référence algorithmique : "Hi-Z Screen Space Cone-Traced Reflections" — Yasin Uludag, GPU Pro 5 (2014).  
> Référence complémentaire : "Stochastic Screen Space Reflections" — Tomasz Stachowiak, SIGGRAPH 2015.

---

## 1. Principe

SSR calcule les réflexions en marchant le rayon de réflexion directement dans le screen space :

1. Reconstruire la position view-space `P` et la normale `N` depuis le GBuffer.
2. Calculer la direction de réflexion `R = reflect(-V, N)` où `V = normalize(-P)`.
3. **Marche HiZ** : traverser la pyramide de profondeur (HiZFinal) en avançant par sauts dont la taille est dictée par le mip actuel — on saute les zones vides rapidement et on raffine sur les hits potentiels.
4. Sur hit confirmé (mip 0) : lire `SceneLit` à l'UV d'impact, pondéré par la rugosité.
5. **Accumulation temporelle** : reprojeter le résultat frame précédente et blender pour stabiliser le bruit.

La rugosité pilote deux choses : le seuil d'activation (au-delà de `roughness_cutoff` on skip) et le mip de lecture de `SceneLit` sur hit (surfaces très rugueuses lisent une version pré-blurée).

---

## 2. Ressources nécessaires

| Ressource             | Format            | Source                      | Notes                                           |
|-----------------------|-------------------|-----------------------------|-------------------------------------------------|
| `DepthBuffer`         | D32_SFLOAT        | GBuffer pass                | Reverse-Z [0,1]                                 |
| `HiZFinal`            | R32_SFLOAT (mips) | HiZBuildPass::new_final()   | MAX reduction reverse-Z — réutilisé tel quel    |
| `GBuffer_Normal`      | RG16_SFLOAT       | GBuffer pass                | Octahédrique world-space                        |
| `GBuffer_RoughMetal`  | R8G8_UNORM        | GBuffer pass                | R = roughness, G = metallic                     |
| `SceneLit`            | R16G16B16A16_SFLOAT | DeferredResolvePass       | Couleur HDR frame courante (avant SSR)          |
| `SSR_Raw`             | R16G16B16A16_SFLOAT | SSR Main pass (output)    | Transient — RGB = couleur réfléchie, A = confiance |
| `SSR_Resolved`        | R16G16B16A16_SFLOAT | SSR Temporal pass         | Transient                                       |
| `SSR_History`         | R16G16B16A16_SFLOAT | TemporalRegistry          | Double-buffered, persiste entre frames          |

> **Note cycle** : `SSR_Raw` lit `SceneLit` du frame courant (sans SSR), ce qui implique que les réflexions ne se reflètent pas entre elles. C'est acceptable pour un premier ordre. Pour un second ordre, `SceneLit` peut être remplacé par `SSR_History` N-1.

---

## 3. Pipeline de passes

```
DepthBuffer + HiZFinal + GBuffer_Normal + GBuffer_RoughMetal + SceneLit
        │
        ▼
  ┌───────────┐
  │  SSR Main │  (compute, 1 thread/pixel, marche HiZ)
  └─────┬─────┘
        │ SSR_Raw (RGB = réflexion, A = confiance [0,1])
        ▼
  ┌──────────────────┐
  │  SSR Temporal    │  (compute, reprojection + variance clipping + blend)
  └──────┬───────────┘
         │ SSR_History N (write) + SSR_Resolved
         ▼
  ┌───────────────────────┐
  │  DeferredResolve /    │  additionne SSR_Resolved × F0 × (1-roughness)
  │  Composite SSR pass   │  F0 = lerp(0.04, albedo, metallic)
  └───────────────────────┘
```

---

## 4. Tâches

---

### M0 — Infrastructure

#### T0.1 — Génération des mips de `SceneLit`

SSR a besoin de lire `SceneLit` à différents niveaux de mip pour les surfaces rugueuses (mip élevé = réflexion floue). Actuellement `SceneLit` n'a qu'un seul niveau.

**Approche retenue : multi-dispatch** (même structure que `HiZBuildPass` — blit + réduction mip par mip en AVG).  
> La migration vers SPD est planifiée dans [spd.md](spd.md) et viendra remplacer cette implémentation une fois SPD validé.

**Fichier (nouveau) :** `passes/scene_lit_mips.rs`

```rust
pub struct SceneLitMipsPass {
    reduce_pipeline: Option<BackendPipeline>,
    // sub-passes ajoutées dans declare() pour chaque mip
}
```

Reduction : moyenne 2×2 (contrairement au MAX du HiZ).

```slang
// scene_lit_mips_reduce.slang
float4 a = src.Load(uint3(s + uint2(0,0), 0));
float4 b = src.Load(uint3(s + uint2(1,0), 0));
float4 c = src.Load(uint3(s + uint2(0,1), 0));
float4 d = src.Load(uint3(s + uint2(1,1), 0));
output_tex[dst_coord] = (a + b + c + d) * 0.25;
```

`SceneLit` doit être déclarée avec `mip_levels > 1` dès sa création (dans `DeferredResolvePass`). Vérifier que le format `R16G16B16A16_SFLOAT` supporte `ImageUsageFlags::STORAGE` pour l'écriture des mips (oui sur toutes les cibles Vulkan 1.2+).

**Test :** RenderDoc → inspecter les mips de `SceneLit`, vérifier que la dégradation est cohérente (mip 1 = moitié résolution, mip 4 = 1/16).

---

#### T0.2 — Déclaration des ressources frame graph

**Fichier :** `render_graph.rs`

```rust
// SSR_History — double-buffered, persiste entre frames
builder.declare_image_history_output("SSR_History", ImageDesc {
    format: Format::R16G16B16A16_SFLOAT,
    usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
    // dimensions = screen
});
```

`SSR_Raw` et `SSR_Resolved` sont transients (déclarés dans leurs passes respectives).

**Nommer `SceneLit`** : vérifier que `DeferredResolvePass` publie son output sous ce nom — sinon le renommer ou ajouter un alias dans `render_graph.rs`.

Ajouter `SceneLitMipsPass` dans le graphe, juste après `DeferredResolvePass` :
```rust
builder.add_pass(&mut self.scene_lit_mips_pass);
```

**Test :** compilation passe.

---

#### T0.3 — Passes Rust squelettes SSR

**Fichiers (nouveaux) :**
- `passes/ssr.rs` — contient `SsrMainPass`, `SsrTemporalPass`, `SsrGroup`

```rust
pub struct SsrGroup {
    pub enabled:         bool,
    pub max_steps:       u32,
    pub thickness:       f32,
    pub max_distance:    f32,
    pub roughness_cutoff:f32,
    pub intensity:       f32,
    pub temporal_alpha:  f32,

    main:     SsrMainPass,
    temporal: SsrTemporalPass,
}
```

Exposer `SsrGroup` dans `passes/mod.rs`.  
L'enregistrer dans `DefaultRenderGraph` après `GtaoGroup`, avant `SkyPass`.

**Test :** compilation passe, passes no-op.

---

### M1 — Pass principale SSR

#### T1.1 — Shader `ssr_main.slang`

**Fichier (nouveau) :** `assets/shaders/ssr_main.slang`

Push constants :
```slang
struct SsrPushConstants {
    float4x4 view;
    float4x4 projection;
    float4x4 invProjection;
    float4x4 invView;
    float2   screenSize;
    float    maxDistance;      // distance max du rayon en view-space (ex. 50m)
    float    thickness;        // épaisseur de surface pour hit (ex. 0.1m)
    float    roughnessCutoff;  // skip si roughness > seuil (ex. 0.8)
    uint     maxSteps;         // nombre max de pas HiZ (ex. 48)
    uint     maxMip;           // mip max du HiZ à utiliser
    uint     frameIndex;
};
```

Bindings (set 0) :
```slang
[[vk::binding(0, 0)]] Texture2D<float>    depthBuffer;
[[vk::binding(1, 0)]] Texture2D<float2>   gbufferNormal;
[[vk::binding(2, 0)]] Texture2D<float2>   gbufferRoughMetal;
[[vk::binding(3, 0)]] Texture2D<float>    hizPyramid;
[[vk::binding(4, 0)]] Texture2D<float4>   sceneLit;
[[vk::binding(5, 0)]] RWTexture2D<float4> ssrOutput;   // RGB = réflexion, A = confiance
```

Algorithme (1 thread = 1 pixel) :
```
1. Lire roughness → si > roughnessCutoff : écrire (0,0,0,0), return
2. Reconstruire P_view depuis depth + invProjection
3. Décoder N_world depuis gbufferNormal (octaédrique)
4. N_view = normalize(mul(float3x3(view), N_world))
5. V = normalize(-P_view)
6. R = reflect(-V, N_view)   // direction réflexion view-space
7. Projeter P_view et (P_view + R * maxDistance) en UV screen
8. rayDir_uv = normalize(uvEnd - uvStart)  (+ composante Z en view-space)
9. Marche HiZ :
   currentUV   = uvStart
   currentZ    = P_view.z  (view-space linéaire)
   currentMip  = maxMip
   stepScale   = texelSize(currentMip)
   for (step = 0; step < maxSteps; step++):
       hizDepth = sampleHiZ(currentUV, currentMip)        // profondeur min de la cellule
       rayDepth = linearToDepth(currentZ)                  // depth [0,1] reverse-Z
       if rayDepth > hizDepth :                            // rayon derrière la surface
           if currentMip == 0 :
               hit = true; break
           else :
               currentUV -= rayDir_uv * stepScale * 0.5   // reculer d'un demi-pas
               currentMip--
               stepScale = texelSize(currentMip)
               continue
       // Pas libre → avancer
       currentUV  += rayDir_uv * stepScale
       currentZ   += R.z * (stepScale / abs(rayDir_uv.x + rayDir_uv.y + 1e-5)) * maxDistance
       currentMip  = clamp(currentMip + 1, 0, maxMip)
       stepScale   = texelSize(currentMip)
10. Si hit :
    colorMip    = roughness * 4.0   // surfaces rugueuses → mip élevé
    color       = sceneLit.SampleLevel(linearSampler, currentUV, colorMip).rgb
    confidence  = computeConfidence(currentUV, currentZ, thickness, P_view)
    ssrOutput[px] = float4(color, confidence)
Sinon :
    ssrOutput[px] = float4(0, 0, 0, 0)
```

Confiance — atténuer aux bords du screen et en fonction de l'épaisseur :
```slang
float computeConfidence(float2 uv, float rayZ, float thickness, float3 P) {
    // Atténuation bords
    float2 edgeFade = smoothstep(0.0, 0.1, uv) * smoothstep(1.0, 0.9, uv);
    float  edge     = edgeFade.x * edgeFade.y;

    // Épaisseur : hit valide si le rayon est dans [hizDepth, hizDepth + thickness]
    float  sceneZ   = depthToLinear(depthBuffer.SampleLevel(..., 0));
    float  behindBy = abs(rayZ - sceneZ);
    float  thick    = 1.0 - saturate(behindBy / thickness);

    // Distance du ray (réflexions courtes = plus fiables)
    float  dist     = length(P) / maxDistance;
    float  distFade = 1.0 - dist * dist;

    return edge * thick * distFade;
}
```

**Test :** `SSR_Raw` visible dans RenderDoc, réflexions grossières mais cohérentes.

---

#### T1.2 — Pipeline `ssr_main.i3p`

**Fichier (nouveau) :** `assets/pipelines/ssr_main.ron`

```ron
PipelineConfig(
    name: "ssr_main",
    shader: Path("../shaders/ssr_main.slang"),
    graphics: None,
)
```

---

#### T1.3 — Rust : `SsrMainPass::declare()` + `execute()`

**Declare :**
```rust
fn declare(&mut self, builder: &mut PassBuilder) {
    self.depth    = builder.resolve_image("DepthBuffer");
    self.normal   = builder.resolve_image("GBuffer_Normal");
    self.roughmet = builder.resolve_image("GBuffer_RoughMetal");
    self.hiz      = builder.resolve_image("HiZFinal");
    self.scene    = builder.resolve_image("SceneLit");

    builder.declare_image_output("SSR_Raw", ImageDesc {
        format: Format::R16G16B16A16_SFLOAT,
        usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
    });
    self.output = builder.resolve_image("SSR_Raw");

    builder.read_image(self.depth,    ResourceUsage::SHADER_READ);
    builder.read_image(self.normal,   ResourceUsage::SHADER_READ);
    builder.read_image(self.roughmet, ResourceUsage::SHADER_READ);
    builder.read_image(self.hiz,      ResourceUsage::SHADER_READ);
    builder.read_image(self.scene,    ResourceUsage::SHADER_READ);
    builder.write_image(self.output,  ResourceUsage::SHADER_WRITE);
}
```

**Execute :**
Dispatch : `ceil(width/8) × ceil(height/8)` groupes de `[8, 8, 1]`.

---

### M2 — Accumulation temporelle

#### T2.1 — Shader `ssr_temporal.slang`

```slang
[[vk::binding(0, 0)]] Texture2D<float4>   ssrRaw;       // SSR_Raw frame courante
[[vk::binding(1, 0)]] Texture2D<float4>   ssrHistory;   // SSR_History N-1
[[vk::binding(2, 0)]] Texture2D<float>    depthBuffer;  // pour reprojection
[[vk::binding(3, 0)]] RWTexture2D<float4> ssrOutput;    // SSR_History N (write)
```

Push constants : `viewProjection`, `prevViewProjection`, `invProjection`, `screenSize`, `temporalAlpha`.

Algorithme :
1. Reconstruire `P_world` depuis depth courant.
2. Reprojeter : `uv_prev = (prevVP * float4(P_world, 1.0)).xy / w * 0.5 + 0.5`.
3. Lire `ssrHistory` à `uv_prev` (bilinéaire).
4. **Variance clipping** : calculer la moyenne/variance couleur dans un voisinage 3×3 de `ssrRaw`, clipper la valeur historique dans la boîte AABB de la variance — évite les ghostings sur disocclusion.
5. Validity check : comparer depth reprojeté vs depth historique (seuil `ε = 0.01`).
6. `alpha_effective = temporalAlpha` si valide, sinon `0.0`.
7. `result = lerp(ssrRaw, ssrHistory_clipped, alpha_effective)`.
8. Écrire dans `ssrOutput`.

**Test :** SSR stable en caméra fixe, se remet à jour correctement lors de mouvements.

---

#### T2.2 — Rust : `SsrTemporalPass::declare()` + `execute()`

Pattern identique à `GtaoTemporalPass` — utiliser comme référence directe.

```rust
fn declare(&mut self, builder: &mut PassBuilder) {
    self.raw     = builder.resolve_image("SSR_Raw");
    self.history = builder.resolve_image("SSR_History_N1");  // frame N-1

    builder.declare_image_output("SSR_Resolved", ImageDesc { ... });
    builder.declare_image_history_output("SSR_History_N", ImageDesc { ... });

    self.resolved  = builder.resolve_image("SSR_Resolved");
    self.hist_out  = builder.resolve_image("SSR_History_N");

    builder.read_image(self.raw,     ResourceUsage::SHADER_READ);
    builder.read_image(self.history, ResourceUsage::SHADER_READ);
    builder.write_image(self.resolved, ResourceUsage::SHADER_WRITE);
    builder.write_image(self.hist_out, ResourceUsage::SHADER_WRITE);
}
```

---

### M3 — Composite dans le rendu final

#### T3.1 — Pass composite ou intégration dans `deferred_resolve.slang`

Deux options :

**Option A (simple)** : Ajouter un pass compute `SsrCompositePass` après le resolve :
```slang
[[vk::binding(0, 0)]] Texture2D<float4>   sceneLit;
[[vk::binding(1, 0)]] Texture2D<float4>   ssrResolved;
[[vk::binding(2, 0)]] Texture2D<float4>   gbufferAlbedo;
[[vk::binding(3, 0)]] Texture2D<float2>   gbufferRoughMetal;
[[vk::binding(4, 0)]] RWTexture2D<float4> sceneOutput;   // "SceneFinal"

// Dans le shader :
float4 ssr      = ssrResolved.Load(px);
float  conf     = ssr.a;
float2 rm       = gbufferRoughMetal.Load(px).rg;
float  rough    = rm.r;
float  metallic = rm.g;
float3 albedo   = gbufferAlbedo.Load(px).rgb;

// F0 PBR : diélectriques → 0.04, métaux → albedo
// Les deux réfléchissent, mais avec des intensités très différentes.
float3 F0       = lerp(float3(0.04), albedo, metallic);
float3 weight   = F0 * (1.0 - rough) * conf * ssrIntensity;
sceneOutput[px] = float4(sceneLit.rgb + ssr.rgb * weight, 1.0);
```

**Option B (intégrée)** : Lire `SSR_Resolved` directement dans `deferred_resolve.slang` à côté de l'IBL et l'additionner avec le même poids.

→ **Recommandé** : Option A pour isolation du code SSR.

**Test :** réflexions visibles dans le rendu final, respectant roughness et metallic.

---

### M4 — Debug visualization

#### T4.1 — Nouveau canal dans `DebugChannel`

**Fichier :** `passes/debug_viz.rs`

Ajouter une variante à l'enum :

```rust
pub enum DebugChannel {
    // ... existants (0-6) ...
    SsrResolved = 7,  // SSR_Resolved après accumulation temporelle
}
```

Ajouter le handle dans `DebugVizPass` et le brancher dans `declare()` :

```rust
pub struct DebugVizPass {
    // ... existants ...
    ssr_resolved: ImageHandle,
}

// Dans declare() :
self.ssr_resolved = builder.resolve_image("SSR_Resolved");
builder.read_image(self.ssr_resolved, ResourceUsage::SHADER_READ);

// Dans descriptor_set(0, ...) — après binding 5 (aoResolved) :
d.sampled_image(self.ssr_resolved, DescriptorImageLayout::ShaderReadOnlyOptimal); // binding 6
```

---

#### T4.2 — Shader `debug_viz.slang`

Ajouter le binding et le case :

```slang
[[vk::binding(6, 0)]] Texture2D<float4> ssrResolved;

// Dans le switch :
case 7: // SSR_Resolved
    color = ssrResolved.Sample(bindlessSamplers[SAMPLER_NEAREST], uv).rgb;
    break;
```

---

#### T4.3 — `render_graph.rs` — toujours exécuter SSR pour le canal debug

Sur le modèle de GTAO (qui tourne toujours pour que `AO_Resolved` soit disponible), SSR doit produire `SSR_Resolved` même en mode debug.

```rust
// SSR — toujours run (même logique que GTAO).
// Le pass early-out avec du noir si `enabled = false`.
builder.add_pass(&mut self.ssr_group);

if channel == DebugChannel::Lit || ... {
    builder.add_pass(&mut self.ssr_composite_pass);
    // ...
}
```

> Quand `SsrGroup.enabled = false`, écrire `float4(0,0,0,0)` dans `SSR_Resolved` — même convention que GTAO qui écrit `1.0`.

---

#### T4.4 — GUI egui : dropdown debug channel

```rust
egui::ComboBox::from_label("Debug Channel")
    .show_ui(ui, |ui| {
        // ... canaux existants ...
        ui.selectable_value(&mut self.debug_channel, DebugChannel::SsrResolved, "SSR Resolved");
    });
```

---

### M5 — GUI paramètres SSR

#### T5.1 — Paramètres egui dans le viewer

```rust
// Dans le panel de debug :
ui.collapsing("SSR", |ui| {
    ui.checkbox(&mut ssr.enabled, "Enabled");
    if ssr.enabled {
        ui.add(egui::Slider::new(&mut ssr.max_steps, 8..=128).text("Max Steps"));
        ui.add(egui::Slider::new(&mut ssr.thickness, 0.01..=1.0).text("Thickness"));
        ui.add(egui::Slider::new(&mut ssr.max_distance, 1.0..=200.0).text("Max Distance (m)"));
        ui.add(egui::Slider::new(&mut ssr.roughness_cutoff, 0.0..=1.0).text("Roughness Cutoff"));
        ui.add(egui::Slider::new(&mut ssr.intensity, 0.0..=2.0).text("Intensity"));
        ui.add(egui::Slider::new(&mut ssr.temporal_alpha, 0.0..=0.99).text("Temporal Alpha"));
    }
});
```

---

## 5. Paramètres exposés

| Paramètre          | Valeur défaut | Plage          | Description                                              |
|--------------------|---------------|----------------|----------------------------------------------------------|
| `enabled`          | `true`        | bool           | On/off global                                            |
| `max_steps`        | 48            | 8 – 128        | Nombre max de pas HiZ                                    |
| `thickness`        | 0.15 m        | 0.01 – 2.0     | Épaisseur de surface pour hit detection                  |
| `max_distance`     | 50.0 m        | 1.0 – 500.0    | Distance max du rayon view-space                         |
| `roughness_cutoff` | 0.75          | 0.0 – 1.0      | Au-delà de cette valeur, SSR désactivé                   |
| `intensity`        | 1.0           | 0.0 – 2.0      | Intensité du composite final                             |
| `temporal_alpha`   | 0.92          | 0.0 – 0.99     | Facteur de blend historique (haut = stable, lent)        |

---

## 6. Notes d'implémentation

### Reconstruction position view-space

Identique à GTAO (reverse-Z, Y-flip NDC) :
```slang
float depth = depthBuffer.Load(int3(px, 0)).r;
float2 uv   = (float2(px) + 0.5) / screenSize;
float2 ndc  = uv * 2.0 - 1.0;
ndc.y       = -ndc.y;  // Vulkan Y-flip
float4 clip = float4(ndc, depth, 1.0);
float4 view = mul(invProjection, clip);
float3 P    = view.xyz / view.w;
```

### Marche HiZ — taille de pas

```slang
float2 texelSize(int mip) {
    return 1.0 / (screenSize * pow(2.0, -float(mip)));
}

// À chaque pas : avancer d'un texel au mip courant dans la direction du rayon
float2 stepUV = rayDir_uv * max(abs(texelSize(currentMip).x / rayDir_uv.x),
                                abs(texelSize(currentMip).y / rayDir_uv.y));
```

### Profondeur linéaire view-space depuis reverse-Z

```slang
// Reverse-Z : depth=1 → near, depth=0 → far
// Reconstruire Z linéaire (positif = devant caméra)
float depthToLinearZ(float d, float4x4 proj) {
    // Pour une projection perspective : Z_view = near * far / (far - d * (far - near))
    // Equivalent : extraire depuis la matrice de projection
    float A = proj[2][2];
    float B = proj[3][2];
    return B / (d + A);  // donne Z négatif en view-space right-handed
}
```

### Avancement Z du rayon en view-space

```slang
// Le rayon marche de P_view vers P_view + R * t
// Pour chaque delta UV, calculer le delta t correspondant
float uvLen       = length(stepUV);
float screenDelta = uvLen * length(screenSize);
float tDelta      = screenDelta / length(R.xy / (-P_view.z));  // perspective
currentZ         += R.z / length(R.xy) * tDelta;               // avancement Z
```

### Encodage/décodage normal octaédrique

```slang
float3 decodeOctahedral(float2 enc) {
    float3 n = float3(enc, 1.0 - abs(enc.x) - abs(enc.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * sign(n.xy);
    return normalize(n);
}
```

### Pondération roughness → mip couleur

```slang
// Surfaces très rugueuses → lire à un mip plus élevé de SceneLit (zone blurée)
// SceneLit doit avoir des mips générés (ajouter HiZBuild ou mipmapGenPass si absent)
float colorMip = roughness * 5.0;  // à calibrer
float3 color   = sceneLit.SampleLevel(linearSampler, hitUV, colorMip).rgb;
```

> **Prérequis** : `SceneLit` doit avoir ses mips générés. Si non, utiliser un pass de blur optionnel (R = cone blur) ou ignorer dans un premier temps (mip fixe = 0).

---

## 7. Ordre de priorité

```
T0.1 (SceneLitMipsPass multi-dispatch + shader avg)
    → Validation RenderDoc : mips de SceneLit corrects
    → T0.2 (SSR_History frame graph)
    → T0.3 (squelettes Rust SSR, compilation)
    → M1 (SSR main shader + pass)
    → Validation RenderDoc : SSR_Raw visible
    → M4 (DebugChannel::SsrResolved + shader + render_graph)
    → Validation GUI : canal SSR dans le debug viewer
    → M3 (composite sur SceneLit avec mips)
    → Validation visuelle : réflexions dans le rendu final
    → M2 (temporal accumulation)
    → M5 (GUI paramètres sliders)
    → Calibration artistique + tuning performances
    ··· (plus tard) ···
    → Migration SPD : voir spd.md
```

Le composite (M3) peut être prototypé avec `colorMip = 0` (pas de rugosité) avant même que SceneLit ait ses mips — utile pour valider le pipeline end-to-end. Les mips de T0.1 débloquent ensuite la qualité sur surfaces rugueuses.

---

## 8. Points de vigilance

| Risque                         | Mitigation                                                                         |
|-------------------------------|------------------------------------------------------------------------------------|
| Cycle de dépendance SceneLit  | SSR lit SceneLit courant (pas de SSR dans SSR) — acceptable pour 1er ordre         |
| Profondeur reverse-Z           | Toutes les comparaisons depth : higher = closer, vérifier sens de la comparaison   |
| Normales world vs view-space   | Décoder depuis GBuffer, puis `mul(float3x3(view), N_world)` avant `reflect()`      |
| HiZ MAX reduction              | Reduction MAX en reverse-Z = valeur la plus proche, correct pour intersections     |
| SceneLit sans mips             | Vérifier que DeferredResolve génère des mips ou ajouter un pass dédié              |
| Ghosting temporel              | Variance clipping obligatoire pour éviter les trainées sur objets en mouvement     |
| Coût GPU max_steps élevé       | Profiler avec 16/32/48 pas — le HiZ réduit fortement le coût pour scènes ouvertes  |
