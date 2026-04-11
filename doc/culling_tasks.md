# Plan d'implémentation : G-Buffer + Hi-Z
> Ancré sur l'état du code au 2026-04-11.  
> Chaque tâche est indépendante, testable, et modifie au maximum 2 fichiers.  
> Référence design : `doc/culling.md`.

---

## Conventions

- **Test de compilation** : `.\tools\i3-cargo.ps1 check` passe sans erreurs ni warnings.  
- **Test runtime** : l'application démarre, affiche la scène, aucune erreur Vulkan Validation Layer.  
- **"publier"** = appeler `builder.publish(name, value)` dans la closure `graph.declare(|builder| {...})`.  
- **"importer"** = appeler `builder.import_buffer(name, physical)` (pour les buffers physiques persistants détenus par une passe).  
- Les shaders Slang sont buildés hors-bande par l'asset pipeline (baker). La tâche crée le `.slang`, le build est supposé disponible en `.i3b`.

---

## M0 — Infrastructure (prérequis bloquants)

### T0.1 — N-Buffering de `TemporalRegistry` + support des images historiques

**Objectif :** La `TemporalRegistry` stocke actuellement 2 copies fixes (`[T; 2]`) de chaque ressource temporelle. Avec une swapchain à 3 images (Triple Buffering), la frame N peut lire un buffer encore écrit par la frame N-2 (même slot physique). Il faut généraliser à N slots, calé sur le nombre d'images de la swapchain.

#### Étape A — Refactoriser `TemporalRegistry`

**Fichier :** `crates/i3_gfx/src/graph/temporal.rs`

Remplacer les tableaux `[T; 2]` par des `Vec<T>` :
```rust
pub struct TemporalRegistry {
    pub current_frame: usize,
    pub capacity: usize,               // = swapchain image count
    pub(crate) buffers: HashMap<String, Vec<BackendBuffer>>,
    pub(crate) images:  HashMap<String, Vec<BackendImage>>,
}
```

`advance_frame` prend maintenant la capacité en paramètre :
```rust
pub fn advance_frame(&mut self, capacity: usize) {
    self.capacity = capacity;
    self.current_frame = (self.current_frame + 1) % capacity;
}
```

Mise à jour de `get_or_create_buffer` et `get_or_create_image` :
```rust
pub fn get_or_create_buffer<B: RenderBackendInternal>(
    &mut self, name: &str, desc: &BufferDesc, backend: &mut B,
) -> BackendBuffer {
    let cap = self.capacity.max(2);
    let entry = self.buffers.entry(name.to_string()).or_insert_with(|| {
        (0..cap).map(|_| backend.create_transient_buffer(desc)).collect()
    });
    entry[self.current_frame % entry.len()]
}

pub fn get_or_create_history_buffer<B: RenderBackendInternal>(
    &mut self, name: &str, desc: &BufferDesc, backend: &mut B,
) -> BackendBuffer {
    let cap = self.capacity.max(2);
    let entry = self.buffers.entry(name.to_string()).or_insert_with(|| {
        (0..cap).map(|_| backend.create_transient_buffer(desc)).collect()
    });
    let history_idx = (self.current_frame + entry.len() - 1) % entry.len();
    entry[history_idx]
}
```
Même logique pour `get_or_create_image` / `get_or_create_history_image`.

Adapter l'appel dans `render_graph.rs` : passer le nombre d'images swapchain réel à `advance_frame`. Le backend expose déjà `swapchain_image_count(window)` ou équivalent — vérifier l'API disponible.

**Test :** compilation passe. Le comportement fonctionnel est identique à 2 slots.

#### Étape B — Ajouter `declare_image_history` / `read_image_history` dans `PassBuilder`

**Fichier :** `crates/i3_gfx/src/graph/node.rs`

Le pattern pour les buffers existe déjà (lignes 237-257). Ajouter la symétrie exacte pour les images.

Dans `InternalPassBuilder` (fichier `pass.rs`), ajouter :
```rust
fn declare_image_history(&mut self, name: &str, desc: ImageDesc) -> ImageHandle;
fn read_image_history(&mut self, name: &str) -> ImageHandle;
```

Dans `impl<'a> InternalPassBuilder for PassRecorder<'a>` (`node.rs`), implémenter :
```rust
fn declare_image_history(&mut self, name: &str, desc: ImageDesc) -> ImageHandle {
    // Même pattern que declare_buffer_history :
    // SymbolLifetime::TemporalHistory, is_output: false
    self.declare_image_impl_with_lifetime(name, desc, SymbolLifetime::TemporalHistory, false)
}

fn read_image_history(&mut self, name: &str) -> ImageHandle {
    let history_name = format!("{}_History", name);
    // Crée un handle virtuel pointant vers le slot N-1 de la TemporalRegistry
    self.declare_image_impl_with_lifetime(
        &history_name,
        ImageDesc::default(),   // taille résolue par le compilateur via la TemporalRegistry
        SymbolLifetime::TemporalHistory,
        false,
    )
}
```

Adapter `declare_image_impl` pour accepter un `SymbolLifetime` (actuellement toujours `Transient`).

Adapter le compilateur du graphe (`compiler.rs`) pour résoudre les images `TemporalHistory` via la `TemporalRegistry`, de la même façon que pour les buffers.

**Test :** compilation passe. Pas de test runtime requis à ce stade.

---

### T0.2 — Ajouter `prev_view_projection` dans `CommonData`

**Fichier :** `crates/i3_renderer/src/render_graph.rs`

`CommonData` est défini ligne 22. Ajouter un champ :
```rust
pub prev_view_projection: nalgebra_glm::Mat4,
```

Dans `DefaultRenderGraph`, ajouter un champ pour stocker la VP de la frame précédente :
```rust
pub prev_view_projection: nalgebra_glm::Mat4,
```
Initialiser à `Mat4::identity()` dans `DefaultRenderGraph::new()`.

Dans `render()` (avant de construire `CommonData`) :
```rust
let prev_view_projection = self.prev_view_projection;
// ... construire CommonData avec prev_view_projection
// À la fin (après execute) :
self.prev_view_projection = view_projection;
```

Propager dans la construction de `CommonData` dans `declare()` également.

**Test :** compilation passe. La valeur est `identity()` à la frame 1, et `view_proj_(N-1)` aux frames suivantes.

---

---

## M1 — HiZBuildPass

### T1.1 — Créer `hiz_build.slang`

**Fichier (nouveau) :** `crates/i3_renderer/assets/shaders/hiz_build.slang`

Ce shader fait deux choses en un seul dispatch :
1. **Mip 0** : lit `DepthBuffer` (D32_FLOAT sampler2D), écrit dans `HiZPyramid` mip0 (R32_SFLOAT storage image).
2. **Mips 1..N** : réduit récursivement par blocs 2×2, opérateur MAX (reverse-Z : MAX = surface la plus proche).

```slang
// Bindings set 0 :
//   binding 0 : sampler2D  depth_input   (D32_SFLOAT, layout ShaderReadOnly)
//   binding 1 : image2D[]  hiz_mips      (R32_SFLOAT, layout General) — tous les mips

[vk::binding(0, 0)] Sampler2D depth_input;
[vk::binding(1, 0)] RWTexture2D<float> hiz_mips[12]; // max 12 mips pour 4096×4096

struct PushConstants {
    uint2  src_size;      // dimensions du DepthBuffer (avant next_power_of_two)
    uint2  hiz_size;      // dimensions du HiZPyramid mip0 (next_power_of_two)
    uint   mip_count;
};
[[vk::push_constant]] PushConstants push;

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // --- Mip 0 : copie depth → hiz_mips[0] ---
    uint2 coord = tid.xy;
    if (any(coord >= push.hiz_size)) return;

    // Clamp vers le bord du depth buffer (résolution peut être non-PoT)
    uint2 src_coord = min(coord, push.src_size - 1u);
    float2 uv = (float2(src_coord) + 0.5) / float2(push.src_size);
    float d = depth_input.SampleLevel(uv, 0).r;
    hiz_mips[0][coord] = d;

    // Synchronisation intra-wavefront non possible ici pour les mips suivants.
    // Ce shader ne fait que le mip 0. Les mips 1..N sont traités par un second dispatch
    // ou par des dispatches séparés (voir T1.3).
}
```

> **Note** : Le SPD complet (tous les mips en 1 dispatch) requiert `globallycoherent` et des atomics complexes. Pour la première implémentation, utiliser **N dispatches séparés** — 1 par niveau de mip. Performance acceptable, implémentation triviale et sans bug.

Shader du mip K→K+1 (même fichier, entry point séparé) :
```slang
[numthreads(8, 8, 1)]
void reduce_mip(uint3 tid : SV_DispatchThreadID,
                [[vk::push_constant]] PushConstants push,
                uint src_mip_push : /* separate push offset */) {
    // Push constants étendus : ajouter `uint src_mip` dans PushConstants
    uint2 dst = tid.xy;
    uint2 src_size = max(uint2(push.hiz_size >> push.src_mip), uint2(1, 1));
    uint2 dst_size = max(src_size >> 1u, uint2(1, 1));
    if (any(dst >= dst_size)) return;

    uint2 s = dst * 2u;
    float a = hiz_mips[push.src_mip][min(s + uint2(0,0), src_size-1u)];
    float b = hiz_mips[push.src_mip][min(s + uint2(1,0), src_size-1u)];
    float c = hiz_mips[push.src_mip][min(s + uint2(0,1), src_size-1u)];
    float e = hiz_mips[push.src_mip][min(s + uint2(1,1), src_size-1u)];
    // Reverse-Z : MAX = surface la plus proche de la caméra
    hiz_mips[push.src_mip + 1u][dst] = max(max(a, b), max(c, e));
}
```

**PushConstants complet :**
```slang
struct PushConstants {
    uint2 src_size;    // dims du DepthBuffer (original, non-PoT)
    uint2 hiz_size;    // dims du mip0 HiZ (next_power_of_two)
    uint  mip_count;   // nombre total de mips
    uint  src_mip;     // mip source pour la passe de réduction (0 = blit depth)
};
```

Deux entry points dans le même fichier : `blit_depth` et `reduce_mip`.

---

### T1.2 — Créer `passes/hiz_build.rs`

**Fichier (nouveau) :** `crates/i3_renderer/src/passes/hiz_build.rs`

```rust
use i3_gfx::prelude::*;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy)]
struct HiZBuildPushConstants {
    src_size:  [u32; 2],
    hiz_size:  [u32; 2],
    mip_count: u32,
    src_mip:   u32,
}

pub struct HiZBuildPass {
    depth_buffer: ImageHandle,
    hiz_pyramid:  ImageHandle,    // image courante (frame N), R32_SFLOAT
    mip_count:    u32,
    src_width:    u32,
    src_height:   u32,
    hiz_width:    u32,
    hiz_height:   u32,

    pipeline_blit:   Option<BackendPipeline>,  // entry: blit_depth
    pipeline_reduce: Option<BackendPipeline>,  // entry: reduce_mip
}

impl HiZBuildPass {
    pub fn new() -> Self {
        Self {
            depth_buffer: ImageHandle::INVALID,
            hiz_pyramid:  ImageHandle::INVALID,
            mip_count:    0,
            src_width:    0,
            src_height:   0,
            hiz_width:    0,
            hiz_height:   0,
            pipeline_blit:   None,
            pipeline_reduce: None,
        }
    }
}

impl RenderPass for HiZBuildPass {
    fn name(&self) -> &str { "HiZBuild" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        // Les deux entry points sont dans le même asset "hiz_build"
        // Le baker produit deux pipelines séparés via entry point annotation.
        // Si le baker ne supporte qu'un entry point par asset, créer "hiz_build_blit" et "hiz_build_reduce".
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("hiz_build_blit").wait_loaded() {
            self.pipeline_blit = Some(backend.create_compute_pipeline_from_baked(
                &asset.reflection_data, &asset.bytecode,
            ));
        }
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("hiz_build_reduce").wait_loaded() {
            self.pipeline_reduce = Some(backend.create_compute_pipeline_from_baked(
                &asset.reflection_data, &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        self.src_width  = common.screen_width;
        self.src_height = common.screen_height;
        self.hiz_width  = common.screen_width.next_power_of_two();
        self.hiz_height = common.screen_height.next_power_of_two();
        self.mip_count  = (self.hiz_width.max(self.hiz_height) as f32).log2() as u32 + 1;

        let hiz_desc = ImageDesc {
            width:        self.hiz_width,
            height:       self.hiz_height,
            depth:        1,
            format:       Format::R32_SFLOAT,
            mip_levels:   self.mip_count,
            array_layers: 1,
            usage:        ImageUsageFlags::STORAGE | ImageUsageFlags::SAMPLED,
            view_type:    ImageViewType::Type2D,
            swizzle:      Default::default(),
            clear_value:  None,
        };

        self.depth_buffer = builder.resolve_image("DepthBuffer");
        // Déclare HiZPyramid comme image temporelle : le frame graph alloue N slots
        // (N = swapchain image count) et gère le ping-pong automatiquement.
        self.hiz_pyramid  = builder.declare_image_history("HiZPyramid", hiz_desc);

        builder.read_image(self.depth_buffer, ResourceUsage::SHADER_READ);
        builder.write_image(self.hiz_pyramid, ResourceUsage::SHADER_WRITE);

        builder.descriptor_set(0, |d| {
            d.sampled_image(self.depth_buffer, DescriptorImageLayout::DepthStencilReadOnlyOptimal)
             .storage_image(self.hiz_pyramid, DescriptorImageLayout::General);
        });
    }

    fn execute(&self, ctx: &mut dyn PassContext, _frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        let (Some(blit), Some(reduce)) = (self.pipeline_blit, self.pipeline_reduce) else {
            tracing::error!("HiZBuildPass: pipelines not initialized");
            return;
        };

        // Passe 0 : blit depth → hiz mip0
        ctx.bind_pipeline_raw(blit);
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &HiZBuildPushConstants {
            src_size:  [self.src_width, self.src_height],
            hiz_size:  [self.hiz_width, self.hiz_height],
            mip_count: self.mip_count,
            src_mip:   0,
        });
        let wg_x = (self.hiz_width  + 7) / 8;
        let wg_y = (self.hiz_height + 7) / 8;
        ctx.dispatch(wg_x, wg_y, 1);

        // Passes 1..mip_count-1 : réduction 2×2
        ctx.bind_pipeline_raw(reduce);
        for mip in 0..(self.mip_count - 1) {
            let mip_w = (self.hiz_width  >> mip).max(1);
            let mip_h = (self.hiz_height >> mip).max(1);
            let dst_w = (mip_w >> 1).max(1);
            let dst_h = (mip_h >> 1).max(1);
            ctx.push_constant_data(ShaderStageFlags::Compute, 0, &HiZBuildPushConstants {
                src_size:  [self.src_width, self.src_height],
                hiz_size:  [self.hiz_width, self.hiz_height],
                mip_count: self.mip_count,
                src_mip:   mip,
            });
            ctx.dispatch((dst_w + 7) / 8, (dst_h + 7) / 8, 1);
        }
    }
}
```

> **Problème connu** : les dispatches de réduction séquentiels dans un seul `execute()` n'ont pas de barrières mémoire entre eux. Vulkan n'en garantit pas l'ordre. **Solution pour la première itération** : chaque niveau de mip est une passe séparée dans le graph (ou utiliser `ctx.pipeline_barrier` si disponible dans `PassContext`). Si `pipeline_barrier` n'existe pas, créer `HiZBuildPass` comme un groupe avec un enfant par mip.  
> **Alternative acceptable** : tolérer les artefacts de race condition sur les mips supérieurs et itérer après validation du mip 0.

---

### T1.3 — Déclarer `HiZBuildPass` dans `mod.rs`

**Fichier :** `crates/i3_renderer/src/passes/mod.rs`

Ajouter :
```rust
pub mod hiz_build;
```

**Test :** compilation passe.

---

### T1.4 — Ajouter `hiz_build_pass` dans `DefaultRenderGraph`

**Fichier :** `crates/i3_renderer/src/render_graph.rs`

1. Ajouter l'import en tête :
```rust
use crate::passes::hiz_build::HiZBuildPass;
```

2. Dans `DefaultRenderGraph` struct, ajouter :
```rust
pub hiz_build_pass: HiZBuildPass,
```

3. Dans `new()`, initialiser :
```rust
let hiz_build_pass = HiZBuildPass::new();
```
Et inclure dans le struct literal.

4. Dans `init()`, ajouter :
```rust
self.graph.init_pass_direct(&mut self.hiz_build_pass, backend);
```

**Test :** compilation passe.

---

### T1.5 — Câbler `HiZBuildPass` dans `declare()`

**Fichier :** `crates/i3_renderer/src/render_graph.rs`

Dans la closure passée à `graph.declare(|builder| {...})` :

**Après `builder.add_pass(&mut self.gbuffer_pass)` et `builder.add_pass(&mut self.sky_pass)`**, ajouter simplement :
```rust
// HiZ Build : lit DepthBuffer N, écrit HiZPyramid N (slot courant géré par TemporalRegistry)
builder.add_pass(&mut self.hiz_build_pass);
```

Aucune gestion manuelle d'images. `HiZBuildPass::declare()` appelle `declare_image_history("HiZPyramid", desc)` directement — le compilateur du graphe résout le bon slot physique via la `TemporalRegistry`.

L'ordre dans la closure doit être :
```
sync_group
blas_update_pass
draw_call_gen_pass        ← inchangé (Niveau 0, sera remplacé en M2)
tlas_rebuild_pass
clustering_group
gbuffer_pass              ← écrit DepthBuffer N
sky_pass
hiz_build_pass            ← NOUVEAU : lit DepthBuffer N, écrit HiZPyramid N
declare_buffer_history("ExposureBuffer", ...)
... deferred_resolve, post_process ...
```

**Test :**
1. Compilation passe.
2. Runtime : aucune validation error.
3. Activer `DebugChannel::DepthBuffer` ou équivalent et vérifier que la scène s'affiche toujours.
4. (Optionnel) Visualiser `HiZPyramid` mip0 via `debug_viz` — doit ressembler au depth buffer en niveaux de gris.

---

## M2 — OcclusionCullPass (remplace DrawCallGenPass)

### T2.1 — Créer `occlusion_cull.slang`

**Fichier (nouveau) :** `crates/i3_renderer/assets/shaders/occlusion_cull.slang`

```slang
// Binding set 0 :
//   0 : StructuredBuffer<GpuInstanceData>   instances
//   1 : StructuredBuffer<GpuMeshDescriptor> meshes
//   2 : RWStructuredBuffer<DrawIndirectCmd>  draw_calls
//   3 : RWStructuredBuffer<uint>             draw_count  (uint à l'offset 0)
//   4 : Sampler2D                            hiz_pyramid (history N-1)

// Types doivent correspondre EXACTEMENT aux structs Rust dans scene.rs
struct GpuInstanceData {
    float4x4 world_transform;
    float4x4 prev_transform;
    uint     mesh_idx;
    uint     material_id;
    uint     flags;
    uint     _pad;
    float3   world_aabb_min;
    float    _pad2;
    float3   world_aabb_max;
    float    _pad3;
};

struct GpuMeshDescriptor {
    uint  vertex_buffer_id;
    uint  index_buffer_id;
    uint  index_count;
    uint  vertex_stride;
    uint  index_offset;
    int   vertex_offset;
    float3 aabb_min;
    float  _pad0;
    float3 aabb_max;
    float  _pad1;
};

struct DrawIndirectCmd {
    uint index_count;
    uint instance_count;
    uint first_index;
    int  vertex_offset;
    uint first_instance;
};

[vk::binding(0, 0)] StructuredBuffer<GpuInstanceData>   instances;
[vk::binding(1, 0)] StructuredBuffer<GpuMeshDescriptor> meshes;
[vk::binding(2, 0)] RWStructuredBuffer<DrawIndirectCmd>  draw_calls;
[vk::binding(3, 0)] RWStructuredBuffer<uint>             draw_count;
[vk::binding(4, 0)] Sampler2D                            hiz_pyramid;

struct PushConstants {
    float4x4 view_proj;       // frame N (frustum cull)
    float4x4 prev_view_proj;  // frame N-1 (Hi-Z reprojection)
    uint     instance_count;
    uint     max_draws;
    uint     hiz_mip_count;
    float    screen_w;
    float    screen_h;
};
[[vk::push_constant]] PushConstants push;

// Teste si l'AABB [aabb_min, aabb_max] est dans le frustum de view_proj.
// Retourne true si VISIBLE, false si culled.
bool frustum_cull(float3 aabb_min, float3 aabb_max, float4x4 vp) {
    // Projeter les 8 coins, tester contre les 6 plans
    float3 corners[8] = {
        float3(aabb_min.x, aabb_min.y, aabb_min.z),
        float3(aabb_max.x, aabb_min.y, aabb_min.z),
        float3(aabb_min.x, aabb_max.y, aabb_min.z),
        float3(aabb_max.x, aabb_max.y, aabb_min.z),
        float3(aabb_min.x, aabb_min.y, aabb_max.z),
        float3(aabb_max.x, aabb_min.y, aabb_max.z),
        float3(aabb_min.x, aabb_max.y, aabb_max.z),
        float3(aabb_max.x, aabb_max.y, aabb_max.z),
    };
    // Pour chaque plan (±x, ±y, ±z en clip space) :
    // si TOUS les coins sont du mauvais côté → cull
    for (int p = 0; p < 6; p++) {
        bool all_outside = true;
        for (int c = 0; c < 8; c++) {
            float4 clip = mul(vp, float4(corners[c], 1.0));
            float  w    = clip.w;
            float  v;
            if      (p == 0) v =  clip.x;
            else if (p == 1) v = -clip.x;
            else if (p == 2) v =  clip.y;
            else if (p == 3) v = -clip.y;
            else if (p == 4) v =  clip.z;          // near (reverse-Z: z > 0)
            else             v = w - clip.z;        // far  (reverse-Z: z < w)
            if (v > -w) { all_outside = false; break; }
        }
        if (all_outside) return false;
    }
    return true;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= push.instance_count) return;

    GpuInstanceData   inst = instances[i];
    GpuMeshDescriptor mesh = meshes[inst.mesh_idx];

    float3 aabb_min = inst.world_aabb_min;
    float3 aabb_max = inst.world_aabb_max;

    // --- 1. Frustum cull (frame N) ---
    if (!frustum_cull(aabb_min, aabb_max, push.view_proj)) return;

    // --- 2. Hi-Z occlusion test (frame N-1, reverse-Z) ---
    // Projeter les 8 coins avec prev_view_proj, trouver le footprint 2D
    // et la depth max (= coin le plus proche caméra en reverse-Z)
    float2 ndc_min = float2( 1.0,  1.0);
    float2 ndc_max = float2(-1.0, -1.0);
    float  ndc_z_max = 0.0;  // depth max en NDC (reverse-Z : plus haut = plus proche)

    float3 corners[8] = {
        float3(aabb_min.x, aabb_min.y, aabb_min.z),
        float3(aabb_max.x, aabb_min.y, aabb_min.z),
        float3(aabb_min.x, aabb_max.y, aabb_min.z),
        float3(aabb_max.x, aabb_max.y, aabb_min.z),
        float3(aabb_min.x, aabb_min.y, aabb_max.z),
        float3(aabb_max.x, aabb_min.y, aabb_max.z),
        float3(aabb_min.x, aabb_max.y, aabb_max.z),
        float3(aabb_max.x, aabb_max.y, aabb_max.z),
    };
    bool behind_camera = true;
    for (int c = 0; c < 8; c++) {
        float4 clip = mul(push.prev_view_proj, float4(corners[c], 1.0));
        if (clip.w <= 0.0) continue;
        behind_camera = false;
        float3 ndc = clip.xyz / clip.w;
        ndc_min   = min(ndc_min, ndc.xy);
        ndc_max   = max(ndc_max, ndc.xy);
        ndc_z_max = max(ndc_z_max, ndc.z);   // MAX en reverse-Z = coin le plus proche
    }
    if (behind_camera) return;  // entièrement derrière la caméra

    // Clipper le footprint NDC à [-1,1]
    ndc_min = clamp(ndc_min, -1.0, 1.0);
    ndc_max = clamp(ndc_max, -1.0, 1.0);
    if (any(ndc_min >= ndc_max)) return;  // footprint nul

    // Taille screen-space → niveau de mip
    float2 uv_size = (ndc_max - ndc_min) * 0.5 * float2(push.screen_w, push.screen_h);
    float  mip     = ceil(log2(max(uv_size.x, uv_size.y)));
    mip = clamp(mip, 0.0, float(push.hiz_mip_count - 1));

    // Sample Hi-Z N-1 (history)
    float2 uv_center = (ndc_min + ndc_max) * 0.5 * 0.5 + 0.5;
    float  hiz_max   = hiz_pyramid.SampleLevel(uv_center, mip).r;

    // Test reverse-Z : ndc_z_max < hiz_max → objet derrière l'occludeur → cull
    if (ndc_z_max < hiz_max) return;

    // --- 3. Émettre le draw call ---
    uint slot;
    InterlockedAdd(draw_count[0], 1u, slot);
    if (slot >= push.max_draws) return;

    draw_calls[slot].index_count    = mesh.index_count;
    draw_calls[slot].instance_count = 1;
    draw_calls[slot].first_index    = mesh.index_offset;
    draw_calls[slot].vertex_offset  = mesh.vertex_offset;
    draw_calls[slot].first_instance = slot;  // lu par VS via VisibleInstanceBuffer
}
```

---

### T2.2 — Créer `passes/occlusion_cull.rs`

**Fichier (nouveau) :** `crates/i3_renderer/src/passes/occlusion_cull.rs`

Copier la structure de `DrawCallGenPass` (`passes/cull.rs`) et modifier :

```rust
use crate::constants::{DRAW_INDIRECT_CMD_SIZE, MAX_INSTANCES};
use i3_gfx::prelude::*;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy)]
struct OcclusionCullPushConstants {
    view_proj:      nalgebra_glm::Mat4,
    prev_view_proj: nalgebra_glm::Mat4,
    instance_count: u32,
    max_draws:      u32,
    hiz_mip_count:  u32,
    screen_w:       f32,
    screen_h:       f32,
}

// --- Passe compute interne ---
struct OcclusionCullComputePass {
    instance_buffer:        BufferHandle,
    mesh_descriptor_buffer: BufferHandle,
    draw_call_buffer:       BufferHandle,
    draw_count_buffer:      BufferHandle,
    hiz_pyramid_history:    ImageHandle,   // Hi-Z N-1
    instance_count:         u32,
    mip_count:              u32,
    screen_w:               f32,
    screen_h:               f32,
    pipeline:               Option<BackendPipeline>,
}

impl OcclusionCullComputePass {
    fn new() -> Self {
        Self {
            instance_buffer:        BufferHandle::INVALID,
            mesh_descriptor_buffer: BufferHandle::INVALID,
            draw_call_buffer:       BufferHandle::INVALID,
            draw_count_buffer:      BufferHandle::INVALID,
            hiz_pyramid_history:    ImageHandle::INVALID,
            instance_count:         0,
            mip_count:              0,
            screen_w:               1.0,
            screen_h:               1.0,
            pipeline:               None,
        }
    }
}

impl RenderPass for OcclusionCullComputePass {
    fn name(&self) -> &str { "OcclusionCullCompute" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("occlusion_cull").wait_loaded() {
            self.pipeline = Some(backend.create_compute_pipeline_from_baked(
                &asset.reflection_data, &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        self.instance_count = builder
            .try_consume::<Vec<crate::scene::GpuInstanceData>>("SceneInstances")
            .map(|v| v.len() as u32)
            .unwrap_or(0);
        self.screen_w   = common.screen_width as f32;
        self.screen_h   = common.screen_height as f32;
        self.mip_count  = common.screen_width.max(common.screen_height)
            .next_power_of_two()
            .max(1)
            .leading_zeros()
            .wrapping_sub(u32::BITS - 1 - 32u32.leading_zeros())
            + 1;
        // Plus simple : stocker mip_count dans CommonData (futur refactor), pour l'instant :
        self.mip_count = (common.screen_width.max(common.screen_height)
            .next_power_of_two() as f32).log2() as u32 + 1;

        self.instance_buffer        = builder.resolve_buffer("InstanceBuffer");
        self.mesh_descriptor_buffer = builder.resolve_buffer("MeshDescriptorBuffer");
        self.draw_call_buffer       = builder.resolve_buffer("DrawCallBuffer");
        self.draw_count_buffer      = builder.resolve_buffer("DrawCountBuffer");
        // Lit le slot N-1 de la pyramide HiZ — géré automatiquement par le frame graph
        self.hiz_pyramid_history    = builder.read_image_history("HiZPyramid");

        builder.read_buffer(self.instance_buffer,        ResourceUsage::SHADER_READ);
        builder.read_buffer(self.mesh_descriptor_buffer, ResourceUsage::SHADER_READ);
        builder.write_buffer(self.draw_call_buffer,      ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.draw_count_buffer,     ResourceUsage::SHADER_WRITE);
        builder.read_image(self.hiz_pyramid_history,     ResourceUsage::SHADER_READ);

        builder.descriptor_set(0, |d| {
            d.storage_buffer(self.instance_buffer)
             .storage_buffer(self.mesh_descriptor_buffer)
             .storage_buffer(self.draw_call_buffer)
             .storage_buffer(self.draw_count_buffer)
             .sampled_image(self.hiz_pyramid_history, DescriptorImageLayout::ShaderReadOnlyOptimal);
        });
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        if self.instance_count == 0 { return; }
        let Some(pipeline) = self.pipeline else {
            tracing::error!("OcclusionCullComputePass: pipeline not initialized");
            return;
        };
        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        ctx.bind_pipeline_raw(pipeline);
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &OcclusionCullPushConstants {
            view_proj:      common.view_projection,
            prev_view_proj: common.prev_view_projection,
            instance_count: self.instance_count,
            max_draws:      MAX_INSTANCES as u32,
            hiz_mip_count:  self.mip_count,
            screen_w:       self.screen_w,
            screen_h:       self.screen_h,
        });
        let groups = (self.instance_count + 63) / 64;
        ctx.dispatch(groups, 1, 1);
    }
}

// --- Groupe parent (même pattern que DrawCallGenPass) ---
pub struct OcclusionCullPass {
    draw_call_buffer_physical:  BackendBuffer,
    draw_count_buffer_physical: BackendBuffer,
    draw_call_buffer:           BufferHandle,
    draw_count_buffer:          BufferHandle,
    compute:                    OcclusionCullComputePass,
}

impl OcclusionCullPass {
    pub fn new() -> Self {
        Self {
            draw_call_buffer_physical:  BackendBuffer::INVALID,
            draw_count_buffer_physical: BackendBuffer::INVALID,
            draw_call_buffer:           BufferHandle::INVALID,
            draw_count_buffer:          BufferHandle::INVALID,
            compute:                    OcclusionCullComputePass::new(),
        }
    }
}

impl RenderPass for OcclusionCullPass {
    fn name(&self) -> &str { "OcclusionCull" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        self.draw_call_buffer_physical = backend.create_buffer(&BufferDesc {
            size:   MAX_INSTANCES * DRAW_INDIRECT_CMD_SIZE,
            usage:  BufferUsageFlags::STORAGE_BUFFER
                  | BufferUsageFlags::INDIRECT_BUFFER
                  | BufferUsageFlags::TRANSFER_DST,
            memory: MemoryType::GpuOnly,
        });
        self.draw_count_buffer_physical = backend.create_buffer(&BufferDesc {
            size:   16,  // u32 + padding
            usage:  BufferUsageFlags::STORAGE_BUFFER
                  | BufferUsageFlags::INDIRECT_BUFFER
                  | BufferUsageFlags::TRANSFER_DST,
            memory: MemoryType::GpuOnly,
        });
        #[cfg(debug_assertions)]
        {
            backend.set_buffer_name(self.draw_call_buffer_physical,  "DrawCallBuffer");
            backend.set_buffer_name(self.draw_count_buffer_physical, "DrawCountBuffer");
        }
        self.compute.init(backend, globals);
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        self.draw_call_buffer =
            builder.import_buffer("DrawCallBuffer",  self.draw_call_buffer_physical);
        self.draw_count_buffer =
            builder.import_buffer("DrawCountBuffer", self.draw_count_buffer_physical);

        // Enfant 1 : clear draw count
        builder.add_owned_pass(crate::render_graph::ClearBufferPass {
            name:   "ClearDrawCount".to_string(),
            buffer: self.draw_count_buffer,
        });

        // Enfant 2 : compute occlusion cull
        builder.add_pass(&mut self.compute);
    }
    // Pas d'execute() — groupe pur
}
```

---

### T2.3 — Déclarer `occlusion_cull` dans `mod.rs`

**Fichier :** `crates/i3_renderer/src/passes/mod.rs`

```rust
pub mod occlusion_cull;
```

---

### T2.4 — Remplacer `DrawCallGenPass` par `OcclusionCullPass` dans `DefaultRenderGraph`

**Fichier :** `crates/i3_renderer/src/render_graph.rs`

1. Remplacer l'import :
```rust
// Avant :
use crate::passes::cull::DrawCallGenPass;
// Après :
use crate::passes::occlusion_cull::OcclusionCullPass;
```

2. Dans `DefaultRenderGraph` struct :
```rust
// Avant :
pub draw_call_gen_pass: DrawCallGenPass,
// Après :
pub occlusion_cull_pass: OcclusionCullPass,
```

3. Dans `new()` :
```rust
// Avant :
let draw_call_gen_pass = DrawCallGenPass::new();
// Après :
let occlusion_cull_pass = OcclusionCullPass::new();
```

4. Dans `init()` :
```rust
// Avant :
self.graph.init_pass_direct(&mut self.draw_call_gen_pass, backend);
// Après :
self.graph.init_pass_direct(&mut self.occlusion_cull_pass, backend);
```

5. Dans `declare()` (closure `graph.declare`) :
```rust
// Avant :
builder.add_pass(&mut self.draw_call_gen_pass);
// Après :
builder.add_pass(&mut self.occlusion_cull_pass);
```

**Test :**
1. Compilation passe (plus de référence à `DrawCallGenPass` ni `cull.rs`).
2. Runtime : tous les objets visibles sont toujours rendus.
3. Frame 1 : `prev_view_projection` est `identity()` → Hi-Z history est invalide (image non écrite) → résultat du test Hi-Z indéfini → **utiliser `hiz_mip_count = 0` comme signal pour bypasser le test Hi-Z à la frame 1.**

> **Ajout nécessaire dans le shader :** si `push.hiz_mip_count == 0`, sauter le test Hi-Z et passer directement à l'émission du draw.

---

### T2.5 — (Optionnel) Supprimer `cull.rs` si plus utilisé

**Fichier :** `crates/i3_renderer/src/passes/cull.rs` et `mod.rs`

Vérifier qu'aucun autre fichier n'importe `cull`. Si c'est le cas, supprimer de `mod.rs` et effacer le fichier.

**Test :** compilation passe.

---

## M3 — LightOcclusionCullPass

### T3.1 — Créer les buffers `VisibleLightBuffer` et `VisibleLightCountBuffer`

**Fichier :** `crates/i3_renderer/src/render_graph.rs`

Dans `DefaultRenderGraph` struct, ajouter :
```rust
pub visible_light_buffer:       i3_gfx::graph::backend::BackendBuffer,
pub visible_light_count_buffer: i3_gfx::graph::backend::BackendBuffer,
```

Dans `new()`, allouer :
```rust
let visible_light_buffer = _backend.create_buffer(&BufferDesc {
    // MAX_LIGHTS indices u32
    size:   crate::constants::MAX_LIGHTS * std::mem::size_of::<u32>() as u64,
    usage:  BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
    memory: MemoryType::GpuOnly,
});
let visible_light_count_buffer = _backend.create_buffer(&BufferDesc {
    size:   16,  // u32 + padding
    usage:  BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST,
    memory: MemoryType::GpuOnly,
});
#[cfg(debug_assertions)]
{
    _backend.set_buffer_name(visible_light_buffer,       "VisibleLightBuffer");
    _backend.set_buffer_name(visible_light_count_buffer, "VisibleLightCountBuffer");
}
```
Inclure dans le struct literal.

Dans `declare()` (closure), importer :
```rust
builder.import_buffer("VisibleLightBuffer",       self.visible_light_buffer);
builder.import_buffer("VisibleLightCountBuffer",  self.visible_light_count_buffer);
```

**Test :** compilation passe.

---

### T3.2 — Créer `light_occlusion_cull.slang`

**Fichier (nouveau) :** `crates/i3_renderer/assets/shaders/light_occlusion_cull.slang`

```slang
// Binding set 0 :
//   0 : StructuredBuffer<GpuLightData>  lights
//   1 : RWStructuredBuffer<uint>        visible_light_ids
//   2 : RWStructuredBuffer<uint>        visible_light_count  (u32 à offset 0)
//   3 : Sampler2D                       hiz_pyramid  (Hi-Z N, frame courante)

// GpuLightData doit correspondre à la struct Rust GpuLightData dans render_graph.rs
struct GpuLightData {
    float3 position;
    float  radius;        // rayon d'influence (0 = directionnel)
    float3 color;
    float  intensity;
    float3 direction;
    uint   light_type;    // 0=point, 1=directionnel, 2=spot
};

[vk::binding(0, 0)] StructuredBuffer<GpuLightData> lights;
[vk::binding(1, 0)] RWStructuredBuffer<uint>        visible_light_ids;
[vk::binding(2, 0)] RWStructuredBuffer<uint>        visible_light_count;
[vk::binding(3, 0)] Sampler2D                       hiz_pyramid;

struct PushConstants {
    float4x4 view_proj;
    uint     light_count;
    uint     hiz_mip_count;
    float    screen_w;
    float    screen_h;
};
[[vk::push_constant]] PushConstants push;

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.light_count) return;

    GpuLightData light = lights[idx];

    // Les lumières directionnelles (radius == 0) ne sont jamais occultées :
    // elles éclairent toute la scène.
    if (light.light_type == 1u || light.radius <= 0.0) {
        uint slot;
        InterlockedAdd(visible_light_count[0], 1u, slot);
        if (slot < push.light_count) visible_light_ids[slot] = idx;
        return;
    }

    // Projeter le centre de la sphère en NDC
    float4 clip = mul(push.view_proj, float4(light.position, 1.0));
    if (clip.w <= 0.0) return;  // derrière la caméra

    float3 ndc    = clip.xyz / clip.w;
    float2 uv_ctr = ndc.xy * 0.5 + 0.5;

    // Approximation du rayon de la sphère en NDC
    float proj_radius = light.radius / clip.w;

    // Mip correspondant au footprint screen-space de la sphère
    float footprint_px = proj_radius * max(push.screen_w, push.screen_h);
    float mip = clamp(ceil(log2(max(footprint_px, 1.0))),
                      0.0, float(push.hiz_mip_count - 1));

    // Profondeur NDC du front-face de la sphère (reverse-Z : plus haut = plus proche)
    float light_front_depth = saturate(ndc.z + proj_radius);

    // Sample Hi-Z N (même frame, produit par HiZBuildPass)
    float hiz_max = hiz_pyramid.SampleLevel(uv_ctr, mip).r;

    // Test occlusion reverse-Z :
    // light_front_depth < hiz_max → la sphère entière est plus loin que l'occludeur → cull
    if (light_front_depth < hiz_max) return;

    // Lumière visible : l'ajouter à la liste
    uint slot;
    InterlockedAdd(visible_light_count[0], 1u, slot);
    if (slot < push.light_count) visible_light_ids[slot] = idx;
}
```

---

### T3.3 — Créer `passes/light_occlusion_cull.rs`

**Fichier (nouveau) :** `crates/i3_renderer/src/passes/light_occlusion_cull.rs`

```rust
use i3_gfx::prelude::*;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy)]
struct LightOcclusionCullPushConstants {
    view_proj:      nalgebra_glm::Mat4,
    light_count:    u32,
    hiz_mip_count:  u32,
    screen_w:       f32,
    screen_h:       f32,
}

pub struct LightOcclusionCullPass {
    lights:                  BufferHandle,
    visible_light_ids:       BufferHandle,
    visible_light_count:     BufferHandle,
    hiz_pyramid:             ImageHandle,    // Hi-Z N (frame courante)
    light_count:             u32,
    mip_count:               u32,
    screen_w:                f32,
    screen_h:                f32,
    pipeline:                Option<BackendPipeline>,
}

impl LightOcclusionCullPass {
    pub fn new() -> Self {
        Self {
            lights:              BufferHandle::INVALID,
            visible_light_ids:   BufferHandle::INVALID,
            visible_light_count: BufferHandle::INVALID,
            hiz_pyramid:         ImageHandle::INVALID,
            light_count:         0,
            mip_count:           0,
            screen_w:            1.0,
            screen_h:            1.0,
            pipeline:            None,
        }
    }
}

impl RenderPass for LightOcclusionCullPass {
    fn name(&self) -> &str { "LightOcclusionCull" }

    fn init(&mut self, backend: &mut dyn RenderBackend, globals: &mut PassBuilder) {
        let loader = globals.consume::<Arc<i3_io::asset::AssetLoader>>("AssetLoader");
        if let Ok(asset) = loader.load::<i3_io::pipeline_asset::PipelineAsset>("light_occlusion_cull").wait_loaded() {
            self.pipeline = Some(backend.create_compute_pipeline_from_baked(
                &asset.reflection_data, &asset.bytecode,
            ));
        }
    }

    fn declare(&mut self, builder: &mut PassBuilder) {
        let common = *builder.consume::<crate::render_graph::CommonData>("Common");
        self.light_count = common.light_count;
        self.screen_w    = common.screen_width as f32;
        self.screen_h    = common.screen_height as f32;
        self.mip_count   = (common.screen_width.max(common.screen_height)
            .next_power_of_two() as f32).log2() as u32 + 1;

        self.lights              = builder.resolve_buffer("LightBuffer");
        self.visible_light_ids   = builder.resolve_buffer("VisibleLightBuffer");
        self.visible_light_count = builder.resolve_buffer("VisibleLightCountBuffer");
        self.hiz_pyramid         = builder.resolve_image("HiZPyramid");

        builder.read_buffer(self.lights,              ResourceUsage::SHADER_READ);
        builder.write_buffer(self.visible_light_ids,  ResourceUsage::SHADER_WRITE);
        builder.write_buffer(self.visible_light_count,ResourceUsage::SHADER_WRITE);
        builder.read_image(self.hiz_pyramid,          ResourceUsage::SHADER_READ);

        builder.descriptor_set(0, |d| {
            d.storage_buffer(self.lights)
             .storage_buffer(self.visible_light_ids)
             .storage_buffer(self.visible_light_count)
             .sampled_image(self.hiz_pyramid, DescriptorImageLayout::ShaderReadOnlyOptimal);
        });
    }

    fn execute(&self, ctx: &mut dyn PassContext, frame: &i3_gfx::graph::compiler::FrameBlackboard) {
        if self.light_count == 0 { return; }
        let Some(pipeline) = self.pipeline else {
            tracing::error!("LightOcclusionCullPass: pipeline not initialized");
            return;
        };
        let common = frame.consume::<crate::render_graph::CommonData>("Common");
        ctx.bind_pipeline_raw(pipeline);
        ctx.push_constant_data(ShaderStageFlags::Compute, 0, &LightOcclusionCullPushConstants {
            view_proj:     common.view_projection,
            light_count:   self.light_count,
            hiz_mip_count: self.mip_count,
            screen_w:      self.screen_w,
            screen_h:      self.screen_h,
        });
        let groups = (self.light_count + 63) / 64;
        ctx.dispatch(groups, 1, 1);
    }
}
```

---

### T3.4 — Déclarer `light_occlusion_cull` dans `mod.rs`

**Fichier :** `crates/i3_renderer/src/passes/mod.rs`

```rust
pub mod light_occlusion_cull;
```

---

### T3.5 — Ajouter `LightOcclusionCullPass` dans `DefaultRenderGraph`

**Fichier :** `crates/i3_renderer/src/render_graph.rs`

Même procédure que T2.4 :
1. Import en tête : `use crate::passes::light_occlusion_cull::LightOcclusionCullPass;`
2. Champ dans le struct : `pub light_occlusion_cull_pass: LightOcclusionCullPass`
3. Init dans `new()` : `let light_occlusion_cull_pass = LightOcclusionCullPass::new();`
4. Dans `init()` : `self.graph.init_pass_direct(&mut self.light_occlusion_cull_pass, backend);`

---

### T3.6 — Câbler `LightOcclusionCullPass` dans `declare()` et adapter `LightCullPass`

**Fichier :** `crates/i3_renderer/src/render_graph.rs`

Dans la closure `graph.declare`, **après `hiz_build_pass`** et **avant `clustering_group`**, ajouter :

```rust
// Clear light count — avant le cull
builder.add_owned_pass(crate::render_graph::ClearBufferPass {
    name:   "ClearVisibleLightCount".to_string(),
    buffer: builder.resolve_buffer("VisibleLightCountBuffer"),
});
// Light occlusion cull (pré-cluster)
builder.add_pass(&mut self.light_occlusion_cull_pass);
```

**Fichier :** `crates/i3_renderer/src/passes/light_cull.rs`

Modifier `LightCullPass` pour lire `VisibleLightBuffer` et `VisibleLightCountBuffer` au lieu de `LightBuffer` directement :

1. Ajouter deux champs :
```rust
visible_light_ids:   BufferHandle,
visible_light_count: BufferHandle,
```

2. Dans `declare()`, ajouter :
```rust
self.visible_light_ids   = builder.resolve_buffer("VisibleLightBuffer");
self.visible_light_count = builder.resolve_buffer("VisibleLightCountBuffer");
builder.read_buffer(self.visible_light_ids,   ResourceUsage::SHADER_READ);
builder.read_buffer(self.visible_light_count, ResourceUsage::SHADER_READ);
```
Ajouter dans le `descriptor_set` :
```rust
d.storage_buffer(self.visible_light_ids)
 .storage_buffer(self.visible_light_count);
```

3. Dans `execute()`, modifier le push constant `light_count` pour lire `VisibleLightCount` depuis un buffer plutôt que depuis `CommonData` — **ou**, comme simplification acceptable pour la première itération : garder `light_count: common.light_count` (le shader itérera tous les lights, dont certains seront filtrés par la `visible_light_ids` indirection).

**Modification du shader `light_cull.slang`** :  
Changer l'itération de `for (uint li = 0; li < light_count; li++)` en :
```slang
uint visible_count = visible_light_count[0];
for (uint vi = 0; vi < visible_count; vi++) {
    uint li = visible_light_ids[vi];
    GpuLightData light = lights[li];
    // ... reste de la logique inchangée
}
```

**Test :**
1. Compilation passe.
2. Runtime : l'éclairage est correct (même résultat que sans le filtre).
3. Avec une scène contenant des lumières ponctuelles cachées derrière des murs, vérifier que `visible_light_count` (lisible via debug) est inférieur à `light_count`.

---

## Récapitulatif des fichiers

| Fichier | Action |
|---|---|
| `crates/i3_gfx/src/graph/pass.rs` | T0.1 : ajouter `import_image` |
| `crates/i3_renderer/src/render_graph.rs` | T0.2, T0.3, T1.4, T1.5, T2.4, T3.1, T3.5, T3.6 |
| `crates/i3_renderer/src/passes/mod.rs` | T1.3, T2.3, T3.4 |
| `crates/i3_renderer/src/passes/hiz_build.rs` | T1.2 (nouveau) |
| `crates/i3_renderer/src/passes/occlusion_cull.rs` | T2.2 (nouveau) |
| `crates/i3_renderer/src/passes/light_occlusion_cull.rs` | T3.3 (nouveau) |
| `crates/i3_renderer/src/passes/light_cull.rs` | T3.6 : adapter pour VisibleLightBuffer |
| `crates/i3_renderer/src/passes/cull.rs` | T2.5 : supprimer si plus utilisé |
| `crates/i3_renderer/assets/shaders/hiz_build.slang` | T1.1 (nouveau) |
| `crates/i3_renderer/assets/shaders/occlusion_cull.slang` | T2.1 (nouveau) |
| `crates/i3_renderer/assets/shaders/light_occlusion_cull.slang` | T3.2 (nouveau) |

## Ordre d'exécution recommandé

```
T0.1 → T0.2 → T0.3                   (infrastructure, pas de runtime visible)
T1.1 → T1.2 → T1.3 → T1.4 → T1.5   (HiZ visible en debug viz)
T2.1 → T2.2 → T2.3 → T2.4 → T2.5   (frustum + Hi-Z cull actif)
T3.1 → T3.2 → T3.3 → T3.4 → T3.5 → T3.6  (light occlusion cull actif)
```

Chaque milestone est **standalone** : si M2 ou M3 ne sont pas commencés, M1 seule produit quand même un HiZ utilisable par debug viz.
