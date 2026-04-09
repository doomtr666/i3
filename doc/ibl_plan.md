# IBL Baking — Plan d'implémentation

Chaque tâche est indépendante et testable. Les dépendances sont indiquées explicitement.

---

## TASK-00 — Bake de l'equirect HDR en BC6H_UF16 (embarqué dans l'IblAsset)

**Pourquoi** : l'HDRI equirectangulaire compressé est nécessaire pour la `SkyPass` (TASK-13). Il est stocké directement dans le payload de l'`IblAsset` — pas de `TextureAsset` séparé, pas de lookup UUID. L'asset IBL est self-contained.

**Fichier** : `crates/i3_baker/src/importers/ibl_bake.rs`

**Fonction** :
```rust
/// Compresse les pixels f32 RGB equirect en BC6H_UF16.
/// Retourne les données brutes compressées (sans header).
pub fn compress_env_bc6h(pixels_rgb: &[[f32; 3]], width: u32, height: u32) -> Vec<u8>
```

**Implémentation** :
```rust
use intel_tex_2::{bc6h, RgbaSurface};

pub fn compress_env_bc6h(pixels_rgb: &[[f32; 3]], width: u32, height: u32) -> Vec<u8> {
    // intel_tex_2 attend RGBA f32
    let mut rgba: Vec<f32> = Vec::with_capacity(width as usize * height as usize * 4);
    for px in pixels_rgb {
        rgba.extend_from_slice(px);
        rgba.push(1.0);
    }
    let surface = RgbaSurface {
        width,
        height,
        stride: width * 16, // 4 × f32
        data: bytemuck::cast_slice(&rgba),
    };
    bc6h::compress_blocks(&bc6h::opaque_settings(), &surface)
}
```

BC6H_UF16 : compression 8:1 par rapport à RGBA16F, plage [0, +∞), supporté sur toute carte moderne (`VK_FORMAT_BC6H_UFLOAT_BLOCK`).

**Vérification** : `compress_env_bc6h` sur une image 4×2 retourne `len == 4 * 2 / 4 * 16` (taille BC6H = nb_blocs_4x4 × 16 octets).

---

## TASK-01 — Ajouter `R16G16_SFLOAT`, `R11G11B10_UFLOAT` et `BC6H_UF16` à `TextureFormat`

**Fichier** : `crates/i3_io/src/texture.rs`

**Modification** : ajouter trois variantes à l'enum `TextureFormat` existant :

```rust
pub enum TextureFormat {
    // ... variantes existantes ...
    R16G16_SFLOAT = 10,      // BRDF LUT
    R11G11B10_UFLOAT = 11,   // Irradiance + Pre-filtered hemi-octa
    BC6H_UF16 = 12,          // Skybox/environment equirect HDR
}
```

**Aussi** : ajouter le mapping dans `crates/i3_vulkan_backend/src/convert.rs` dans la fonction `convert_format` :
```rust
Format::R16G16_SFLOAT    => vk::Format::R16G16_SFLOAT,
Format::R11G11B10_UFLOAT => vk::Format::B10G11R11_UFLOAT_PACK32,
Format::BC6H_UF16        => vk::Format::BC6H_UFLOAT_BLOCK,
```

**Vérification** : `cargo check -p i3_io && cargo check -p i3_vulkan_backend` passent sans erreur.

---

## TASK-02 — Ajouter `IBL_ASSET_TYPE` UUID dans `i3_io`

**Fichier** : `crates/i3_io/src/ibl.rs` (nouveau fichier)

**Contenu complet** :

```rust
use uuid::{Uuid, uuid};

pub const IBL_ASSET_TYPE: Uuid = uuid!("4a7b1c2d-3e4f-5a6b-7c8d-9e0f1a2b3c4d");

/// Données soleil extraites automatiquement du HDR.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IblSunData {
    pub direction: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
    pub _pad: f32,
}

/// Header sérialisé en tête du BakeOutput IBL.
/// Layout binaire : IblHeader + brdf_lut_data + irradiance_data + prefiltered_data + env_data
/// L'asset est self-contained : aucune référence UUID externe nécessaire au runtime.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IblHeader {
    // BRDF LUT : R16G16_SFLOAT, 256x256, 1 mip
    pub lut_width: u32,
    pub lut_height: u32,
    pub lut_format: u32,       // TextureFormat::R16G16_SFLOAT as u32 = 10
    pub lut_data_size: u32,

    // Irradiance hemi-octa : R11G11B10_UFLOAT, 64x64, 1 mip
    pub irr_width: u32,
    pub irr_height: u32,
    pub irr_format: u32,       // TextureFormat::R11G11B10_UFLOAT as u32 = 11
    pub irr_data_size: u32,

    // Pre-filtered hemi-octa : R11G11B10_UFLOAT, 256x256, 6 mips
    pub pref_width: u32,
    pub pref_height: u32,
    pub pref_format: u32,      // TextureFormat::R11G11B10_UFLOAT as u32 = 11
    pub pref_mip_levels: u32,
    pub pref_data_size: u32,

    // Equirect HDR compressé : BC6H_UF16, résolution originale, 1 mip
    pub env_width: u32,
    pub env_height: u32,
    pub env_format: u32,       // TextureFormat::BC6H_UF16 as u32 = 12
    pub env_data_size: u32,

    /// Facteur de conversion HDR brut → unités physiques (lux).
    /// Appliqué à sun_intensity au bake. Le renderer multiplie l'IBL ambient par cette valeur.
    pub intensity_scale: f32,

    pub _pad: [u32; 3],

    pub sun: IblSunData,
}

pub struct IblAsset {
    pub header: IblHeader,
    pub data: Vec<u8>, // brdf_lut || irradiance || prefiltered || env_equirect (concaténés)
}

impl crate::asset::Asset for IblAsset {
    const ASSET_TYPE_ID: [u8; 16] = *IBL_ASSET_TYPE.as_bytes();

    fn load(_header: &crate::AssetHeader, data: &[u8]) -> crate::error::Result<Self> {
        let hdr_size = std::mem::size_of::<IblHeader>();
        let ibl_header: IblHeader = bytemuck::pod_read_unaligned(&data[..hdr_size]);
        Ok(Self { header: ibl_header, data: data[hdr_size..].to_vec() })
    }
}
```

**Aussi** : ajouter `pub mod ibl;` dans `crates/i3_io/src/lib.rs`.

**Vérification** : `cargo check -p i3_io` passe.

---

## TASK-03 — Fonctions de projection hemi-octa (module partagé)

**Fichier** : `crates/i3_baker/src/importers/ibl_math.rs` (nouveau fichier)

**Contenu** : fonctions CPU pures, pas de dépendances externes.

```rust
use std::f32::consts::PI;

/// Decode hemi-octa UV [0,1]² → direction sur la sphère unité.
/// L'espace [0, 0.5] en V = hémisphère nord (y >= 0).
/// L'espace [0.5, 1] en V = hémisphère sud (y < 0).
pub fn hemi_octa_decode(u: f32, v: f32) -> [f32; 3] {
    // Remappe V dans [0,1] selon hémisphère
    let (vv, sign_y) = if v < 0.5 { (v * 2.0, 1.0f32) } else { ((v - 0.5) * 2.0, -1.0f32) };
    let fx = u * 2.0 - 1.0;
    let fz = vv * 2.0 - 1.0;
    let fy = 1.0 - fx.abs() - fz.abs();
    let len = (fx * fx + fy * fy + fz * fz).sqrt().max(1e-8);
    [fx / len, sign_y * fy.abs() / len, fz / len]
}

/// Encode direction sphère → hemi-octa UV [0,1]².
pub fn hemi_octa_encode(d: [f32; 3]) -> (f32, f32) {
    let [x, y, z] = d;
    let l = x.abs() + y.abs() + z.abs();
    let ox = x / l;
    let oz = z / l;
    let u = ox * 0.5 + 0.5;
    // Replie selon hémisphère
    let v_local = oz * 0.5 + 0.5;
    let v = if y >= 0.0 { v_local * 0.5 } else { 0.5 + v_local * 0.5 };
    (u, v)
}

/// Pixel equirect (ix, iy) → direction sphère unité.
pub fn equirect_to_dir(ix: u32, iy: u32, width: u32, height: u32) -> [f32; 3] {
    let u = (ix as f32 + 0.5) / width as f32;
    let v = (iy as f32 + 0.5) / height as f32;
    let phi = u * 2.0 * PI - PI;       // [-π, π]
    let theta = v * PI;                  // [0, π]
    let sin_theta = theta.sin();
    [sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin()]
}

/// Solid angle d'un pixel equirect (correction cos(lat)).
pub fn equirect_solid_angle(iy: u32, height: u32) -> f32 {
    let theta = (iy as f32 + 0.5) / height as f32 * PI;
    (2.0 * PI / (height as f32)) * (PI / height as f32) * theta.sin()
}

/// Luminance perceptuelle d'un pixel RGB linéaire.
pub fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}
```

**Vérification** : `cargo test -p i3_baker` — écrire 2 tests : `hemi_octa_decode(0.5, 0.25)` ≈ `[0,1,0]` (zénith nord), `hemi_octa_decode(0.5, 0.75)` ≈ `[0,-1,0]` (zénith sud).

---

## TASK-04 — Bake BRDF LUT (CPU, importance sampling GGX)

**Fichier** : `crates/i3_baker/src/importers/ibl_bake.rs` (nouveau fichier)

**Fonction** :

```rust
/// Retourne Vec<u8> de taille width*height*4 octets (2× f16 par pixel = scale, bias).
/// Utilise rayon::par_iter pour paralléliser.
pub fn bake_brdf_lut(width: u32, height: u32, num_samples: u32) -> Vec<u8>
```

**Algorithme** (split-sum de Brian Karis, Epic Games 2013) :

Pour chaque pixel `(ix, iy)` :
- `nov = (ix as f32 + 0.5) / width as f32`  — NdotV
- `roughness = (iy as f32 + 0.5) / height as f32`
- `alpha = roughness * roughness`
- Accumuler `num_samples` échantillons via importance sampling GGX :
  - Générer `(xi0, xi1)` par séquence Hammersley
  - `H = importance_sample_ggx(xi0, xi1, alpha)` → half vector
  - `L = reflect(-V, H)` avec `V = [sqrt(1-nov²), nov, 0]`
  - `nov_h = dot(V, H).max(0)`
  - `nol = L[1].max(0)`, `noh = H[1].max(0)`
  - Terme G = G_Smith_schlick(nol, nov, roughness)
  - `G_vis = G * nov_h / (noh * nov).max(1e-4)`
  - `fc = (1 - nov_h).powi(5)`
  - `scale += (1 - fc) * G_vis`, `bias += fc * G_vis`
- `scale /= num_samples`, `bias /= num_samples`
- Encoder en f16 (`half` crate) : `[f16::from_f32(scale), f16::from_f32(bias)]`

**Dépendance** : ajouter `half = "2"` dans `Cargo.toml` de `i3_baker`.

**Vérification** : appeler `bake_brdf_lut(256, 256, 1024)` et vérifier que le pixel `(255, 0)` (NdotV=1, roughness≈0) ≈ `(1.0, 0.0)`.

---

## TASK-05 — Bake Irradiance hemi-octa (CPU, intégrale cosine-weighted)

**Fichier** : `crates/i3_baker/src/importers/ibl_bake.rs` (même fichier que TASK-04)

**Dépendance** : TASK-03 (`ibl_math`)

**Fonction** :

```rust
/// pixels_rgb : slice de width_src*height_src triplets f32 (RGB linéaire, equirect).
/// sun_threshold : luminance limite — pixels au-dessus sont exclus (masquage solaire).
///   Passer 0.0 pour désactiver le masquage (tout intégrer).
/// Retourne Vec<u8> taille out_size*out_size*4 (R11G11B10 packed u32).
pub fn bake_irradiance(
    pixels_rgb: &[[f32; 3]],
    src_width: u32,
    src_height: u32,
    out_size: u32,   // typiquement 64
    sun_threshold: f32,
) -> Vec<u8>
```

**Algorithme** (par pixel de sortie) :

Pour chaque texel `(ix, iy)` de la map hemi-octa `out_size × out_size` :
- `n = hemi_octa_decode(u, v)` — direction normale
- Accumuler tous les pixels source `(sx, sy)` :
  - `dir = equirect_to_dir(sx, sy, src_width, src_height)`
  - `cos_theta = dot(n, dir)` — si `<= 0` : skip
  - **Si `luminance(px) > sun_threshold && sun_threshold > 0.0` : skip** (masquage solaire)
  - `solid = equirect_solid_angle(sy, src_height)`
  - `irr += pixel_rgb[sx + sy*src_width] * cos_theta * solid`
- Diviser par `π`
- Encoder en R11G11B10 : voir TASK-06

**Pourquoi le masquage** : le soleil contribue de façon dominante à l'intégrale. Si on garde sa contribution ici et qu'on ajoute aussi une `DirectionalLight` extraite du HDR, on double-compte son énergie. En excluant les pixels solaires de l'intégration, l'irradiance ne représente que le ciel ambiant. La `DirectionalLight` reconstitue la contribution directe exacte.

**Note** : utiliser `rayon::par_iter` sur les pixels de sortie.

---

## TASK-06 — Encodage R11G11B10_UFLOAT

**Fichier** : `crates/i3_baker/src/importers/ibl_bake.rs`

**Fonction** :

```rust
/// Encode un triplet RGB f32 en R11G11B10 unsigned float packed dans un u32.
/// R = bits [0..10], G = bits [11..21], B = bits [22..31]
/// Format : 5 bits exposant + 6/6/5 bits mantisse (pas de signe).
pub fn encode_r11g11b10(r: f32, g: f32, b: f32) -> u32
```

Encodage d'un composant en float non signé n bits (r=6 mantisse, b=5 mantisse) :
```rust
fn encode_ufloat(val: f32, exp_bits: u32, mant_bits: u32) -> u32 {
    let val = val.max(0.0);
    if val == 0.0 { return 0; }
    let bits = val.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + (1 << (exp_bits - 1)) as i32 - 1;
    if exp <= 0 { return 0; }
    if exp >= (1 << exp_bits) as i32 - 1 { return ((1 << exp_bits) - 1) << mant_bits | ((1 << mant_bits) - 1); }
    let mant = (bits >> (23 - mant_bits)) & ((1 << mant_bits) - 1);
    ((exp as u32) << mant_bits) | mant
}
```

**Vérification** : `encode_r11g11b10(1.0, 1.0, 1.0)` doit produire une valeur connue (comparer avec une implémentation de référence).

---

## TASK-07 — Bake Pre-filtered hemi-octa (CPU, importance sampling GGX)

**Fichier** : `crates/i3_baker/src/importers/ibl_bake.rs`

**Dépendances** : TASK-03, TASK-06

**Fonction** :

```rust
/// Retourne Vec<u8> contenant les données concaténées de tous les mips.
/// Chaque mip est out_size/(2^mip) × out_size/(2^mip) texels R11G11B10.
/// sun_threshold : même valeur que bake_irradiance — exclut les pixels solaires.
pub fn bake_prefiltered(
    pixels_rgb: &[[f32; 3]],
    src_width: u32,
    src_height: u32,
    out_size: u32,     // typiquement 256
    num_mips: u32,     // typiquement 6
    num_samples: u32,  // typiquement 512
    sun_threshold: f32,
) -> Vec<u8>
```

**Algorithme** pour chaque mip `m` :
- `roughness = m as f32 / (num_mips - 1) as f32`
- `alpha = roughness * roughness`
- Taille mip : `mip_size = (out_size >> m).max(1)`
- Pour chaque texel `(ix, iy)` :
  - `r = hemi_octa_decode(u, v)` — direction reflet
  - `N = V = R = r` (approximation split-sum)
  - Accumuler `num_samples` par importance sampling GGX :
    - `H = importance_sample_ggx(hammersley(i, num_samples), alpha)`
    - transformer H dans le repère de N (TBN)
    - `L = reflect(-V, H)`
    - `nol = dot(N, L).max(0)` — si `<= 0` : skip
    - **Si `luminance(sample_equirect(L)) > sun_threshold && sun_threshold > 0.0` : skip** (masquage solaire)
    - `color += sample_equirect(pixels_rgb, L, src_width, src_height) * nol`
    - `weight += nol`
  - `color /= weight.max(1e-4)`
  - Encoder en R11G11B10
- Appender les données du mip au Vec résultat

**Fonctions helper nécessaires** :
```rust
fn hammersley(i: u32, n: u32) -> (f32, f32)  // séquence quasi-Monte Carlo
fn importance_sample_ggx(xi: (f32, f32), alpha: f32) -> [f32; 3]  // half-vector GGX
fn tbn_transform(h: [f32; 3], n: [f32; 3]) -> [f32; 3]  // tangent space → world
fn sample_equirect(pixels: &[[f32;3]], dir: [f32;3], w: u32, h: u32) -> [f32;3]  // nearest
```

---

## TASK-08 — Extraction du soleil depuis HDR

**Fichier** : `crates/i3_baker/src/importers/ibl_bake.rs`

**Dépendances** : TASK-03

**Struct d'options** (à placer en tête de `ibl_bake.rs`) :

```rust
/// Options de bake IBL.
/// extract_sun = true  → HDR extérieur avec soleil dominant.
///   Les pixels au-dessus du seuil sont masqués dans les maps IBL et retournés
///   en DirectionalLight. Nécessite une source d'intensité très élevée pour
///   projeter des ombres cohérentes.
/// extract_sun = false → HDR intérieur ou multi-sources comparables.
///   Aucun masquage, IblSunData zeroed. L'IBL fournit tout l'éclairage.
#[derive(Debug, Clone)]
pub struct IblBakeOptions {
    pub extract_sun: bool,

    /// Multiplicateur appliqué à sun_intensity et stocké dans IblHeader.
    /// Convertit les valeurs HDR brutes vers des unités physiques (lux).
    /// Extérieur ensoleillé typique : ~3000.0 (soleil HDR ≈ 32 → ~100 000 lux).
    /// Défaut 1.0 — calibration à affiner ultérieurement.
    pub intensity_scale: f32,
}

impl Default for IblBakeOptions {
    fn default() -> Self { Self { extract_sun: true, intensity_scale: 1.0 } }
}
```

**Fonctions** :

```rust
/// Retourne le seuil de luminance au 99.9ème percentile.
/// Partagé avec bake_irradiance / bake_prefiltered pour garantir la cohérence du masquage.
pub fn compute_sun_threshold(pixels_rgb: &[[f32; 3]]) -> f32

/// Extrait la lumière directionnelle dominante depuis les pixels au-dessus du seuil.
pub fn extract_sun(
    pixels_rgb: &[[f32; 3]],
    src_width: u32,
    src_height: u32,
    sun_threshold: f32,
) -> IblSunData  // struct de TASK-02
```

**Algorithme `compute_sun_threshold`** :
1. Calculer luminance de chaque pixel
2. Trier et retourner le 99.9ème percentile
   (alternative plus simple : `median * 10.0` — suffisant pour les HDRs extérieurs)

**Algorithme `extract_sun`** :
1. Utiliser `sun_threshold` passé en paramètre (déjà calculé en amont)
2. Weighted centroid des pixels au-dessus du seuil :
   ```
   pour chaque pixel (ix, iy) avec lum > threshold :
       dir = equirect_to_dir(ix, iy, w, h)
       solid = equirect_solid_angle(iy, h)
       w_dir   += dir * lum * solid
       w_color += pixel_rgb * solid
       total   += lum * solid
       count   += 1
   sun_dir = normalize(w_dir / total)
   sun_color = normalize(w_color / count)  // teinte normalisée
   sun_intensity = total / count
   ```
4. Si `count == 0` : retourner soleil par défaut `direction=[0,-1,0], color=[1,1,1], intensity=1`

---

## TASK-09 — `HdrIblImporter` : struct + `import()`

**Fichier** : `crates/i3_baker/src/importers/hdr_ibl_importer.rs` (nouveau)

**Dépendances** : TASK-02, TASK-03

```rust
use image::Rgb32FImage;
use std::path::{Path, PathBuf};
use crate::pipeline::{ImportedData, Importer, BakeOutput, BakeContext, Result};

pub struct HdrImportedData {
    pub source_path: PathBuf,
    pub pixels: Vec<[f32; 3]>,
    pub width: u32,
    pub height: u32,
}

impl ImportedData for HdrImportedData {
    fn source_path(&self) -> &Path { &self.source_path }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

pub struct HdrIblImporter {
    pub options: IblBakeOptions,
}

impl Default for HdrIblImporter {
    fn default() -> Self { Self { options: IblBakeOptions::default() } }
}

impl Importer for HdrIblImporter {
    fn name(&self) -> &str { "HdrIblImporter" }
    fn source_extensions(&self) -> &[&str] { &["hdr", "exr"] }

    fn import(&self, source_path: &Path) -> Result<Box<dyn ImportedData>> {
        let img = image::open(source_path)
            .map_err(|e| crate::BakerError::Pipeline(e.to_string()))?
            .into_rgb32f();
        let width = img.width();
        let height = img.height();
        let pixels: Vec<[f32; 3]> = img.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        Ok(Box::new(HdrImportedData { source_path: source_path.to_path_buf(), pixels, width, height }))
    }

    fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
        // Implémenté dans TASK-10
        todo!()
    }
}
```

**Vérification** : `import()` sur un `.hdr` 16×8 de test ne panique pas, retourne width=16 height=8.

---

## TASK-10 — `HdrIblImporter::extract()` : assembler les 3 maps + soleil

**Fichier** : `crates/i3_baker/src/importers/hdr_ibl_importer.rs`

**Dépendances** : TASK-02, TASK-04, TASK-05, TASK-07, TASK-08

Remplacer le `todo!()` de TASK-09 par :

```rust
fn extract(&self, data: &dyn ImportedData, _ctx: &BakeContext) -> Result<Vec<BakeOutput>> {
    use crate::importers::ibl_bake::*;
    use i3_io::ibl::{IblHeader, IBL_ASSET_TYPE};

    let hdr = data.as_any().downcast_ref::<HdrImportedData>().unwrap();

    // Calcul du seuil solaire — partagé entre masquage IBL et extraction DirectionalLight.
    // Si extract_sun == false (HDR intérieur) : threshold = f32::MAX → rien masqué,
    // IblSunData zeroed, pas de DirectionalLight ajoutée au runtime.
    let sun_threshold = if self.options.extract_sun {
        compute_sun_threshold(&hdr.pixels) // 99.9ème percentile de luminance
    } else {
        f32::MAX
    };

    // Bake les 3 maps (masquage conditionnel selon le threshold)
    let lut_data   = bake_brdf_lut(256, 256, 1024);
    let irr_data   = bake_irradiance(&hdr.pixels, hdr.width, hdr.height, 64, sun_threshold);
    let pref_data  = bake_prefiltered(&hdr.pixels, hdr.width, hdr.height, 256, 6, 512, sun_threshold);

    // Extraction soleil — retourne IblSunData::zeroed() si threshold == f32::MAX
    let mut sun = extract_sun(&hdr.pixels, hdr.width, hdr.height, sun_threshold);
    sun.intensity *= self.options.intensity_scale;

    // Compression equirect (après bake, avant sérialisation)
    let env_data = compress_env_bc6h(&hdr.pixels, hdr.width, hdr.height);

    // Header — asset self-contained, aucune référence UUID externe
    let header = IblHeader {
        lut_width: 256, lut_height: 256,
        lut_format: 10, // R16G16_SFLOAT
        lut_data_size: lut_data.len() as u32,
        irr_width: 64, irr_height: 64,
        irr_format: 11, // R11G11B10_UFLOAT
        irr_data_size: irr_data.len() as u32,
        pref_width: 256, pref_height: 256,
        pref_format: 11,
        pref_mip_levels: 6,
        pref_data_size: pref_data.len() as u32,
        env_width: hdr.width, env_height: hdr.height,
        env_format: 12, // BC6H_UF16
        env_data_size: env_data.len() as u32,
        intensity_scale: self.options.intensity_scale,
        _pad: [0; 3],
        sun,
    };

    // Sérialisation : header + lut + irr + pref + env
    let mut payload = bytemuck::bytes_of(&header).to_vec();
    payload.extend_from_slice(&lut_data);
    payload.extend_from_slice(&irr_data);
    payload.extend_from_slice(&pref_data);
    payload.extend_from_slice(&env_data);

    // UUID déterministe depuis le path
    let asset_id = uuid::Uuid::new_v5(
        &uuid::Uuid::NAMESPACE_URL,
        hdr.source_path.to_string_lossy().as_bytes(),
    );

    Ok(vec![BakeOutput {
        asset_id,
        asset_type: IBL_ASSET_TYPE,
        data: payload,
        name: hdr.source_path.file_stem().unwrap_or_default().to_string_lossy().into_owned(),
    }])
}
```

---

## TASK-11 — Enregistrer `HdrIblImporter` dans le baker

**Fichiers** :

1. `crates/i3_baker/src/importers/mod.rs` — ajouter :
```rust
pub mod hdr_ibl_importer;
pub use hdr_ibl_importer::HdrIblImporter;
```

2. `crates/i3_baker/src/pipeline.rs` — ajouter méthode à `BundleBaker` :
```rust
pub fn add_hdr_ibl(self, source_path: impl AsRef<Path>, options: IblBakeOptions) -> Self {
    self.add_asset(source_path, crate::importers::HdrIblImporter { options })
}
```

3. `crates/i3_baker/src/bin/baker.rs` — ajouter après `add_pipelines(egui_pipelines)` :
```rust
// Chemin à adapter selon où les HDR sont stockés
.add_hdr_ibl("crates/i3_renderer/assets/env/default.hdr")
```

**Vérification** : `cargo run -p i3_baker --bin baker` produit un asset IBL dans `assets/system.i3b`.

---

## TASK-12 — Mettre à jour le prelude `i3_baker`

**Fichier** : `crates/i3_baker/src/lib.rs`

Ajouter dans les exports publics :
```rust
pub use importers::HdrIblImporter;
```

---

## TASK-13 — SkyPass : remplacer le ciel procédural par l'equirect HDRI

**Dépendance** : TASK-00 (equirect baked as BC6H_UF16) + TASK-02 (IblHeader::env_texture_id)

**Objectif** : afficher la carte d'environnement comme ciel au lieu du gradient Rayleigh.

### `crates/i3_renderer/assets/shaders/sky.slang`

Remplacer `get_sky_color()` par un sample de l'equirect :

```hlsl
// Ajouter après les push constants (descriptor set 1 pour l'env map) :
[[vk::binding(0, 1)]] Texture2D envMap;
[[vk::binding(0, 1)]] SamplerState envSampler;

// Ajouter la fonction de projection equirect :
float2 equirect_uv(float3 dir) {
    float u = atan2(dir.z, dir.x) / (2.0 * 3.14159265) + 0.5;
    float v = acos(clamp(dir.y, -1.0, 1.0)) / 3.14159265;
    return float2(u, v);
}

// Dans fragmentMain — remplacer tout le corps par :
[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    float3 dir = normalize(input.view_dir);
    float3 color = envMap.SampleLevel(envSampler, equirect_uv(dir), 0).rgb;
    return float4(color, 1.0);
}
```

Les push constants `sun_direction/intensity/color` peuvent rester (utilisées par d'autres passes ou futures features).

### `crates/i3_renderer/src/passes/sky.rs`

**Ajouter les champs** dans `SkyPass` :
```rust
env_map: ImageHandle,        // handle de la texture equirect
ibl_sampler: SamplerHandle,  // sampler pour l'env map
```

**Dans `init()`** — après le chargement du pipeline, charger l'IBL :
```rust
// L'IblAsset est self-contained : la texture equirect est dans ibl_asset.data
if let Ok(ibl_asset) = loader.load::<i3_io::ibl::IblAsset>("default_ibl").wait_loaded() {
    let h = &ibl_asset.header;
    let lut_end  = h.lut_data_size as usize;
    let irr_end  = lut_end + h.irr_data_size as usize;
    let pref_end = irr_end + h.pref_data_size as usize;
    // env_data est le dernier bloc
    let env_tex = build_texture(h.env_width, h.env_height, 1, h.env_format,
                                &ibl_asset.data[pref_end..]);
    self.env_map = backend.upload_texture(&env_tex);
}
self.ibl_sampler = backend.create_sampler(SamplerDesc::linear_clamp());
```

Note : si `env_map` reste `INVALID` (IBL non baked), le descriptor set 1 n'est pas émis — acceptable en dev.

**Dans `declare()`** — ajouter le descriptor set 1 si la texture est disponible :
```rust
if self.env_map != ImageHandle::INVALID {
    builder.read_image(self.env_map, ResourceUsage::SHADER_READ);
    builder.descriptor_set(1, |d| {
        d.combined_image_sampler(
            self.env_map,
            DescriptorImageLayout::ShaderReadOnlyOptimal,
            self.ibl_sampler,
        );
    });
}
```

**Vérification** : lancer le viewer avec un bundle contenant un IBL baked → le ciel affiche l'equirect au lieu du gradient bleu.

---

## TASK-14 — DeferredResolvePass : terme ambiant IBL (split-sum)

**Dépendances** : TASK-02, TASK-04, TASK-05, TASK-07 (maps baked) + TASK-13 (IblAsset chargé)

**Objectif** : remplacer `0.005 * albedo` par un terme ambiant PBR complet (diffuse IBL + specular IBL via split-sum).

### `crates/i3_renderer/assets/shaders/deferred_resolve.slang`

**Ajouter 3 nouveaux bindings** (après binding 9 tlas) :
```hlsl
[[vk::binding(10, 0)]] Texture2D brdfLut;
[[vk::binding(10, 0)]] SamplerState lutSampler;

[[vk::binding(11, 0)]] Texture2D irradianceMap;   // hemi-octa dual R11G11B10
[[vk::binding(11, 0)]] SamplerState irrSampler;

[[vk::binding(12, 0)]] Texture2D prefilteredMap;  // hemi-octa dual R11G11B10, mips
[[vk::binding(12, 0)]] SamplerState prefSampler;
```

**Ajouter la constante et la fonction hemi-octa** :
```hlsl
static const float MAX_PREFILTER_MIP = 5.0; // num_mips - 1

float2 hemisphereOctEncode(float3 n) {
    float l = abs(n.x) + abs(n.y) + abs(n.z);
    float ox = n.x / l;
    float oz = n.z / l;
    float u = ox * 0.5 + 0.5;
    float v_local = oz * 0.5 + 0.5;
    float v = (n.y >= 0.0) ? v_local * 0.5 : 0.5 + v_local * 0.5;
    return float2(u, v);
}

float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness) {
    return F0 + (max(float3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
```

**Dans `fragmentMain`** — remplacer la ligne `float3 color = emissive + 0.005 * albedo;` par :
```hlsl
// IBL ambient
float3 F0 = lerp(float3(0.04), albedo, metallic);
float NdotV = saturate(dot(normal, V));

// Diffuse IBL
float3 kS = fresnelSchlickRoughness(NdotV, F0, roughness);
float3 kD = (1.0 - kS) * (1.0 - metallic);
float3 irradiance = irradianceMap.Sample(irrSampler, hemisphereOctEncode(normal)).rgb;
float3 diffuse_ambient = kD * albedo * irradiance;

// Specular IBL (split-sum)
float3 R = reflect(-V, normal);
float mip_level = roughness * MAX_PREFILTER_MIP;
float3 prefiltered = prefilteredMap.SampleLevel(prefSampler, hemisphereOctEncode(R), mip_level).rgb;
float2 brdf = brdfLut.Sample(lutSampler, float2(NdotV, roughness)).rg;
float3 specular_ambient = prefiltered * (F0 * brdf.x + brdf.y);

float3 color = emissive + diffuse_ambient + specular_ambient;
```

### `crates/i3_renderer/src/passes/deferred_resolve.rs`

**Ajouter 3 champs** dans `DeferredResolvePass` :
```rust
ibl_brdf_lut: ImageHandle,
ibl_irradiance: ImageHandle,
ibl_prefiltered: ImageHandle,
```
Initialisés à `ImageHandle::INVALID` dans `new()`.

**Dans `init()`** — charger les textures IBL (toutes embarquées dans le même asset) :
```rust
if let Ok(ibl_asset) = loader.load::<i3_io::ibl::IblAsset>("default_ibl").wait_loaded() {
    let header = &ibl_asset.header;

    // Offsets dans ibl_asset.data : lut | irr | pref | env (dans l'ordre du header)
    let lut_end  = header.lut_data_size as usize;
    let irr_end  = lut_end  + header.irr_data_size as usize;
    let pref_end = irr_end  + header.pref_data_size as usize;

    let lut_tex = i3_io::texture::TextureAsset::from_raw(
        header.lut_width, header.lut_height, 1, 1, header.lut_format,
        &ibl_asset.data[..lut_end],
    );
    let irr_tex = i3_io::texture::TextureAsset::from_raw(
        header.irr_width, header.irr_height, 1, 1, header.irr_format,
        &ibl_asset.data[lut_end..irr_end],
    );
    let pref_tex = i3_io::texture::TextureAsset::from_raw(
        header.pref_width, header.pref_height, 1, header.pref_mip_levels, header.pref_format,
        &ibl_asset.data[irr_end..pref_end],
    );

    self.ibl_brdf_lut    = backend.upload_texture(&lut_tex);
    self.ibl_irradiance  = backend.upload_texture(&irr_tex);
    self.ibl_prefiltered = backend.upload_texture(&pref_tex);
}
```

Note : `TextureAsset::from_raw` est à créer si absent (méthode de construction directe sans passer par `Asset::load`). Sinon, construire manuellement le `TextureHeader` + `TextureData` struct.

**Dans `declare()`** — ajouter les 3 bindings au descriptor set existant (après binding 9) :
```rust
if self.ibl_brdf_lut != ImageHandle::INVALID {
    builder.read_image(self.ibl_brdf_lut,    ResourceUsage::SHADER_READ);
    builder.read_image(self.ibl_irradiance,  ResourceUsage::SHADER_READ);
    builder.read_image(self.ibl_prefiltered, ResourceUsage::SHADER_READ);
}

// Dans descriptor_set(0, |d| { ... }) — ajouter après l'acceleration_structure :
if self.ibl_brdf_lut != ImageHandle::INVALID {
    d.bind(10).combined_image_sampler(
        self.ibl_brdf_lut, DescriptorImageLayout::ShaderReadOnlyOptimal, self.sampler,
    );
    d.bind(11).combined_image_sampler(
        self.ibl_irradiance, DescriptorImageLayout::ShaderReadOnlyOptimal, self.sampler,
    );
    d.bind(12).combined_image_sampler(
        self.ibl_prefiltered, DescriptorImageLayout::ShaderReadOnlyOptimal, self.sampler,
    );
}
```

**Vérification** : avec un bundle contenant un IBL baked, les surfaces opaques reçoivent un terme ambiant coloreé par l'environnement. Sans IBL : `ImageHandle::INVALID` → bindings ignorés, shader lit des textures non liées (comportement indéfini → à corriger avec un fallback 1×1 noir pour release).

---

## Ordre d'exécution recommandé

```
TASK-00 (R16G16B16A16_SFLOAT + equirect bake — spec)
TASK-01 (TextureFormat R16G16 + R11G11B10)
TASK-02 (IblAsset i3_io — inclure env_texture_id)
TASK-03 (ibl_math)
TASK-06 (encode R11G11B10)
TASK-04 (BRDF LUT)        ← dépend TASK-03, TASK-06
TASK-05 (Irradiance)      ← dépend TASK-03, TASK-06
TASK-08 (Sun extract)     ← dépend TASK-03
TASK-07 (Pre-filtered)    ← dépend TASK-03, TASK-06
TASK-09 (Importer import) ← dépend TASK-02, TASK-03
TASK-10 (Importer extract)← dépend tout
TASK-11 (Registration)    ← dépend TASK-10
TASK-12 (Prelude)         ← dépend TASK-11
TASK-13 (SkyPass HDRI)    ← dépend TASK-00, TASK-02
TASK-14 (IBL ambient)     ← dépend TASK-04, TASK-05, TASK-07
```

---

## Dépendances Cargo à ajouter dans `crates/i3_baker/Cargo.toml`

```toml
half = "2"        # f16 pour BRDF LUT (TASK-04)
```

La crate `image` est déjà présente (utilisée par `ImageImporter`). `rayon`, `bytemuck`, `uuid` aussi.
