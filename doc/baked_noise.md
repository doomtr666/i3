# Baked Noise Textures

## Context

The GTAO and SSSR passes use stochastic sampling that requires high-quality spatial noise with
blue-noise spectral properties to minimize visible structure. The current 128×128 RG8 texture
has insufficient resolution (visible 128-px tiling) and 8-bit quantization that limits temporal
sequence quality. Generating offline at 1024×1024 in 16-bit eliminates both problems. A new
`NoiseAsset` type is introduced (rather than reusing `TextureAsset`) to carry generation
metadata (algorithm, seed, sigma) and to let the renderer distinguish noise from colour textures.

---

## New files

| Path | Role |
|---|---|
| `crates/i3_io/src/noise_asset.rs` | `NoiseAsset`, `NoiseAssetHeader`, `NOISE_ASSET_TYPE` |
| `crates/i3_baker/src/importers/noise_importer.rs` | `NoiseImporter` — procedural generator (no source file) |

## Modified files

| Path | Change |
|---|---|
| `crates/i3_io/src/lib.rs` | `pub mod noise_asset;` + re-export |
| `crates/i3_io/src/texture.rs` | Add `R16G16_UNORM = 13` to `TextureFormat` |
| `crates/i3_baker/src/importers/mod.rs` | `pub mod noise_importer;` + re-export |
| `crates/i3_baker/src/lib.rs` | re-export `NoiseImporter` |
| `crates/i3_baker/src/manifest.rs` | Add `noise: Vec<NoiseManifestEntry>` to `BakeManifest`; wire `NoiseImporter` |
| `crates/i3_baker/Cargo.toml` | Add `rustfft = "6"` |
| `crates/i3_renderer/assets/system.bake.ron` | Add `noise` entries |
| `crates/i3_renderer/assets/shaders/blue_noise.slangh` | `% 1024u`, RG16_UNORM read |
| `crates/i3_renderer/src/passes/gtao.rs` | Remove `blue_noise_index` (GTAO now uses IGN) OR rewire to NoiseAsset bindless index |

---

## 1. `NoiseAsset` (i3_io)

```rust
// crates/i3_io/src/noise_asset.rs

pub const NOISE_ASSET_TYPE: Uuid = uuid!("7f3a1b2c-4d5e-6f70-8192-a3b4c5d6e7f8");

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NoiseAssetHeader {
    pub width:     u32,
    pub height:    u32,
    pub channels:  u32,   // 1 = R, 2 = RG
    pub format:    u32,   // TextureFormat discriminant (R16G16_UNORM = 13)
    pub seed:      u64,
    pub algorithm: u32,   // 0 = WhiteNoise, 1 = BlueNoise
    pub sigma:     f32,   // blue-noise filter width in normalised freq [0,1]
    pub data_size: u64,
    pub _pad:      [u8; 24],
}
// Total: 64 bytes, matches TextureHeader alignment convention.

pub struct NoiseAsset {
    pub header: NoiseAssetHeader,
    pub data:   Vec<u8>,   // width × height × channels × 2 bytes (u16 LE, UNORM)
}

impl Asset for NoiseAsset {
    const ASSET_TYPE_ID: [u8; 16] = *NOISE_ASSET_TYPE.as_bytes();
    fn load(_: &AssetHeader, data: &[u8]) -> Result<Self> { … }
}
```

---

## 2. RON manifest extension

Add field to `BakeManifest`:

```rust
#[serde(default)]
pub noise: Vec<NoiseManifestEntry>,
```

```rust
#[derive(Debug, Deserialize)]
pub struct NoiseManifestEntry {
    pub name:      String,              // asset lookup name, e.g. "blue_noise"
    pub size:      u32,                 // square side in pixels
    #[serde(default)]
    pub seed:      u64,
    pub algorithm: NoiseAlgorithm,
}

#[derive(Debug, Deserialize)]
pub enum NoiseAlgorithm {
    WhiteNoise,
    BlueNoise { sigma: f32 },           // sigma default: 0.5
}
```

**`system.bake.ron` additions:**

```ron
noise: [
    (
        name: "blue_noise",
        size: 1024,
        seed: 42,
        algorithm: BlueNoise(sigma: 0.5),
    ),
    (
        name: "white_noise",
        size: 1024,
        seed: 0,
        algorithm: WhiteNoise,
    ),
]
```

---

## 3. `NoiseImporter` (i3_baker)

The importer is **procedural** (no source file on disk). `ManifestBaker` fabricates a virtual
path `"noise::{name}"` so the per-asset caching key is stable and reproducible.

```rust
pub struct NoiseImporter {
    pub entry: NoiseManifestEntry,
}

impl Importer for NoiseImporter {
    fn name(&self) -> &str { "NoiseImporter" }
    fn source_extensions(&self) -> &[&str] { &[] }

    fn import(&self, _: &Path) -> Result<Box<dyn ImportedData>> {
        Ok(Box::new(NoiseImportedData { entry: self.entry.clone() }))
    }

    fn extract(&self, data: &dyn ImportedData, _: &BakeContext) -> Result<Vec<BakeOutput>> {
        let d = data.as_any().downcast_ref::<NoiseImportedData>().unwrap();
        let pixel_data = match &d.entry.algorithm {
            NoiseAlgorithm::WhiteNoise =>
                generate_white(d.entry.size as usize, d.entry.seed),
            NoiseAlgorithm::BlueNoise { sigma } =>
                generate_blue(d.entry.size as usize, *sigma as f64, d.entry.seed),
        };
        // pack NoiseAssetHeader + pixel_data into BakeOutput.data
        …
    }
}
```

**Cache key** = `postcard::to_allocvec(&entry)` so any parameter change triggers a rebuild.
The virtual source path `format!("noise::{}", entry.name)` is passed to `add_asset_keyed`.

---

## 4. Generation algorithms

### White noise

PCG32 hash per pixel, independent per channel.

```rust
fn generate_white(size: usize, seed: u64) -> Vec<u8> {
    // 2 channels × u16 LE per pixel
    // seed_r = seed, seed_g = seed ^ 0x9e3779b97f4a7c15
    // pcg32(x + width*y) → u16 per channel
}
```

### Blue noise — frequency-domain filter + 2D IFFT

**Algorithm:**

1. For each channel (independent seeds):
   a. Fill a `size × size` complex array `F[u,v]` in frequency domain:
      - Centred radial freq: `r = sqrt(fu² + fv²) / (size/2)` (normalised, 0 → DC)
      - High-pass weight: `w = 1 − exp(−r² / (2σ²))`
      - Random phase: `φ = pcg_f64(u, v, seed) * 2π`
      - `F[u,v] = w · (cos φ, sin φ)`
   b. Apply separable 2D IFFT using `rustfft`:
      - IFFT each column (size FFTs of length size)
      - IFFT each row (size FFTs of length size)
      - Take real part
   c. Normalise to `[0, 65535]` → `u16 LE`

**Complexity:** O(N² log N) with `rustfft`. For N=1024: ~10M ops/channel, sub-second with rayon.

**Dependency:** `rustfft = "6"` in `i3_baker/Cargo.toml`. No runtime dependency.

**Parallelism:** Column IFFTs and row IFFTs parallelised with `rayon::par_iter`.

---

## 5. Shader update (`blue_noise.slangh`)

```slang
float2 blueNoiseBase(uint2 px, uint bn_index)
{
    uint2 coord = px % 1024u;
    // Texture is RG16_UNORM — hardware returns float in [0, 1) automatically
    return bindlessTextures[bn_index].Load(int3(int2(coord), 0)).rg;
}
```

No change to `bnTemporal` or `bnJitter`.

---

## 6. Renderer wiring

`GtaoPass` and `SssrSamplePass` already carry `blue_noise_index: u32` in their push constants.
At startup, the renderer loads the `"blue_noise"` `NoiseAsset`, uploads its data as a
`R16G16_UNORM` image, and passes the bindless index — same path as today's RG8 texture, just
a different asset type UUID for the load call.

---

## Verification

1. `cargo build -p i3_baker` — must compile with `rustfft` dependency.
2. Run the baker on the system bundle — check log output: `"blue_noise"` and `"white_noise"` assets baked.
3. Inspect the catalog (`.i3c`) — two `NoiseAsset` entries present with correct name and UUID.
4. Launch viewer — GTAO and SSSR use the new 1024×1024 texture; verify visually that tiling patterns at 128-px intervals are gone.
5. Toggle GTAO on/off — AO output should be smooth, no repeated halos.
