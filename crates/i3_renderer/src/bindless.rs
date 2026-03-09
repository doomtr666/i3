use i3_gfx::prelude::*;
use std::collections::HashMap;

/// Global manager for Bindless Textures.
///
/// It maps loaded image handles to specific indices in the global descriptor array.
/// Textures are registered dynamically when the scene loads them.
pub struct BindlessManager {
    /// Maps a global physical backend image id to a bindless index
    texture_registry: HashMap<u64, u32>,
    next_index: u32,
    pub max_textures: u32,
    pub bindless_set: u64,
    pub bindless_binding: u32,
    pub default_sampler: SamplerHandle,
}

impl BindlessManager {
    pub fn new(max_textures: u32, default_sampler: SamplerHandle) -> Self {
        Self {
            texture_registry: HashMap::new(),
            next_index: 0,
            max_textures,
            bindless_set: 0, // Will be set by RenderGraph
            bindless_binding: 0,
            default_sampler,
        }
    }

    pub fn register_texture<T: i3_gfx::graph::backend::RenderBackend + ?Sized>(
        &mut self,
        backend: &mut T,
        image: ImageHandle,
    ) -> u32 {
        let physical = backend.resolve_image(image);
        self.register_physical_texture(backend, physical)
    }

    /// Registers a physical backend image into the bindless array.
    pub fn register_physical_texture<T: i3_gfx::graph::backend::RenderBackend + ?Sized>(
        &mut self,
        backend: &mut T,
        image: BackendImage,
    ) -> u32 {
        let physical_id = image.0;

        if let Some(&index) = self.texture_registry.get(&physical_id) {
            return index;
        }

        if self.next_index >= self.max_textures {
            tracing::warn!("Bindless texture array is full! Max: {}", self.max_textures);
            return 0;
        }

        let index = self.next_index;
        self.next_index += 1;

        self.texture_registry.insert(physical_id, index);

        // Update Backend Descriptor
        backend.update_bindless_texture_raw(
            image,
            self.default_sampler,
            index,
            self.bindless_set,
            self.bindless_binding,
        );

        tracing::info!(
            "Registered bindless physical texture {} -> index {}",
            physical_id,
            index
        );

        index
    }

    /// Retrieve the bindless index for a given image, if registered
    pub fn get_texture_index(&self, image: ImageHandle) -> Option<u32> {
        // Technically we need backend to resolve logical to physical
        // Normally scenes just store the physical id. For now we assume image.0.0 maps well enough
        // or we expect callers to manage resolving.
        self.texture_registry.get(&image.0.0).copied()
    }
}
