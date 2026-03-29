//! # Resource Arena - Generational Index System
//!
//! This module implements a resource management system based on **generational indices**.
//! This is a critical pattern for memory safety in a rendering engine.
//!
//! ## Concept
//!
//! Each resource receives a `u64` handle that encodes two pieces of information:
//! - **Index** (lower 32 bits): position in the `slots` array
//! - **Generation** (upper 32 bits): counter incremented each time the slot is reused
//!
//! ```text
//! Handle u64 = [Generation (32 bits) | Index (32 bits)]
//! ```
//!
//! ## Safety
//!
//! When a resource is deleted, its slot is marked as `Free` and its generation is incremented.
//! If an old handle is used after deletion, the generation will no longer match and
//! access will return `None` instead of corrupting memory.
//!
//! ## Performance
//!
//! - Insertion: O(1) amortized (reuses free slots via a linked list)
//! - Access: O(1) (direct indexing into the vector)
//! - Removal: O(1)
//!
//! ## Typical Usage
//!
//! ```ignore
//! let arena: ResourceArena<PhysicalImage> = ResourceArena::new();
//! let handle = arena.insert(physical_image);  // Returns u64
//! let image = arena.get(handle);              // Option<&PhysicalImage>
//! arena.remove(handle);                       // Frees the slot
//! ```

use ash::vk;
use i3_gfx::graph::types::*;

/// Physical representation of a Vulkan pipeline.
///
/// Contains the raw Vulkan handle and metadata needed for binding.
#[derive(Clone)]
pub struct PhysicalPipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub bind_point: vk::PipelineBindPoint,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub physical_id: u64,
}

/// Physical representation of a Vulkan image.
///
/// Contains the image, its view, and synchronization state.
/// The `last_*` fields track the last known state for automatic
/// memory barrier generation.
pub struct PhysicalImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Option<vk_mem::Allocation>,
    pub desc: ImageDesc,
    pub format: vk::Format,

    /// Last known layout (for image transitions)
    pub last_layout: vk::ImageLayout,
    /// Last access flags (for memory barriers)
    pub last_access: vk::AccessFlags2,
    /// Last pipeline stage (for memory barriers)
    pub last_stage: vk::PipelineStageFlags2,
    /// Frame count at last write (for load op)
    pub last_write_frame: u64,
    /// Last queue family that owned this image (for ownership transfers)
    pub last_queue_family: u32,
    /// Whether this is a swapchain image (presented to screen)
    pub is_swapchain: bool,
    /// Whether this resource was created with CONCURRENT sharing mode (no ownership transfer needed)
    pub concurrent: bool,
}

/// Physical representation of a Vulkan buffer.
///
/// Similar to PhysicalImage but for buffers.
pub struct PhysicalBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<vk_mem::Allocation>,
    pub desc: BufferDesc,

    // Synchronization state (Sync2)
    pub last_access: vk::AccessFlags2,
    pub last_stage: vk::PipelineStageFlags2,
    pub last_queue_family: u32,
    /// Whether this resource was created with CONCURRENT sharing mode (no ownership transfer needed)
    pub concurrent: bool,
}

/// Slot in the arena - either occupied with data or free with generation tracking.
enum Slot<T> {
    Occupied {
        data: T,
        generation: u32,
    },
    Free {
        next_free: Option<u32>,
        generation: u32,
    },
}

/// Arène de ressources générique avec indices générationnels pour un accès sûr via handles.
///
/// Les ressources sont insérées et reçoivent un handle u64 encodant à la fois un index et une génération.
/// Cela permet de détecter les use-after-free : si la génération d'un handle ne correspond pas à
/// celle du slot, l'accès retourne None.
pub struct ResourceArena<T> {
    slots: Vec<Slot<T>>,
    free_head: Option<u32>,
}

impl<T> ResourceArena<T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::with_capacity(256),
            free_head: None,
        }
    }

    /// Insert a resource and return its generational handle.
    pub fn insert(&mut self, data: T) -> u64 {
        if let Some(index) = self.free_head {
            let slot = &mut self.slots[index as usize];
            if let Slot::Free {
                next_free,
                generation,
            } = *slot
            {
                let generation_val = generation;
                *slot = Slot::Occupied {
                    data,
                    generation: generation_val,
                };
                self.free_head = next_free;
                return ((generation_val as u64) << 32) | (index as u64);
            }
        }

        let index = self.slots.len() as u32;
        let generation = 1u32;
        self.slots.push(Slot::Occupied { data, generation });
        ((generation as u64) << 32) | (index as u64)
    }

    /// Get a reference to a resource by handle, returning None if the handle is stale.
    pub fn get(&self, id: u64) -> Option<&T> {
        let index = (id & 0xFFFFFFFF) as usize;
        let generation_val = (id >> 32) as u32;
        if let Some(Slot::Occupied { data, generation }) = self.slots.get(index) {
            if *generation == generation_val {
                return Some(data);
            }
        }
        None
    }

    /// Get a mutable reference to a resource by handle.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut T> {
        let index = (id & 0xFFFFFFFF) as usize;
        let generation_val = (id >> 32) as u32;
        if let Some(Slot::Occupied { data, generation }) = self.slots.get_mut(index) {
            if *generation == generation_val {
                return Some(data);
            }
        }
        None
    }

    /// Remove a resource by handle, returning it if the handle is valid.
    pub fn remove(&mut self, id: u64) -> Option<T> {
        let index = (id & 0xFFFFFFFF) as usize;
        let generation_val = (id >> 32) as u32;
        if index >= self.slots.len() {
            return None;
        }

        match self.slots[index] {
            Slot::Occupied { generation, .. } if generation == generation_val => {
                let old_slot = std::mem::replace(
                    &mut self.slots[index],
                    Slot::Free {
                        next_free: self.free_head,
                        generation: generation + 1,
                    },
                );
                self.free_head = Some(index as u32);
                if let Slot::Occupied { data, .. } = old_slot {
                    Some(data)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Iterate over all occupied slots with mutable access.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u64, &mut T)> {
        self.slots.iter_mut().enumerate().filter_map(|(i, slot)| {
            if let Slot::Occupied { data, generation } = slot {
                let id = ((*generation as u64) << 32) | (i as u64);
                Some((id, data))
            } else {
                None
            }
        })
    }

    /// Iterate over all occupied slots with immutable access.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| {
            if let Slot::Occupied { data, generation } = slot {
                let id = ((*generation as u64) << 32) | (i as u64);
                Some((id, data))
            } else {
                None
            }
        })
    }

    /// Get all occupied resource IDs.
    pub fn ids(&self) -> Vec<u64> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                if let Slot::Occupied { generation, .. } = slot {
                    Some(((*generation as u64) << 32) | (i as u64))
                } else {
                    None
                }
            })
            .collect()
    }
}
